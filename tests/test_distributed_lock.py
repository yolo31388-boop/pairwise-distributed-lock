"""
游戏内分布式锁与协调系统 - 验收测试
运行方式：python -m pytest tests/test_distributed_lock.py -q
共 15 个测试用例
"""
import pytest
import sys
import os
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distributed.lock import DistributedLock


class TestDistributedLock:
    def setup_method(self):
        self.system = DistributedLock()

    # ---- acquire_lock：互斥 / 超时 / 重入 ----

    def test_case_01(self):
        """锁真实互斥：两个玩家同时买同一道具，库存不会变负数"""
        stock = {"count": 1}
        results = []

        def buy(player):
            if self.system.acquire_lock("item:1001", owner=player, timeout=2.0):
                try:
                    if stock["count"] > 0:
                        time.sleep(0.01)  # 模拟业务处理耗时
                        stock["count"] -= 1
                        results.append(player)
                finally:
                    self.system.release_lock("item:1001", owner=player)

        threads = [threading.Thread(target=buy, args=(f"player{i}",))
                   for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        assert stock["count"] == 0  # 库存不能为负
        assert len(results) == 1    # 只有一个玩家买到

    def test_case_02(self):
        """锁获取有超时：拿不到锁在 timeout 后返回 False，不会永远等"""
        assert self.system.acquire_lock("item:1001", owner="playerA") is True
        start = time.monotonic()
        ok = self.system.acquire_lock("item:1001", owner="playerB", timeout=0.2)
        elapsed = time.monotonic() - start
        assert ok is False
        assert 0.2 <= elapsed < 1.0

    def test_case_03(self):
        """锁可重入：同一线程/持有者再次获取不会死锁"""
        assert self.system.acquire_lock("item:1001", owner="playerA") is True
        # 同一持有者再次获取，立即成功
        assert self.system.acquire_lock("item:1001", owner="playerA",
                                        timeout=0.1) is True
        # 重入计数：只释放一次锁仍被持有
        assert self.system.release_lock("item:1001", owner="playerA") is True
        assert self.system.is_locked("item:1001") is True
        assert self.system.release_lock("item:1001", owner="playerA") is True
        assert self.system.is_locked("item:1001") is False

    # ---- release_lock：真正释放 / 持有者检查 / 通知等待者 ----

    def test_case_04(self):
        """锁释放真正生效：用完释放后其他玩家能拿到锁"""
        assert self.system.acquire_lock("item:1001", owner="playerA") is True
        assert self.system.release_lock("item:1001", owner="playerA") is True
        assert self.system.acquire_lock("item:1001", owner="playerB",
                                        timeout=0.2) is True

    def test_case_05(self):
        """释放有检查：不是锁的持有者不能释放"""
        assert self.system.acquire_lock("item:1001", owner="playerA") is True
        # playerB 不是持有者，释放被拒绝
        assert self.system.release_lock("item:1001", owner="playerB") is False
        # 锁仍被 playerA 持有
        assert self.system.is_locked("item:1001") is True
        assert self.system.get_lock_owner("item:1001") == "playerA"

    def test_case_06(self):
        """释放有通知：锁释放后等待中的线程被唤醒并拿到锁"""
        assert self.system.acquire_lock("item:1001", owner="playerA") is True
        acquired = []

        def waiter():
            if self.system.acquire_lock("item:1001", owner="playerB",
                                        timeout=2.0):
                acquired.append("playerB")

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.1)  # 确保 playerB 已在等待
        self.system.release_lock("item:1001", owner="playerA")
        t.join(timeout=2.0)
        assert acquired == ["playerB"]  # 等待者被通知并获取成功

    # ---- leader_election：唯一leader / 任期 / 投票 ----

    def test_case_07(self):
        """选举唯一leader：多个节点中只有一个成为leader"""
        nodes = ["node1", "node2", "node3"]
        leader = self.system.leader_election(nodes)
        assert leader in nodes
        assert self.system.get_leader() == leader
        # 全局只有一个 leader
        assert sum(1 for n in nodes if self.system.is_leader(n)) == 1

    def test_case_08(self):
        """选举有任期：每次选举任期递增，过期任期的选举被拒绝"""
        nodes = ["node1", "node2", "node3"]
        self.system.leader_election(nodes)
        term1 = self.system.get_term()
        self.system.leader_election(nodes)
        term2 = self.system.get_term()
        assert term2 > term1  # leader 会随新选举更换任期
        # 过期任期的选举请求被拒绝，不影响当前任期
        self.system.leader_election(nodes, term=term1)
        assert self.system.get_term() == term2

    def test_case_09(self):
        """选举有投票：获得多数票的候选人才是leader，不是谁先启动谁当"""
        nodes = ["node1", "node2", "node3"]
        votes = {"node1": "node2", "node2": "node2", "node3": "node1"}
        leader = self.system.leader_election(nodes, votes=votes)
        assert leader == "node2"  # 2/3 多数票当选
        # 没有任何候选人获得多数票时，选举失败，不产生leader
        split = {"node1": "node1", "node2": "node2", "node3": "node3"}
        assert self.system.leader_election(nodes, votes=split) is None
        assert self.system.get_leader() is None

    # ---- distributed_queue：不重复消费 / 确认 / 顺序 ----

    def test_case_10(self):
        """队列不重复消费：每条消息只被一个消费者拿到"""
        queue = self.system.distributed_queue("tasks")
        for i in range(20):
            queue.send(f"msg{i}")
        consumed = []

        def consumer():
            while True:
                msg = queue.receive(timeout=0.2)
                if msg is None:
                    break
                consumed.append(msg.payload)
                queue.ack(msg.message_id)

        threads = [threading.Thread(target=consumer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        assert sorted(consumed) == sorted(f"msg{i}" for i in range(20))
        assert len(consumed) == len(set(consumed))  # 没有重复消费

    def test_case_11(self):
        """队列有确认：ack后消息移除；消费失败nack后消息不丢失可重新投递"""
        queue = self.system.distributed_queue("tasks")
        queue.send("msg1")
        msg = queue.receive()
        assert msg.payload == "msg1"
        # 消费失败：nack 后消息回到队列，不丢失
        assert queue.nack(msg.message_id) is True
        again = queue.receive()
        assert again is not None
        assert again.payload == "msg1"
        # 消费成功：ack 后消息彻底移除，不再投递
        assert queue.ack(again.message_id) is True
        assert queue.receive() is None

    def test_case_12(self):
        """队列有顺序：消息按发送顺序（FIFO）被消费"""
        queue = self.system.distributed_queue("tasks")
        for i in range(10):
            queue.send(f"msg{i}")
        received = []
        while True:
            msg = queue.receive()
            if msg is None:
                break
            received.append(msg.payload)
            queue.ack(msg.message_id)
        assert received == [f"msg{i}" for i in range(10)]

    # ---- distributed_barrier：同步 / 超时 / 通知 ----

    def test_case_13(self):
        """屏障同步：所有节点到齐之前，先到的节点必须等待"""
        barrier = self.system.distributed_barrier("phase1", parties=3)
        passed = []

        def node(node_id):
            if barrier.wait(node_id, timeout=2.0):
                passed.append((node_id, len(passed)))

        t1 = threading.Thread(target=node, args=("node1",))
        t1.start()
        time.sleep(0.2)  # node1 先到，必须等待
        assert passed == []
        t2 = threading.Thread(target=node, args=("node2",))
        t2.start()
        time.sleep(0.2)  # node2 到了，仍然不够，继续等
        assert passed == []
        node("node3")  # 全部到齐
        t1.join(timeout=2.0)
        t2.join(timeout=2.0)
        assert len(passed) == 3

    def test_case_14(self):
        """屏障有超时：节点没到齐时不会永远等，超时返回 False"""
        barrier = self.system.distributed_barrier("phase1", parties=3)
        start = time.monotonic()
        ok = barrier.wait("node1", timeout=0.2)  # 只有1个节点到达
        elapsed = time.monotonic() - start
        assert ok is False
        assert 0.2 <= elapsed < 1.0

    def test_case_15(self):
        """屏障有通知：最后一个节点到齐时，所有等待节点都被唤醒"""
        barrier = self.system.distributed_barrier("phase1", parties=3)
        released = []
        entered = threading.Event()
        count = [0]

        def node(node_id):
            count[0] += 1
            if count[0] == 2:
                entered.set()
            if barrier.wait(node_id, timeout=2.0):
                released.append(node_id)

        threads = [threading.Thread(target=node, args=(f"node{i}",))
                   for i in (1, 2)]
        for t in threads:
            t.start()
        entered.wait(timeout=2.0)
        time.sleep(0.1)
        assert released == []  # 两个节点都在等待中
        barrier.wait("node3", timeout=2.0)  # 最后一人到齐，触发通知
        for t in threads:
            t.join(timeout=2.0)
        assert sorted(released) == ["node1", "node2"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
