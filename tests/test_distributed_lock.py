"""
游戏内分布式锁与协调系统 - 验收测试
运行方式：python -m pytest tests/test_distributed_lock.py -q
共 15 个测试用例：
01-05 可重入分布式锁（互斥 / 超时 / 重入 / 持有者校验 / 释放通知）
06-09 领导者选举（任期+多数派投票 / 唯一leader / 平票重选 / 旧任期拒绝）
10-12 分布式队列（FIFO顺序 / ack确认不重复 / 失败重投）
13-15 分布式屏障（到齐通知 / 超时 / 新一代复用）
"""
import threading
import time

import pytest

from distributed.lock import (
    BarrierBrokenError,
    DistributedBarrier,
    DistributedLock,
    DistributedQueue,
    LeaderElection,
    LockOwnershipError,
    LockTimeoutError,
)


class TestDistributedLock:
    def setup_method(self):
        self.system = DistributedLock()

    def test_case_01(self):
        """锁必须真正互斥：持有者之外的线程拿不到锁。"""
        lock = self.system
        assert lock.try_acquire() is True
        assert lock.locked is True

        results = []

        def other():
            results.append(lock.try_acquire())

        t = threading.Thread(target=other)
        t.start()
        t.join(timeout=1.0)
        assert results == [False]

        lock.release()
        assert lock.locked is False

    def test_case_02(self):
        """acquire 必须有超时：拿不到锁时抛 LockTimeoutError，而不是死等。"""
        lock = self.system
        lock.acquire()
        start = time.monotonic()
        with pytest.raises(LockTimeoutError):
            lock.acquire(timeout=0.2, owner=999)
        elapsed = time.monotonic() - start
        assert 0.2 <= elapsed < 1.0
        assert lock.owner == threading.get_ident()

    def test_case_03(self):
        """锁必须可重入：同一持有者重复 acquire 不死锁，按次数 release。"""
        lock = self.system
        assert lock.acquire() is True
        assert lock.acquire() is True
        assert lock.acquire() is True
        assert lock.locked is True

        lock.release()
        assert lock.locked is True
        lock.release()
        assert lock.locked is True
        lock.release()
        assert lock.locked is False

    def test_case_04(self):
        """release 必须校验持有者：非持有者/无人持有时释放都要报错。"""
        lock = self.system
        with pytest.raises(LockOwnershipError):
            lock.release()

        lock.acquire()
        with pytest.raises(LockOwnershipError):
            lock.release(owner=999)
        assert lock.locked is True

        lock.release()
        assert lock.locked is False

    def test_case_05(self):
        """release 必须通知等待者：锁释放后等待线程应被唤醒并拿到锁。"""
        lock = self.system
        lock.acquire()
        acquired = threading.Event()

        def waiter():
            lock.acquire(timeout=2.0)
            acquired.set()
            lock.release()

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.15)
        assert not acquired.is_set()

        lock.release()
        assert acquired.wait(timeout=1.0) is True
        t.join(timeout=1.0)
        assert lock.locked is False

    def test_case_06(self):
        """选举必须有任期和多数派投票：票不够不当选，每轮 term 单调递增。"""
        election = LeaderElection(["node-a", "node-b", "node-c"])
        term = election.start_election("node-a")
        assert term == 1
        assert election.leader is None

        assert election.request_vote("node-a", term) is True
        assert election.leader is None
        assert election.request_vote("node-a", term) is True
        assert election.leader == "node-a"
        assert election.is_leader("node-a") is True
        assert election.is_leader("node-b") is False

        term2 = election.start_election("node-b")
        assert term2 == 2
        assert election.leader is None

    def test_case_07(self):
        """多个节点并发选举时，同一时刻只能有一个 leader。"""
        nodes = ["n1", "n2", "n3", "n4", "n5"]
        election = LeaderElection(nodes)
        winners = []
        winners_lock = threading.Lock()

        def campaign(node):
            for _ in range(3):
                if election.try_become_leader(node):
                    with winners_lock:
                        winners.append((election.current_term, node))
                    election.step_down()

        threads = [threading.Thread(target=campaign, args=(n,)) for n in nodes]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        leader_nodes = [n for n in nodes if election.is_leader(n) or
                        election.leader == n]
        assert election.leader is None or len(leader_nodes) == 1
        terms = [term for term, _ in winners]
        assert len(terms) == len(set(terms))

    def test_case_08(self):
        """平票/票不足无人当选时，可在更高任期重新选举并成功。"""
        election = LeaderElection(["a", "b", "c"])
        term = election.start_election("a")
        election.request_vote("a", term)
        election.request_vote("b", term)
        assert election.leader is None

        assert election.try_become_leader("c") is True
        assert election.current_term > term
        assert election.leader == "c"

    def test_case_09(self):
        """旧任期的投票请求必须被拒绝，term 只能单调递增。"""
        election = LeaderElection(["a", "b", "c"])
        old_term = election.start_election("a")
        election.start_election("b")

        assert election.request_vote("a", old_term) is False
        assert election.leader is None
        assert election.request_vote("a", 9999) is False

    def test_case_10(self):
        """队列必须保证顺序：消息严格按发送顺序 FIFO 消费。"""
        queue = DistributedQueue()
        for item in ["sword", "shield", "potion"]:
            queue.put(item)

        consumed = []
        for _ in range(3):
            result = queue.get(timeout=0.2)
            assert result is not None
            _, payload = result
            consumed.append(payload)
        assert consumed == ["sword", "shield", "potion"]
        assert queue.get(timeout=0.1) is None

    def test_case_11(self):
        """队列必须有确认机制：未 ack 的消息不会被重复投递给其他消费者。"""
        queue = DistributedQueue()
        queue.put("rare-item")

        delivery_id, payload = queue.get(timeout=0.2)
        assert payload == "rare-item"
        assert queue.inflight_count() == 1
        assert queue.get(timeout=0.1) is None

        with pytest.raises(KeyError):
            queue.ack("not-a-real-id")
        queue.ack(delivery_id)
        assert queue.empty() is True
        with pytest.raises(KeyError):
            queue.ack(delivery_id)

    def test_case_12(self):
        """消费失败（nack 或 ack 超时）的消息必须按原顺序重新投递，不丢消息。"""
        queue = DistributedQueue(visibility_timeout=0.05)
        queue.put("msg-1")

        delivery_id, payload = queue.get(timeout=0.2)
        assert payload == "msg-1"
        queue.nack(delivery_id)

        redelivered = queue.get(timeout=0.5)
        assert redelivered is not None
        second_id, second_payload = redelivered
        assert second_payload == "msg-1"
        assert second_id != delivery_id

        redelivered_by_timeout = queue.get(timeout=0.5)
        assert redelivered_by_timeout is not None
        timeout_id, timeout_payload = redelivered_by_timeout
        assert timeout_payload == "msg-1"
        time.sleep(0.1)
        again = queue.get(timeout=0.5)
        assert again is not None and again[1] == "msg-1"
        queue.ack(again[0])
        assert queue.empty() is True

    def test_case_13(self):
        """屏障必须同步所有节点：到齐前无人放行，到齐后统一通知放行。"""
        parties = 4
        trips = []
        trips_lock = threading.Lock()

        def on_trip():
            with trips_lock:
                trips.append(barrier.generation)

        barrier = DistributedBarrier(parties, action=on_trip)
        release_generations = []
        gen_lock = threading.Lock()
        all_done = threading.Event()

        def node():
            gen = barrier.wait(timeout=2.0)
            with gen_lock:
                release_generations.append(gen)
            if len(release_generations) == parties:
                all_done.set()

        threads = [threading.Thread(target=node) for _ in range(parties - 1)]
        for t in threads:
            t.start()
        time.sleep(0.2)
        assert barrier.waiting == parties - 1
        assert not all_done.is_set()

        last = threading.Thread(target=node)
        last.start()
        assert all_done.wait(timeout=1.0) is True
        last.join(timeout=1.0)
        for t in threads:
            t.join(timeout=1.0)
        assert release_generations == [0] * parties
        assert trips == [0]
        assert barrier.waiting == 0

    def test_case_14(self):
        """屏障必须有超时：节点始终到不齐时抛 BarrierBrokenError，而不是死等。"""
        barrier = DistributedBarrier(3)
        start = time.monotonic()
        with pytest.raises(BarrierBrokenError):
            barrier.wait(timeout=0.2)
        elapsed = time.monotonic() - start
        assert 0.2 <= elapsed < 1.0
        assert barrier.waiting == 0
        assert barrier.broken is True

    def test_case_15(self):
        """屏障放行后必须开新一代：generation 递增，可重复同步下一批节点。"""
        barrier = DistributedBarrier(2)

        def round_trip():
            return barrier.wait(timeout=1.0)

        t1 = threading.Thread(target=round_trip)
        t1.start()
        time.sleep(0.1)
        gen0 = round_trip()
        t1.join(timeout=1.0)
        assert gen0 == 0
        assert barrier.generation == 1
        assert barrier.waiting == 0

        t2 = threading.Thread(target=round_trip)
        t2.start()
        time.sleep(0.1)
        assert barrier.waiting == 1
        gen1 = round_trip()
        t2.join(timeout=1.0)
        assert gen1 == 1
        assert barrier.generation == 2
        assert barrier.waiting == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
