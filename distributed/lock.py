"""
游戏内分布式锁与协调系统 - 核心模块

包含四个协调原语（纯标准库实现）：

- DistributedLock  : 可重入互斥锁（获取超时 / 同线程重入 / 释放校验 / 通知等待者）
- LeaderElection   : 多数派领导者选举（单调递增任期 term / 每任期每节点一票）
- DistributedQueue : FIFO 消息队列（顺序投递 / ack 确认 / 失败重投 / 不重复消费）
- DistributedBarrier : 分布式屏障（到齐通知 / 超时返回 / 自动开新一代）
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


class LockTimeoutError(TimeoutError):
    """在给定时间内未能获取锁。"""


class LockOwnershipError(RuntimeError):
    """释放锁的线程并不是锁的持有者。"""


class NotLeaderError(RuntimeError):
    """当前节点不是 leader，无法执行该操作。"""


class BarrierBrokenError(RuntimeError):
    """屏障在超时时间内未凑齐所有节点。"""


def _default_owner() -> int:
    return threading.get_ident()


class DistributedLock:
    """可重入的分布式互斥锁（进程内模拟实现）。

    - acquire 支持 timeout：超时抛出 LockTimeoutError，绝不无限等待。
    - 同线程可重入：重复 acquire 需要对应次数的 release。
    - release 校验持有者：非持有者释放抛出 LockOwnershipError。
    - release 后通过 Condition 通知所有等待线程。
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._cond = threading.Condition(threading.RLock())
        self._owner: Optional[int] = None
        self._count = 0
        self._state: Dict[str, int] = {}
        self._history: List[Tuple[float, str, int]] = []

    def acquire(self, timeout: Optional[float] = None,
                owner: Optional[int] = None) -> bool:
        """获取锁，最多等待 timeout 秒；timeout=None 表示一直等待。

        获取成功返回 True；超时抛出 LockTimeoutError。
        """
        me = owner if owner is not None else _default_owner()
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._owner is not None and self._owner != me:
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise LockTimeoutError(
                            "failed to acquire lock within timeout")
                self._cond.wait(remaining)
            if self._owner is None:
                self._owner = me
            self._count += 1
            self._state["owner"] = me
            self._state["count"] = self._count
            self._history.append((time.time(), "acquire", me))
            return True

    def try_acquire(self, owner: Optional[int] = None) -> bool:
        """非阻塞尝试获取锁，成功 True / 锁被占用 False。"""
        me = owner if owner is not None else _default_owner()
        with self._cond:
            if self._owner is not None and self._owner != me:
                return False
            if self._owner is None:
                self._owner = me
            self._count += 1
            self._state["owner"] = me
            self._state["count"] = self._count
            self._history.append((time.time(), "try_acquire", me))
            return True

    def release(self, owner: Optional[int] = None) -> None:
        """释放一次锁；只有持有者可以释放。

        重入次数归零后真正释放，并通知所有等待线程。
        """
        me = owner if owner is not None else _default_owner()
        with self._cond:
            if self._owner is None:
                raise LockOwnershipError("lock is not held by anyone")
            if self._owner != me:
                raise LockOwnershipError(
                    f"lock is held by {self._owner}, not {me}")
            self._count -= 1
            if self._count == 0:
                self._owner = None
                self._state.pop("owner", None)
                self._state["count"] = 0
                self._history.append((time.time(), "release", me))
                self._cond.notify_all()
            else:
                self._state["count"] = self._count

    @property
    def owner(self) -> Optional[int]:
        with self._cond:
            return self._owner

    @property
    def locked(self) -> bool:
        with self._cond:
            return self._owner is not None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False

    def update(self, dt: float) -> None:
        pass

    def reset(self) -> None:
        with self._cond:
            self._owner = None
            self._count = 0
            self._state = {}
            self._history = []
            self._cond.notify_all()


class LeaderElection:
    """基于任期(term)与多数派投票的领导者选举。

    模拟 Raft 风格的选举：

    - 每次开始新一轮选举 term 单调递增；
    - 每个节点在每个 term 只投一票（先到先得）；
    - 获得多数节点投票才能成为唯一 leader；
    - 旧任期的投票不会影响新一轮选举，不会出现多个 leader。
    """

    def __init__(self, nodes: List[str]):
        if not nodes:
            raise ValueError("nodes must not be empty")
        self.nodes = list(nodes)
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self.current_term = 0
        self.leader: Optional[str] = None
        self._voted_for: Dict[int, Dict[str, Optional[str]]] = {}
        self._term_votes: Dict[int, set] = {}

    def start_election(self, candidate: str) -> int:
        """候选人发起新一轮选举，返回新的任期号。"""
        with self._lock:
            if candidate not in self.nodes:
                raise ValueError(f"unknown node: {candidate}")
            self.current_term += 1
            term = self.current_term
            self.leader = None
            self._term_votes[term] = set()
            self._voted_for[term] = {node: None for node in self.nodes}
            self._cond.notify_all()
            return term

    def request_vote(self, candidate: str, term: int) -> bool:
        """节点集群对该候选人的投票请求进行投票。

        候选人会先投给自己。每个节点每个 term 只投一票。
        """
        with self._lock:
            if candidate not in self.nodes:
                raise ValueError(f"unknown node: {candidate}")
            if term <= 0 or term != self.current_term:
                return False
            if self.leader is not None and self.current_term == term:
                return False
            votes = self._voted_for[term]
            for voter in self.nodes:
                if votes[voter] is None:
                    votes[voter] = candidate
                    self._term_votes[term].add(candidate)
                    break
            won = len([v for v in votes.values() if v == candidate])
            if won > len(self.nodes) // 2:
                self.leader = candidate
                self._cond.notify_all()
            return True

    def try_become_leader(self, candidate: str,
                          term: Optional[int] = None) -> bool:
        """发起一轮选举并向全部节点拉票；获得多数票则成为 leader。"""
        with self._lock:
            new_term = self.start_election(candidate)
            if term is not None:
                new_term = term
            majority = len(self.nodes) // 2 + 1
            granted = 0
            votes = self._voted_for[new_term]
            for voter in self.nodes:
                if votes[voter] is None:
                    votes[voter] = candidate
                    granted += 1
                    if granted >= majority:
                        break
            if granted >= majority:
                self._term_votes[new_term].add(candidate)
                self.leader = candidate
                self._cond.notify_all()
                return True
            return False

    def elect(self, candidate: str) -> bool:
        """try_become_leader 的语义化别名。"""
        return self.try_become_leader(candidate)

    def is_leader(self, node: str) -> bool:
        with self._lock:
            return self.leader == node

    def step_down(self) -> None:
        """现任 leader 主动下台，允许其它节点发起新选举。"""
        with self._lock:
            self.leader = None
            self._cond.notify_all()

    def wait_for_leader(self, timeout: Optional[float] = None) -> Optional[str]:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._lock:
            while self.leader is None:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._cond.wait(remaining)
            return self.leader


@dataclass
class _QueueEntry:
    seq: int
    payload: object
    delivery_count: int = 0


class DistributedQueue:
    """FIFO 分布式消息队列（带确认机制）。

    - put/get 严格按发送顺序（FIFO）投递；
    - get 返回 (delivery_id, payload)，消息转为 in-flight，不会重复投给别人；
    - 必须显式 ack 才从队列移除；
    - nack / 超时未 ack 的消息按原顺序重新投递，不丢消息；
    - visibility_timeout 为 0 时消息立刻可被再次投递（模拟重投场景）。
    """

    def __init__(self, visibility_timeout: float = 30.0):
        self.visibility_timeout = visibility_timeout
        self._cond = threading.Condition(threading.RLock())
        self._ready: List[_QueueEntry] = []
        self._inflight: Dict[str, Tuple[_QueueEntry, float]] = {}
        self._seq = 0

    def put(self, payload: object) -> None:
        with self._cond:
            self._seq += 1
            self._ready.append(_QueueEntry(self._seq, payload))
            self._cond.notify()

    def _reclaim_expired(self, now: float) -> None:
        expired = []
        for delivery_id, (entry, deadline) in self._inflight.items():
            if deadline <= now:
                expired.append((delivery_id, entry))
        for delivery_id, entry in expired:
            del self._inflight[delivery_id]
            self._ready.append(entry)
        if expired:
            self._ready.sort(key=lambda e: e.seq)

    def get(self, timeout: Optional[float] = None
            ) -> Optional[Tuple[str, object]]:
        """阻塞获取下一条消息；超时无消息返回 None。"""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while True:
                now = time.monotonic()
                self._reclaim_expired(now)
                if self._ready:
                    entry = self._ready.pop(0)
                    entry.delivery_count += 1
                    delivery_id = uuid.uuid4().hex
                    self._inflight[delivery_id] = (
                        entry, now + self.visibility_timeout)
                    return delivery_id, entry.payload
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._cond.wait(remaining)
                else:
                    self._cond.wait()

    def ack(self, delivery_id: str) -> None:
        """确认消费成功；重复 ack / 未知 id 抛 KeyError。"""
        with self._cond:
            if delivery_id not in self._inflight:
                raise KeyError(f"unknown or already-acked delivery: {delivery_id}")
            del self._inflight[delivery_id]
            self._cond.notify_all()

    def nack(self, delivery_id: str, requeue: bool = True) -> None:
        """消费失败：消息重新入队（保持原顺序）；requeue=False 则丢弃。"""
        with self._cond:
            if delivery_id not in self._inflight:
                raise KeyError(f"unknown or already-acked delivery: {delivery_id}")
            entry, _ = self._inflight.pop(delivery_id)
            if requeue:
                self._ready.append(entry)
                self._ready.sort(key=lambda e: e.seq)
            self._cond.notify_all()

    def qsize(self) -> int:
        with self._cond:
            self._reclaim_expired(time.monotonic())
            return len(self._ready)

    def inflight_count(self) -> int:
        with self._cond:
            self._reclaim_expired(time.monotonic())
            return len(self._inflight)

    def empty(self) -> bool:
        with self._cond:
            self._reclaim_expired(time.monotonic())
            return not self._ready and not self._inflight


class DistributedBarrier:
    """分布式屏障：所有节点到齐后一起放行，并自动开新一代。

    - wait 阻塞直到 parties 个节点全部到达；
    - 最后一个到达的节点触发通知，所有等待者同时放行；
    - wait(timeout=...) 超时未到齐抛出 BarrierBrokenError；
    - 放行后 generation 加 1，可循环使用。
    """

    def __init__(self, parties: int,
                 action: Optional[Callable[[], None]] = None):
        if parties < 1:
            raise ValueError("parties must be >= 1")
        self.parties = parties
        self._action = action
        self._cond = threading.Condition(threading.RLock())
        self.generation = 0
        self._waiting = 0
        self.broken = False

    def wait(self, timeout: Optional[float] = None) -> int:
        """到达屏障并等待其它节点，返回离开时的 generation。"""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            gen = self.generation
            self._waiting += 1
            if self._waiting == self.parties:
                if self._action is not None:
                    self._action()
                self._waiting = 0
                self.generation += 1
                self._cond.notify_all()
                return self.generation - 1
            while True:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    self.broken = True
                    self._waiting -= 1
                    self._cond.notify_all()
                    raise BarrierBrokenError(
                        f"barrier timed out: only {self._waiting} "
                        f"of {self.parties} parties arrived")
                self._cond.wait(remaining)
                if self.generation != gen:
                    return gen
                if self.broken:
                    raise BarrierBrokenError("barrier broken by peer timeout")

    def reset(self) -> None:
        with self._cond:
            self._waiting = 0
            self.broken = False
            self.generation += 1
            self._cond.notify_all()

    @property
    def waiting(self) -> int:
        with self._cond:
            return self._waiting
