"""
游戏内分布式锁与协调系统 - 核心模块

修复内容：
- acquire_lock():       真实互斥 + 获取超时 + 同持有者可重入
- release_lock():       持有者检查 + 真正释放 + 通知等待者
- leader_election():    任期(term)单调递增 + 多数派投票 + 全局唯一leader
- distributed_queue():  FIFO顺序 + 消费确认(ack) + 未确认不丢失/不重复消费
- distributed_barrier(): 节点到齐通知 + 等待超时
"""
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional


@dataclass
class Config:
    pass


@dataclass
class Message:
    """分布式队列消息。"""

    message_id: str
    payload: Any
    queue: str
    timestamp: float = field(default_factory=time.time)
    deliveries: int = 0

    @property
    def id(self) -> str:
        return self.message_id

    @property
    def data(self) -> Any:
        return self.payload

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class _LockEntry:
    """单个命名锁的状态：持有者 + 重入计数 + 等待者条件变量。"""

    def __init__(self) -> None:
        self.owner: Optional[str] = None
        self.count: int = 0
        self.cond = threading.Condition()


class DistributedQueue:
    """分布式队列：FIFO顺序 + 确认机制(ack) + 失败不丢失。"""

    def __init__(self, name: str, visibility_timeout: Optional[float] = None):
        self.name = name
        self.visibility_timeout = visibility_timeout
        self._cond = threading.Condition()
        self._pending: Deque[Message] = deque()
        self._inflight: Dict[str, Message] = {}
        self._inflight_since: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # 发送：追加到队尾，保证先进先出
    # ------------------------------------------------------------------ #
    def send(self, payload: Any) -> str:
        with self._cond:
            message = Message(message_id="msg-" + uuid.uuid4().hex,
                              payload=payload, queue=self.name)
            self._pending.append(message)
            self._cond.notify()
            return message.message_id

    # 兼容别名
    put = send
    enqueue = send

    # ------------------------------------------------------------------ #
    # 接收：从队头取出，进入“待确认”状态；未确认不会丢也不会被别人重复消费
    # ------------------------------------------------------------------ #
    def receive(self, timeout: Optional[float] = None) -> Optional[Message]:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while True:
                self._requeue_expired()
                if self._pending:
                    message = self._pending.popleft()
                    message.deliveries += 1
                    self._inflight[message.message_id] = message
                    self._inflight_since[message.message_id] = time.monotonic()
                    return message
                if timeout is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._cond.wait(remaining)
                else:
                    return None

    get = receive
    dequeue = receive

    # ------------------------------------------------------------------ #
    # 确认：消费成功后删除；未确认的消息可被重新投递，不会丢失
    # ------------------------------------------------------------------ #
    def ack(self, message_id: str) -> bool:
        with self._cond:
            if message_id not in self._inflight:
                return False
            del self._inflight[message_id]
            self._inflight_since.pop(message_id, None)
            return True

    def nack(self, message_id: str) -> bool:
        """消费失败：消息重新入队（保持原有顺序位置），等待再次投递。"""
        with self._cond:
            message = self._inflight.pop(message_id, None)
            if message is None:
                return False
            self._inflight_since.pop(message_id, None)
            self._pending.appendleft(message)
            self._cond.notify()
            return True

    def _requeue_expired(self) -> None:
        """可见性超时：长时间未确认的消息自动重新入队，防止消费失败丢失。"""
        if self.visibility_timeout is None:
            return
        now = time.monotonic()
        expired = [mid for mid, since in self._inflight_since.items()
                   if now - since >= self.visibility_timeout]
        for mid in sorted(expired, key=lambda m: self._inflight_since[m]):
            message = self._inflight.pop(mid)
            del self._inflight_since[mid]
            self._pending.appendleft(message)

    def __len__(self) -> int:
        with self._cond:
            return len(self._pending)

    @property
    def pending_count(self) -> int:
        return len(self)

    @property
    def inflight_count(self) -> int:
        with self._cond:
            return len(self._inflight)


class DistributedBarrier:
    """分布式屏障：所有节点到齐后同时放行（通知），支持等待超时。"""

    def __init__(self, name: str, parties: int):
        if parties <= 0:
            raise ValueError("parties 必须为正整数")
        self.name = name
        self.parties = parties
        self._cond = threading.Condition()
        self._arrived: List[str] = []
        self._generation = 0

    def wait(self, node_id: Optional[str] = None,
             timeout: Optional[float] = None) -> bool:
        """节点到达并等待。全部到齐返回 True（并通知所有等待者），超时返回 False。"""
        node_id = node_id or f"thread-{threading.get_ident()}"
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            generation = self._generation
            if node_id not in self._arrived:
                self._arrived.append(node_id)
            if len(self._arrived) >= self.parties:
                # 最后一个到达者触发屏障：通知所有等待节点，开启新一代
                self._arrived = []
                self._generation += 1
                self._cond.notify_all()
                return True
            while self._generation == generation:
                if timeout is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._cond.wait(remaining)
                else:
                    self._cond.wait()
            return True

    arrive_and_wait = wait

    @property
    def arrived_count(self) -> int:
        with self._cond:
            return len(self._arrived)


class DistributedLock:
    """游戏内分布式锁与协调系统：锁 + 选主 + 队列 + 屏障。"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state: Dict[str, Any] = {}
        self._history: List[Dict[str, Any]] = []
        self._locks: Dict[str, _LockEntry] = {}
        self._queues: Dict[str, DistributedQueue] = {}
        self._barriers: Dict[str, DistributedBarrier] = {}
        self._nodes: List[str] = []
        self._leader: Optional[str] = None
        self._term: int = 0
        self._election_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # 基础生命周期
    # ------------------------------------------------------------------ #
    def update(self, dt: float):
        for queue in self._queues.values():
            with queue._cond:
                queue._requeue_expired()

    def reset(self):
        self._state = {}
        self._history = []
        self._locks = {}
        self._queues = {}
        self._barriers = {}
        self._nodes = []
        self._leader = None
        self._term = 0

    def _record(self, action: str, **data) -> None:
        self._history.append({"action": action,
                              "timestamp": time.time(), **data})

    @staticmethod
    def _default_owner() -> str:
        return f"thread-{threading.get_ident()}"

    # ------------------------------------------------------------------ #
    # 分布式锁：真实互斥 + 超时 + 重入
    # ------------------------------------------------------------------ #
    def _get_lock_entry(self, name: str) -> _LockEntry:
        entry = self._locks.get(name)
        if entry is None:
            entry = self._locks[name] = _LockEntry()
        return entry

    def acquire_lock(self, name: str, owner: Optional[str] = None,
                     timeout: Optional[float] = None) -> bool:
        """获取命名锁。

        - 互斥：锁被其他持有者占用时阻塞等待
        - 超时：timeout 秒内拿不到锁返回 False，不会永远等
        - 重入：同一持有者（默认当前线程）可重复获取，计数 +1
        """
        owner = owner or self._default_owner()
        if timeout is None:
            timeout = self.config.get("lock_timeout")
        deadline = None if timeout is None else time.monotonic() + timeout
        entry = self._get_lock_entry(name)
        with entry.cond:
            while True:
                if entry.owner is None or entry.owner == owner:
                    entry.owner = owner
                    entry.count += 1
                    self._record("acquire", lock=name, owner=owner,
                                 count=entry.count)
                    return True
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._record("acquire_timeout", lock=name, owner=owner)
                        return False
                    entry.cond.wait(remaining)
                else:
                    entry.cond.wait()

    def release_lock(self, name: str, owner: Optional[str] = None) -> bool:
        """释放命名锁。

        - 检查：只有持有者本人能释放，否则返回 False
        - 释放：重入计数归零后真正释放
        - 通知：释放后唤醒所有等待该锁的线程
        """
        owner = owner or self._default_owner()
        entry = self._get_lock_entry(name)
        with entry.cond:
            if entry.owner is None or entry.owner != owner:
                self._record("release_rejected", lock=name, owner=owner)
                return False
            entry.count -= 1
            self._record("release", lock=name, owner=owner,
                         count=entry.count)
            if entry.count == 0:
                entry.owner = None
                entry.cond.notify_all()
            return True

    def is_locked(self, name: str) -> bool:
        entry = self._locks.get(name)
        return entry is not None and entry.owner is not None

    def get_lock_owner(self, name: str) -> Optional[str]:
        entry = self._locks.get(name)
        return entry.owner if entry else None

    lock_holder = get_lock_owner

    # ------------------------------------------------------------------ #
    # 领导者选举：任期 + 多数派投票 + 全局唯一
    # ------------------------------------------------------------------ #
    def register_node(self, node_id: str) -> None:
        if node_id not in self._nodes:
            self._nodes.append(node_id)

    def leader_election(self, nodes: Optional[List[str]] = None,
                        term: Optional[int] = None,
                        votes: Optional[Dict[str, str]] = None,
                        candidate: Optional[str] = None) -> Optional[str]:
        """发起一次领导者选举，返回当选的 leader（无多数派则返回 None）。

        - 任期：每次选举 term 单调递增；过期的 term 选举被拒绝
        - 投票：获得严格多数票（> n/2）的候选人才成为 leader
        - 唯一：任一时刻全系统只有一个 leader，所有节点可见一致
        """
        with self._election_lock:
            if nodes is None:
                nodes = list(self._nodes)
            else:
                for node in nodes:
                    self.register_node(node)
            if not nodes:
                return None

            if term is not None and term < self._term:
                # 过期任期的选举请求直接拒绝
                self._record("election_rejected", term=term)
                return self._leader
            self._term = max(self._term + 1, term or 0)

            if votes is None:
                # 默认投票：Bully 算法，标识最大的节点当选
                candidate = candidate or max(nodes)
                votes = {node: candidate for node in nodes}

            tally: Dict[str, int] = {}
            for voter, choice in votes.items():
                tally[choice] = tally.get(choice, 0) + 1

            majority = len(nodes) // 2 + 1
            winner = None
            for choice, count in tally.items():
                if count >= majority and (
                        winner is None or count > tally[winner]
                        or (count == tally[winner] and choice > winner)):
                    winner = choice

            self._leader = winner
            self._record("election", term=self._term, leader=winner,
                         votes=tally)
            return winner

    def get_leader(self) -> Optional[str]:
        return self._leader

    def is_leader(self, node_id: str) -> bool:
        return self._leader is not None and self._leader == node_id

    def get_term(self) -> int:
        return self._term

    @property
    def current_term(self) -> int:
        return self._term

    # ------------------------------------------------------------------ #
    # 分布式队列：确认 + 顺序
    # ------------------------------------------------------------------ #
    def distributed_queue(self, name: str = "default") -> DistributedQueue:
        queue = self._queues.get(name)
        if queue is None:
            queue = DistributedQueue(
                name, visibility_timeout=self.config.get("visibility_timeout"))
            self._queues[name] = queue
        return queue

    def send_message(self, queue: str, payload: Any) -> str:
        message_id = self.distributed_queue(queue).send(payload)
        self._record("send", queue=queue, message_id=message_id)
        return message_id

    def receive_message(self, queue: str,
                        timeout: Optional[float] = None) -> Optional[Message]:
        message = self.distributed_queue(queue).receive(timeout=timeout)
        if message is not None:
            self._record("receive", queue=queue,
                         message_id=message.message_id)
        return message

    def ack_message(self, queue: str, message_id: str) -> bool:
        ok = self.distributed_queue(queue).ack(message_id)
        self._record("ack", queue=queue, message_id=message_id, ok=ok)
        return ok

    # ------------------------------------------------------------------ #
    # 分布式屏障：超时 + 通知
    # ------------------------------------------------------------------ #
    def distributed_barrier(self, name: str, parties: int) -> DistributedBarrier:
        barrier = self._barriers.get(name)
        if barrier is None:
            barrier = DistributedBarrier(name, parties)
            self._barriers[name] = barrier
        return barrier

    def wait_barrier(self, name: str, node_id: Optional[str] = None,
                     timeout: Optional[float] = None) -> bool:
        barrier = self._barriers.get(name)
        if barrier is None:
            raise KeyError(f"未知屏障: {name}")
        ok = barrier.wait(node_id=node_id, timeout=timeout)
        self._record("barrier", barrier=name, node=node_id, ok=ok)
        return ok
