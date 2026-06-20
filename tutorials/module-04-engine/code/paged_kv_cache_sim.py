"""M4 Lesson 3:PagedAttention 的 KV Cache 分页管理模拟。

借鉴 OS 虚拟内存分页:把 KV Cache 切成固定大小的"块(block)",
每个请求用一张"块表(block table)"记录自己的逻辑块 -> 物理块映射。
请求按需申请块,结束后归还,从而消除"为每个请求按最大长度整块预留"造成的浪费和碎片。

这个模拟不做真实 attention 计算,只演示**显存块的分配/释放/共享**机制,
对照"静态预留(naive)"和"分页(paged)"两种策略能容纳多少并发请求。

运行:
    python paged_kv_cache_sim.py
"""

import logging
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BLOCK_SIZE = 16  # 每个物理块能存多少个 token 的 KV(vLLM 默认也是 16)


class OutOfMemory(Exception):
    """显存(物理块)耗尽。"""


@dataclass
class BlockAllocator:
    """物理块分配器:管理一池固定大小的 KV 块,类比 OS 的物理页帧管理。"""

    num_blocks: int
    free_blocks: list = field(default_factory=list)
    ref_count: dict = field(default_factory=dict)  # 物理块 -> 引用计数,支持共享(CoW)

    def __post_init__(self):
        # 物理块编号 0..num_blocks-1,初始全部空闲。
        self.free_blocks = list(range(self.num_blocks))
        self.ref_count = {b: 0 for b in range(self.num_blocks)}

    def allocate(self) -> int:
        if not self.free_blocks:
            raise OutOfMemory("没有空闲物理块了")
        block = self.free_blocks.pop()
        self.ref_count[block] = 1
        return block

    def share(self, block: int) -> None:
        # 多个请求共享同一物理块(如共享 prompt 前缀),引用计数 +1。
        self.ref_count[block] += 1

    def free(self, block: int) -> None:
        # 引用计数归零才真正回收,这是前缀/beam 共享安全释放的关键。
        self.ref_count[block] -= 1
        if self.ref_count[block] == 0:
            self.free_blocks.append(block)

    @property
    def num_free(self) -> int:
        return len(self.free_blocks)


@dataclass
class Sequence:
    """一个请求序列,持有自己的块表(逻辑块号 -> 物理块号)。"""

    sid: int
    block_table: list = field(default_factory=list)  # 索引 = 逻辑块号,值 = 物理块号
    length: int = 0  # 当前已占用的 token 数

    def num_blocks_needed(self, total_tokens: int) -> int:
        # 向上取整:存 total_tokens 个 token 需要多少个块。
        return (total_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE

    def append_tokens(self, n: int, allocator: BlockAllocator) -> None:
        # 模拟生成了 n 个新 token,按需向分配器申请新的物理块。
        new_len = self.length + n
        need = self.num_blocks_needed(new_len)
        while len(self.block_table) < need:
            phys = allocator.allocate()
            self.block_table.append(phys)
        self.length = new_len

    def free(self, allocator: BlockAllocator) -> None:
        for phys in self.block_table:
            allocator.free(phys)
        self.block_table.clear()
        self.length = 0


def naive_capacity(total_blocks: int, max_len: int) -> int:
    """静态预留:每个请求按 max_len 整块预占,不管它实际生成多少。"""
    blocks_per_req = (max_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    return total_blocks // blocks_per_req


def demo_paging() -> None:
    """演示:在同样的物理块预算下,分页能容纳远多于静态预留的请求。"""
    total_blocks = 100
    max_len = 512  # 假设每个请求"最多可能"生成 512 token

    cap_naive = naive_capacity(total_blocks, max_len)
    logger.info("物理块总数=%d,每块=%d token,max_len=%d", total_blocks, BLOCK_SIZE, max_len)
    logger.info("静态预留:每请求占 %d 块 -> 最多容纳 %d 个请求",
                (max_len + BLOCK_SIZE - 1) // BLOCK_SIZE, cap_naive)

    # 分页:大多数请求其实只生成几十个 token,按需分配。
    allocator = BlockAllocator(num_blocks=total_blocks)
    seqs = []
    actual_len = 40  # 典型请求实际长度,远小于 max_len
    sid = 0
    try:
        while True:
            s = Sequence(sid=sid)
            s.append_tokens(actual_len, allocator)
            seqs.append(s)
            sid += 1
    except OutOfMemory:
        pass
    logger.info("分页按需:每请求实际 %d token -> 容纳了 %d 个请求(同样 %d 块)",
                actual_len, len(seqs), total_blocks)
    logger.info("提升约 %.1fx", len(seqs) / max(cap_naive, 1))


def demo_prefix_sharing() -> None:
    """演示:多个请求共享同一段 prompt 前缀,对应物理块只存一份(引用计数)。"""
    logger.info("")
    logger.info("--- 前缀共享演示 ---")
    allocator = BlockAllocator(num_blocks=100)

    # 请求 A 先把一段 32 token 的公共前缀(2 个块)装进 KV。
    a = Sequence(sid=0)
    a.append_tokens(32, allocator)
    shared_prefix = list(a.block_table)
    logger.info("请求 A 前缀占用物理块: %s,剩余空闲块=%d", shared_prefix, allocator.num_free)

    # 请求 B 复用同样的前缀:块表直接指向 A 的物理块,引用计数 +1,不重复分配。
    b = Sequence(sid=1)
    for phys in shared_prefix:
        allocator.share(phys)
        b.block_table.append(phys)
    b.length = 32
    logger.info("请求 B 复用前缀(零额外显存),剩余空闲块=%d", allocator.num_free)
    logger.info("共享块 %d 的引用计数=%d", shared_prefix[0], allocator.ref_count[shared_prefix[0]])

    # A 结束释放,但前缀块还被 B 引用,不会被真正回收。
    a.free(allocator)
    logger.info("A 释放后,共享块 %d 引用计数=%d(仍被 B 持有,未回收)",
                shared_prefix[0], allocator.ref_count[shared_prefix[0]])


def main() -> None:
    demo_paging()
    demo_prefix_sharing()


if __name__ == "__main__":
    main()
