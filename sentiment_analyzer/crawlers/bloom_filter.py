"""
布隆过滤器模块

提供布隆过滤器和可扩展布隆过滤器实现，用于高效判断元素是否存在。
"""

import math
import pickle
import struct
from dataclasses import dataclass, field
from hashlib import md5, sha256
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass
class BloomFilter:
    """
    布隆过滤器

    布隆过滤器是一种空间效率很高的概率型数据结构，
    用于判断一个元素是否在集合中。可能存在假阳性，但不会存在假阴性。

    Attributes:
        capacity: 预期元素数量
        error_rate: 可接受的假阳性率
        bit_array: 位数组
        bit_size: 位数组大小
        hash_count: 哈希函数数量
        element_count: 已添加元素数量

    Example:
        >>> bf = BloomFilter(capacity=100000, error_rate=0.001)
        >>> bf.add("example.com")
        >>> "example.com" in bf
        True
        >>> bf.save("filter.bloom")
    """

    capacity: int
    error_rate: float = 0.001
    bit_array: bytearray = field(default=b"", repr=False)
    bit_size: int = field(default=0)
    hash_count: int = field(default=0)
    element_count: int = field(default=0)

    def __post_init__(self) -> None:
        if self.error_rate <= 0 or self.error_rate >= 1:
            raise ValueError("error_rate must be between 0 and 1")

        if self.capacity <= 0:
            raise ValueError("capacity must be positive")

        if not self.bit_array:
            self.bit_size = self._calculate_bit_size(self.capacity, self.error_rate)
            self.hash_count = self._calculate_hash_count(self.bit_size, self.capacity)
            self.bit_array = bytearray((self.bit_size + 7) // 8)

    @staticmethod
    def _calculate_bit_size(n: int, p: float) -> int:
        """计算需要的位数组大小"""
        m = -n * math.log(p) / (math.log(2) ** 2)
        return int(math.ceil(m))

    @staticmethod
    def _calculate_hash_count(m: int, n: int) -> int:
        """计算需要的哈希函数数量"""
        k = m / n * math.log(2)
        return int(math.ceil(k))

    def _get_hashes(self, item: Any) -> Iterator[int]:
        """
        生成多个哈希值

        使用双哈希技术生成多个哈希值，减少实际需要的哈希函数数量。

        Args:
            item: 要哈希的元素

        Yields:
            哈希值
        """
        item_bytes = str(item).encode("utf-8")

        hash1 = int.from_bytes(md5(item_bytes).digest(), "big")
        hash2 = int.from_bytes(sha256(item_bytes).digest(), "big")

        for i in range(self.hash_count):
            combined_hash = (hash1 + i * hash2) % self.bit_size
            yield combined_hash

    def add(self, item: Any) -> None:
        """
        添加元素到过滤器

        Args:
            item: 要添加的元素
        """
        for hash_val in self._get_hashes(item):
            byte_index = hash_val // 8
            bit_index = hash_val % 8
            self.bit_array[byte_index] |= 1 << bit_index

        self.element_count += 1

    def __contains__(self, item: Any) -> bool:
        """
        检查元素是否可能在集合中

        Args:
            item: 要检查的元素

        Returns:
            True 如果元素可能在集合中，False 如果元素一定不在集合中
        """
        for hash_val in self._get_hashes(item):
            byte_index = hash_val // 8
            bit_index = hash_val % 8
            if not (self.bit_array[byte_index] & (1 << bit_index)):
                return False
        return True

    def __len__(self) -> int:
        return self.element_count

    def __ior__(self, other: "BloomFilter") -> "BloomFilter":
        """合并两个布隆过滤器（按位或）"""
        if self.bit_size != other.bit_size:
            raise ValueError("Cannot merge bloom filters with different sizes")

        for i in range(len(self.bit_array)):
            self.bit_array[i] |= other.bit_array[i]

        self.element_count += other.element_count
        return self

    def union(self, other: "BloomFilter") -> "BloomFilter":
        """返回两个布隆过滤器的并集"""
        if self.bit_size != other.bit_size:
            raise ValueError("Cannot merge bloom filters with different sizes")

        result = BloomFilter(
            capacity=self.capacity,
            error_rate=self.error_rate,
            bit_array=bytearray(self.bit_array),
            bit_size=self.bit_size,
            hash_count=self.hash_count,
            element_count=self.element_count + other.element_count,
        )

        for i in range(len(result.bit_array)):
            result.bit_array[i] |= other.bit_array[i]

        return result

    def save(self, path: str | Path) -> None:
        """
        持久化到文件

        Args:
            path: 文件路径
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "capacity": self.capacity,
            "error_rate": self.error_rate,
            "bit_array": bytes(self.bit_array),
            "bit_size": self.bit_size,
            "hash_count": self.hash_count,
            "element_count": self.element_count,
        }

        with open(path, "wb") as f:
            f.write(struct.pack(">I", len(pickle.dumps(data))))
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> "BloomFilter":
        """
        从文件加载

        Args:
            path: 文件路径

        Returns:
            加载的布隆过滤器
        """
        path = Path(path)

        with open(path, "rb") as f:
            _ = struct.unpack(">I", f.read(4))[0]
            data = pickle.load(f)

        return cls(
            capacity=data["capacity"],
            error_rate=data["error_rate"],
            bit_array=bytearray(data["bit_array"]),
            bit_size=data["bit_size"],
            hash_count=data["hash_count"],
            element_count=data["element_count"],
        )

    def clear(self) -> None:
        """清空过滤器"""
        self.bit_array = bytearray((self.bit_size + 7) // 8)
        self.element_count = 0

    @property
    def load_factor(self) -> float:
        """当前负载因子"""
        return self.element_count / self.capacity

    @property
    def estimated_false_positive_rate(self) -> float:
        """估计的当前假阳性率"""
        if self.element_count == 0:
            return 0.0

        return (
            1
            - math.exp(-self.hash_count * self.element_count / self.bit_size)
        ) ** self.hash_count

    def get_info(self) -> dict:
        """获取过滤器信息"""
        return {
            "capacity": self.capacity,
            "error_rate": self.error_rate,
            "bit_size": self.bit_size,
            "hash_count": self.hash_count,
            "element_count": self.element_count,
            "load_factor": self.load_factor,
            "estimated_fp_rate": self.estimated_false_positive_rate,
            "memory_bytes": len(self.bit_array),
        }


class ScalableBloomFilter:
    """
    可扩展布隆过滤器

    当元素数量超过预期时，自动扩展容量。
    使用多个布隆过滤器组成层级结构，新元素添加到最新的过滤器中。

    Attributes:
        initial_capacity: 初始容量
        error_rate: 目标假阳性率
        scale_factor: 容量扩展因子
        mode: 扩展模式

    Example:
        >>> sbf = ScalableBloomFilter(initial_capacity=1000, error_rate=0.001)
        >>> for i in range(10000):
        ...     sbf.add(f"item_{i}")
        >>> "item_5000" in sbf
        True
    """

    SMALL_GROWTH = 0
    LARGE_GROWTH = 1

    def __init__(
        self,
        initial_capacity: int = 1000,
        error_rate: float = 0.001,
        scale_factor: float = 2.0,
        mode: int = SMALL_GROWTH,
    ) -> None:
        if error_rate <= 0 or error_rate >= 1:
            raise ValueError("error_rate must be between 0 and 1")

        if initial_capacity <= 0:
            raise ValueError("initial_capacity must be positive")

        self._initial_capacity = initial_capacity
        self._error_rate = error_rate
        self._scale_factor = scale_factor
        self._mode = mode
        self._filters: list[BloomFilter] = []
        self._element_count = 0
        self._current_capacity = initial_capacity

        self._add_filter()

    def _add_filter(self) -> None:
        """添加新的布隆过滤器层"""
        num_filters = len(self._filters)

        if self._mode == self.SMALL_GROWTH:
            error_rate = self._error_rate * (0.9 ** num_filters)
        else:
            error_rate = self._error_rate / (2 ** num_filters)

        capacity = int(self._initial_capacity * (self._scale_factor ** num_filters))

        new_filter = BloomFilter(capacity=capacity, error_rate=error_rate)
        self._filters.append(new_filter)
        self._current_capacity = capacity

    def add(self, item: Any) -> None:
        """
        添加元素

        Args:
            item: 要添加的元素
        """
        if item in self:
            return

        current_filter = self._filters[-1]

        if current_filter.element_count >= current_filter.capacity:
            self._add_filter()
            current_filter = self._filters[-1]

        current_filter.add(item)
        self._element_count += 1

    def __contains__(self, item: Any) -> bool:
        """
        检查元素是否可能在集合中

        Args:
            item: 要检查的元素

        Returns:
            True 如果元素可能在集合中
        """
        return any(item in f for f in self._filters)

    def __len__(self) -> int:
        return self._element_count

    def add_many(self, items: Iterable[Any]) -> None:
        """
        批量添加元素

        Args:
            items: 要添加的元素集合
        """
        for item in items:
            self.add(item)

    def save(self, path: str | Path) -> None:
        """
        持久化到文件

        Args:
            path: 文件路径
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "initial_capacity": self._initial_capacity,
            "error_rate": self._error_rate,
            "scale_factor": self._scale_factor,
            "mode": self._mode,
            "element_count": self._element_count,
            "filters": [
                {
                    "capacity": f.capacity,
                    "error_rate": f.error_rate,
                    "bit_array": bytes(f.bit_array),
                    "bit_size": f.bit_size,
                    "hash_count": f.hash_count,
                    "element_count": f.element_count,
                }
                for f in self._filters
            ],
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> "ScalableBloomFilter":
        """
        从文件加载

        Args:
            path: 文件路径

        Returns:
            加载的可扩展布隆过滤器
        """
        path = Path(path)

        with open(path, "rb") as f:
            data = pickle.load(f)

        sbf = cls(
            initial_capacity=data["initial_capacity"],
            error_rate=data["error_rate"],
            scale_factor=data["scale_factor"],
            mode=data["mode"],
        )

        sbf._element_count = data["element_count"]
        sbf._filters = []

        for filter_data in data["filters"]:
            bf = BloomFilter(
                capacity=filter_data["capacity"],
                error_rate=filter_data["error_rate"],
                bit_array=bytearray(filter_data["bit_array"]),
                bit_size=filter_data["bit_size"],
                hash_count=filter_data["hash_count"],
                element_count=filter_data["element_count"],
            )
            sbf._filters.append(bf)

        return sbf

    def clear(self) -> None:
        """清空过滤器"""
        self._filters = []
        self._element_count = 0
        self._current_capacity = self._initial_capacity
        self._add_filter()

    def get_info(self) -> dict:
        """获取过滤器信息"""
        return {
            "initial_capacity": self._initial_capacity,
            "error_rate": self._error_rate,
            "scale_factor": self._scale_factor,
            "mode": "SMALL_GROWTH" if self._mode == self.SMALL_GROWTH else "LARGE_GROWTH",
            "element_count": self._element_count,
            "filter_count": len(self._filters),
            "current_capacity": self._current_capacity,
            "total_memory_bytes": sum(len(f.bit_array) for f in self._filters),
        }
