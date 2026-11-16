from heapq import heappop, heappush
from typing import Any, Generic, Protocol, TypeVar


class _RemovableHeapComparable(Protocol):
    def __lt__(self, other: Any, /) -> bool: ...


_RemovableHeapT = TypeVar("_RemovableHeapT", bound=_RemovableHeapComparable)


class RemovableHeap(Generic[_RemovableHeapT]):
    """任意の要素を取り除ける最小ヒープ。各操作は償却 O(log n)。

    値の追加・削除をしながら最小値を求める問題で使う。SortedList でも同じことが
    できるが、最小値（か最大値）しか見ないならこちらが PyPy で数倍速い。
    heapq と同じく最小ヒープで、最大ヒープが欲しいときは符号を反転して入れる。

    remove(x) は x を削除待ちのヒープに入れるだけで、x が最小値になったときに
    まとめて取り除く（遅延削除）。そのため remove(x) する x は、その時点で
    ヒープに含まれていること（含まれない x を渡すと以後の結果が壊れる。
    存在の確認はしない）。要素どうしは < と == で比較できること。

    Examples:
        >>> h = RemovableHeap[int]()
        >>> for x in [5, 1, 4, 1]:
        ...     h.push(x)
        >>> h.remove(1)
        >>> h.top(), len(h)
        (1, 3)
        >>> h.remove(1)
        >>> h.pop(), h.pop(), len(h)
        (4, 5, 0)
    """

    def __init__(self) -> None:
        self._heap: list[_RemovableHeapT] = []
        # 削除待ちの要素。_heap の先頭と一致したら両方から取り除く
        self._removed: list[_RemovableHeapT] = []
        self._n = 0

    def __len__(self) -> int:
        return self._n

    def push(self, x: _RemovableHeapT) -> None:
        """x を 1 個追加する。O(log n)。"""
        heappush(self._heap, x)
        self._n += 1

    def remove(self, x: _RemovableHeapT) -> None:
        """x を 1 個取り除く。x はヒープに含まれていること。O(log n)。"""
        heappush(self._removed, x)
        self._n -= 1

    def top(self) -> _RemovableHeapT:
        """最小値を返す。空なら IndexError を送出する。償却 O(log n)。"""
        heap = self._heap
        removed = self._removed
        while removed and heap[0] == removed[0]:
            heappop(heap)
            heappop(removed)
        return heap[0]

    def pop(self) -> _RemovableHeapT:
        """最小値を取り除いて返す。空なら IndexError を送出する。償却 O(log n)。"""
        heap = self._heap
        removed = self._removed
        while removed and heap[0] == removed[0]:
            heappop(heap)
            heappop(removed)
        x = heappop(heap)
        self._n -= 1
        return x
