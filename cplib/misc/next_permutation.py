from typing import Any, Protocol, TypeVar


class _NextPermutationComparable(Protocol):
    def __lt__(self, other: Any, /) -> bool: ...


_NextPermutationT = TypeVar("_NextPermutationT", bound=_NextPermutationComparable)


def next_permutation(a: list[_NextPermutationT]) -> bool:
    """a を辞書順で次の順列に書き換え、書き換えたら True を返す。

    C++ の std::next_permutation と同じ。昇順に並べた a から始めて False が返るまで
    呼ぶと、重複を除いた全順列を辞書順にたどれる。a が最後の順列（降順）なら昇順に
    戻して False を返す。要素どうしは < で比較できること。1 回あたり最悪 O(n)、
    全順列をたどると 1 回あたり償却 O(1)。

    Examples:
        >>> a = [1, 2, 2]
        >>> next_permutation(a), a
        (True, [2, 1, 2])
        >>> next_permutation(a), a
        (True, [2, 2, 1])
        >>> next_permutation(a), a
        (False, [1, 2, 2])
    """
    n = len(a)
    i = n - 2
    while i >= 0 and not a[i] < a[i + 1]:
        i -= 1
    if i < 0:
        a.reverse()
        return False
    j = n - 1
    while not a[i] < a[j]:
        j -= 1
    a[i], a[j] = a[j], a[i]
    a[i + 1 :] = a[:i:-1]
    return True
