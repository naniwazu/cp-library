from heapq import heappush, heappushpop

_SLOPE_TRICK_INF = 1 << 62


class SlopeTrick:
    """凸関数 f（初期値 0）に max(x - a, 0) や |x - a| を足しながら最小値を求める。

    「各項を a_i に近づけるコストの和を最小化」のような DP を、DP テーブルを
    関数として持って高速化するときに使う。各操作は O(log n)（n は操作回数）。

    Examples:
        >>> f = SlopeTrick()
        >>> f.add_abs(1)
        >>> f.add_abs(5)
        >>> f.min()
        4
        >>> f.add_a_minus_x(7)
        >>> f.min()
        6
        >>> f.add_const(10)
        >>> f.min()
        16
    """

    def __init__(self) -> None:
        self._min_f = 0
        # 傾きが変わる点の多重集合。_l は最小値をとる区間より左（符号を反転した
        # 最大ヒープ）、_r は右。番兵を入れて空の場合分けをなくす
        self._l = [_SLOPE_TRICK_INF]
        self._r = [_SLOPE_TRICK_INF]

    def min(self) -> int:
        """f の最小値を返す。O(1)。"""
        return self._min_f

    def add_const(self, a: int) -> None:
        """f(x) += a を行う。"""
        self._min_f += a

    def add_x_minus_a(self, a: int) -> None:
        """f(x) += max(x - a, 0) を行う。"""
        self._min_f += max(-self._l[0] - a, 0)
        heappush(self._r, -heappushpop(self._l, -a))

    def add_a_minus_x(self, a: int) -> None:
        """f(x) += max(a - x, 0) を行う。"""
        self._min_f += max(a - self._r[0], 0)
        heappush(self._l, -heappushpop(self._r, a))

    def add_abs(self, a: int) -> None:
        """f(x) += |x - a| を行う。"""
        self.add_x_minus_a(a)
        self.add_a_minus_x(a)
