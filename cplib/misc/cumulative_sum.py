from collections.abc import Sequence
from itertools import accumulate


class CumulativeSum2D:
    """整数の 2 次元配列の長方形領域の和を O(1) で返す（前計算 O(hw)）。

    グリッド上の長方形に含まれる個数や重みの和を何度も求める問題に使う。値は変更できない。
    grid の各行の長さは等しいこと。

    Examples:
        >>> cs = CumulativeSum2D([[1, 2, 3], [4, 5, 6]])
        >>> cs.sum(0, 0, 2, 3)
        21
        >>> cs.sum(1, 1, 2, 3)
        11
    """

    def __init__(self, grid: Sequence[Sequence[int]]) -> None:
        prev = [0] * (len(grid[0]) + 1 if grid else 1)
        # _s[y][x] は grid[:y] の各行の [:x] の和
        s = [prev]
        for row in grid:
            prev = [
                p + q for p, q in zip(prev, accumulate(row, initial=0), strict=True)
            ]
            s.append(prev)
        self._s = s

    def sum(self, y1: int, x1: int, y2: int, x2: int) -> int:
        """y1 <= y < y2 かつ x1 <= x < x2 の grid[y][x] の和を返す。

        h, w を grid の行数と列数として、0 <= y1 <= y2 <= h かつ
        0 <= x1 <= x2 <= w であること。
        """
        s = self._s
        r1 = s[y1]
        r2 = s[y2]
        return r2[x2] - r1[x2] - r2[x1] + r1[x1]
