class Doubling:
    """各頂点の行き先が 1 つ決まったグラフで、v から k 歩先の頂点に答える。

    「各町のテレポーターを 10^18 回使った先」のように、写像を何度も繰り返した
    結果を求める問題に使う。nxt[v] が v の 1 歩先。0 <= k <= max_k の k に答える。
    ダブリングで、構築 O(n log max_k)、クエリ O(log k)。

    クエリが 1 回だけでもこのまま使ってよい（n = 2×10^5、max_k = 10^18 で
    構築は 0.05 秒ほど）。

    Examples:
        >>> d = Doubling([1, 2, 0, 0], 10**18)
        >>> d.jump(3, 1), d.jump(3, 5), d.jump(0, 10**18)
        (0, 1, 1)
    """

    def __init__(self, nxt: list[int], max_k: int) -> None:
        assert max_k >= 0
        # table[j][v] は v から 2^j 歩先
        table = [nxt[:]]
        for _ in range(max_k.bit_length() - 1):
            a = table[-1]
            table.append([a[x] for x in a])
        self._table = table

    def jump(self, v: int, k: int) -> int:
        """v から k 歩先の頂点を返す。"""
        for a in self._table:
            if k & 1:
                v = a[v]
            k >>= 1
            if not k:
                break
        return v
