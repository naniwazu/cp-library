from cplib.graph.scc import SCCGraph


class TwoSAT:
    """2-SAT を解く。n 変数 m 節で O(n + m)。

    「x_i = f または x_j = g」という形の条件をすべて満たす真偽の割り当てを求める。

    Examples:
        >>> ts = TwoSAT(2)
        >>> ts.add_clause(0, True, 1, True)
        >>> ts.add_clause(0, False, 0, False)
        >>> ts.satisfiable()
        True
        >>> ts.answer()
        [False, True]
    """

    def __init__(self, n: int) -> None:
        self._n = n
        self._answer = [False] * n
        self._scc = SCCGraph(2 * n)

    def add_clause(self, i: int, f: bool, j: int, g: bool) -> None:
        """節 (x_i = f) or (x_j = g) を足す。"""
        self._scc.add_edge(2 * i + (not f), 2 * j + g)
        self._scc.add_edge(2 * j + (not g), 2 * i + f)

    def satisfiable(self) -> bool:
        """すべての節を満たす割り当てがあるかを返す。あれば answer() で取れる。"""
        ids = self._scc.scc_ids()[1]
        for i in range(self._n):
            if ids[2 * i] == ids[2 * i + 1]:
                return False
            self._answer[i] = ids[2 * i] < ids[2 * i + 1]
        return True

    def answer(self) -> list[bool]:
        """直前の satisfiable() が True のときの割り当てを返す。"""
        return self._answer
