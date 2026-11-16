from collections.abc import Sequence


class Trie:
    """文字列の集合を接頭辞木で持ち、追加と接頭辞の個数の数え上げを O(|s|) で行う。

    「s を接頭辞にもつ文字列がいくつあるか」を数える問題や、各節点の個数を使って
    全組の最長共通接頭辞の長さの和などを求める問題で使う。重複を許す。
    str のほか、整数の列も受け付ける（1 つの Trie にはどちらか一方だけを入れる。
    整数の絶対値が 2^31 以上だと内部のキーが多倍長になり遅くなる）。

    節点は 0（根、空列）から順に番号が振られ、child で子をたどれる。
    prefix_counts[v] は節点 v までの経路を接頭辞にもつ追加済みの列の個数、
    counts[v] は経路そのものに等しい列の個数。

    Examples:
        >>> t = Trie()
        >>> for s in ["abc", "ab", "abd", "ab", "b"]:
        ...     t.add(s)
        >>> t.count("ab"), t.count("a"), t.count_prefix("ab"), t.count_prefix("")
        (2, 0, 4, 5)
        >>> v = t.child(t.child(0, "a"), "b")
        >>> t.prefix_counts[v], t.counts[v], t.child(v, "x")
        (4, 2, -1)
    """

    def __init__(self) -> None:
        # (文字コード << 32 | 親) -> 子。節点ごとに dict を持つより PyPy で速い
        self._next: dict[int, int] = {}
        self.prefix_counts = [0]
        self.counts = [0]

    def add(self, s: str | Sequence[int]) -> None:
        """s を 1 個追加する。O(|s|)。"""
        nxt = self._next
        pre = self.prefix_counts
        cnt = self.counts
        v = 0
        pre[0] += 1
        for x in map(ord, s) if isinstance(s, str) else s:
            k = x << 32 | v
            u = nxt.get(k, -1)
            if u < 0:
                u = len(pre)
                nxt[k] = u
                pre.append(0)
                cnt.append(0)
            v = u
            pre[v] += 1
        cnt[v] += 1

    def count(self, s: str | Sequence[int]) -> int:
        """s に等しい列の個数を返す。O(|s|)。"""
        nxt = self._next
        v = 0
        for x in map(ord, s) if isinstance(s, str) else s:
            v = nxt.get(x << 32 | v, -1)
            if v < 0:
                return 0
        return self.counts[v]

    def count_prefix(self, s: str | Sequence[int]) -> int:
        """s を接頭辞にもつ列の個数を返す。O(|s|)。"""
        nxt = self._next
        v = 0
        for x in map(ord, s) if isinstance(s, str) else s:
            v = nxt.get(x << 32 | v, -1)
            if v < 0:
                return 0
        return self.prefix_counts[v]

    def child(self, v: int, c: str | int) -> int:
        """節点 v から文字 c で進んだ子の番号を返す。なければ -1。O(1)。"""
        x = ord(c) if isinstance(c, str) else c
        return self._next.get(x << 32 | v, -1)
