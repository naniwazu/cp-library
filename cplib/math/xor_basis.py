class XorBasis:
    """非負整数の集合の XOR で張られる空間（GF(2) 上の線形空間）を管理する。

    「いくつか選んで XOR を取ったときの最大値」「x が作れるか」「作れる値の種類数
    （2 の階数乗）」を求める問題に使う。値のビット長を w として、各操作は O(w)。

    Examples:
        >>> xb = XorBasis()
        >>> xb.add(5), xb.add(3), xb.add(6)
        (True, True, False)
        >>> len(xb), 6 in xb, 1 in xb
        (2, True, False)
        >>> xb.max(), xb.min(7)
        (6, 1)
    """

    def __init__(self) -> None:
        # _slot[i] は最上位ビットが i の基底ベクトル（なければ 0）。基底を値の順に
        # 並べて持つより add が PyPy で 2 倍ほど速い
        self._slot: list[int] = []
        self._rank = 0

    def add(self, x: int) -> bool:
        """非負整数 x を追加する。既存の基底と独立なら基底に加えて True を返す。"""
        s = self._slot
        while x:
            i = x.bit_length() - 1
            if i >= len(s):
                s.extend([0] * (i + 1 - len(s)))
            b = s[i]
            if not b:
                s[i] = x
                self._rank += 1
                return True
            x ^= b
        return False

    def __contains__(self, x: int) -> bool:
        s = self._slot
        while x:
            i = x.bit_length() - 1
            if i >= len(s) or not s[i]:
                return False
            x ^= s[i]
        return True

    def __len__(self) -> int:
        """基底の大きさ（階数）を返す。張る空間の要素数は 2 ** len(self)。"""
        return self._rank

    def max(self, x: int = 0) -> int:
        """張る空間の元 v について x ^ v の最大値を返す。"""
        s = self._slot
        for i in range(len(s) - 1, -1, -1):
            x = max(x, x ^ s[i])
        return x

    def min(self, x: int = 0) -> int:
        """張る空間の元 v について x ^ v の最小値を返す。x = 0 なら 0。"""
        s = self._slot
        for i in range(len(s) - 1, -1, -1):
            x = min(x, x ^ s[i])
        return x
