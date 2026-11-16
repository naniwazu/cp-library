def subset_zeta(a: list[int]) -> None:
    """a[S] を S の部分集合 T すべてについての a[T] の和に置き換える（高速ゼータ変換）。

    len(a) = 2^n で、添字をビット集合とみなす。「S に含まれる集合の個数・和」を
    全 S について求める問題や、subset_mobius と組んで OR 畳み込みを求めるのに使う。
    O(n 2^n)。mod を取る場合は、各要素が mod 未満なら n <= 22 程度まで途中で
    多倍長にならないので、変換後にまとめて剰余を取ればよい。

    Examples:
        >>> a = [1, 2, 3, 4]
        >>> subset_zeta(a)
        >>> a
        [1, 3, 4, 10]
    """
    n = len(a)
    assert n & (n - 1) == 0
    b = 1
    while b < n:
        # ブロックごとに回すより、全添字を回してビットで判定するほうが
        # PyPy で 2 割ほど速い
        for s in range(n):
            if s & b:
                a[s] += a[s ^ b]
        b <<= 1


def subset_mobius(a: list[int]) -> None:
    """subset_zeta の逆変換（高速メビウス変換）。O(n 2^n)。

    「ちょうど S を使う」個数を、「S の部分集合を使う」個数から包除で求めるのに使う。

    Examples:
        >>> a = [1, 3, 4, 10]
        >>> subset_mobius(a)
        >>> a
        [1, 2, 3, 4]
    """
    n = len(a)
    assert n & (n - 1) == 0
    b = 1
    while b < n:
        for s in range(n):
            if s & b:
                a[s] -= a[s ^ b]
        b <<= 1


def superset_zeta(a: list[int]) -> None:
    """a[S] を S を含む集合 T すべてについての a[T] の和に置き換える。O(n 2^n)。

    len(a) = 2^n。「S を含む集合の個数・和」を全 S について求める問題や、
    superset_mobius と組んで AND 畳み込みを求めるのに使う。

    Examples:
        >>> a = [1, 2, 3, 4]
        >>> superset_zeta(a)
        >>> a
        [10, 6, 7, 4]
    """
    n = len(a)
    assert n & (n - 1) == 0
    b = 1
    while b < n:
        for s in range(n):
            if s & b:
                a[s ^ b] += a[s]
        b <<= 1


def superset_mobius(a: list[int]) -> None:
    """superset_zeta の逆変換。O(n 2^n)。

    Examples:
        >>> a = [10, 6, 7, 4]
        >>> superset_mobius(a)
        >>> a
        [1, 2, 3, 4]
    """
    n = len(a)
    assert n & (n - 1) == 0
    b = 1
    while b < n:
        for s in range(n):
            if s & b:
                a[s ^ b] -= a[s]
        b <<= 1
