from cplib.math.prime import primes


def divisor_zeta(a: list[int]) -> None:
    """a[k] を k の約数 d すべてについての a[d] の和に置き換える（1 <= k <= n）。

    len(a) = n + 1 で、a[0] は使わず変更もしない。「k の約数であるものの個数・和」を
    全 k について求める問題や、divisor_mobius と組んで LCM 畳み込みを求めるのに使う。
    素数ごとに累積和を取るので O(n log log n)。

    Examples:
        >>> a = [0, 1, 1, 1, 1, 1, 1]
        >>> divisor_zeta(a)  # 約数の個数
        >>> a
        [0, 1, 2, 2, 3, 2, 4]
    """
    n = len(a) - 1
    for p in primes(n):
        for i in range(1, n // p + 1):
            a[i * p] += a[i]


def divisor_mobius(a: list[int]) -> None:
    """divisor_zeta の逆変換。O(n log log n)。

    Examples:
        >>> a = [0, 1, 2, 2, 3, 2, 4]
        >>> divisor_mobius(a)
        >>> a
        [0, 1, 1, 1, 1, 1, 1]
    """
    n = len(a) - 1
    for p in primes(n):
        for i in range(n // p, 0, -1):
            a[i * p] -= a[i]


def multiple_zeta(a: list[int]) -> None:
    """a[d] を n 以下の d の倍数 m すべてについての a[m] の和に置き換える。

    1 <= d <= n で、len(a) = n + 1。a[0] は使わず変更もしない。「d で割り切れる
    ものの個数・和」を全 d について求める問題や、multiple_mobius と組んで
    「gcd がちょうど d」の個数や GCD 畳み込みを求めるのに使う。O(n log log n)。

    Examples:
        >>> a = [0, 1, 1, 1, 1, 1, 1]
        >>> multiple_zeta(a)  # n // d
        >>> a
        [0, 6, 3, 2, 1, 1, 1]
    """
    n = len(a) - 1
    for p in primes(n):
        for i in range(n // p, 0, -1):
            a[i] += a[i * p]


def multiple_mobius(a: list[int]) -> None:
    """multiple_zeta の逆変換。O(n log log n)。

    Examples:
        >>> a = [0, 6, 3, 2, 1, 1, 1]
        >>> multiple_mobius(a)
        >>> a
        [0, 1, 1, 1, 1, 1, 1]
    """
    n = len(a) - 1
    for p in primes(n):
        for i in range(1, n // p + 1):
            a[i] -= a[i * p]
