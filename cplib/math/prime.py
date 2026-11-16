from math import gcd, isqrt


def primes(n: int) -> list[int]:
    """n 以下の素数を昇順に返す（エラトステネスの篩）。O(n log log n)。

    Examples:
        >>> primes(20)
        [2, 3, 5, 7, 11, 13, 17, 19]
    """
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            sieve[i * i :: i] = bytes(len(range(i * i, n + 1, i)))
    return [i for i in range(n + 1) if sieve[i]]


def min_factor_table(n: int) -> list[int]:
    """0 から n までの各整数の最小の素因数の表を返す。O(n log log n)。

    n 以下の整数を多数素因数分解する問題で、x を t[x] で割り続けて O(log x) で
    分解するのに使う。
    t[0] = 0、t[1] = 1 で、素数 p では t[p] = p。

    Examples:
        >>> min_factor_table(10)
        [0, 1, 2, 3, 2, 5, 2, 7, 2, 3, 2]
    """
    t = list(range(n + 1))
    for i in range(2, isqrt(n) + 1):
        if t[i] == i:
            for j in range(i * i, n + 1, i):
                if t[j] == j:
                    t[j] = i
    return t


def is_prime(n: int) -> bool:
    """n が素数かを判定する（決定的 Miller-Rabin、n < 2^64）。O(log n) 回の乗算。

    篩では間に合わない大きな数の素数判定に使う。

    Examples:
        >>> is_prime(1), is_prime(2), is_prime(998244353), is_prime(561)
        (False, True, True, False)
    """
    if n < 2:
        return False
    # 2^64 未満ならこの 12 個を底にすれば判定が確定する
    bases = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in bases:
        if n % p == 0:
            return n == p
    s = ((n - 1) & (1 - n)).bit_length() - 1
    d = (n - 1) >> s
    for a in bases:
        y = pow(a, d, n)
        if y == 1 or y == n - 1:
            continue
        for _ in range(s - 1):
            y = y * y % n
            if y == n - 1:
                break
        else:
            return False
    return True


def _pollard_rho(n: int) -> int:
    # 合成数 n の自明でない約数を 1 つ返す（Brent の改良版）
    m = 1 << (n.bit_length() // 8)
    for c in range(1, n):
        y, r, q, g = 2, 1, 1, 1
        x = ys = 0
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = gcd(q, n)
                k += m
            r <<= 1
        if g == n:
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = gcd(abs(x - ys), n)
        if g < n:
            return g
    raise ValueError(f"no factor found: {n}")


def factorize(n: int) -> dict[int, int]:
    """n を素因数分解し、{素因数: 指数} を素因数の昇順で返す（1 <= n < 2^64）。

    Pollard の rho 法を使うので、10^18 程度の数 1 つを高速に分解できる。

    Examples:
        >>> factorize(360)
        {2: 3, 3: 2, 5: 1}
        >>> factorize(999999999999999989 * 3)
        {3: 1, 999999999999999989: 1}
    """
    # 0 は試し割りが終わらない
    assert n >= 1
    res: dict[int, int] = {}
    # 小さい素因数は試し割りのほうが速い
    for p in range(2, 100):
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            res[p] = e
    rest = n
    stack = [n] if n > 1 else []
    while stack:
        x = stack.pop()
        if x in res:
            continue
        if is_prime(x):
            e = 0
            while rest % x == 0:
                rest //= x
                e += 1
            res[x] = e
        else:
            d = _pollard_rho(x)
            stack.append(d)
            stack.append(x // d)
    return dict(sorted(res.items()))
