from math import gcd


def inv_mod(x: int, m: int) -> int:
    """x * y ≡ 1 (mod m) となる 0 <= y < m を返す（gcd(x, m) == 1、1 <= m）。O(log m)。

    m が素数でなくても使える。x は負でもよい。逆元がなければ ValueError。

    Examples:
        >>> inv_mod(3, 7), inv_mod(-1, 10), inv_mod(5, 1)
        (5, 9, 0)
    """
    return pow(x, -1, m)


def crt(r: list[int], m: list[int]) -> tuple[int, int]:
    """連立合同式 x ≡ r[i] (mod m[i]) の解を x ≡ y (mod z) として (y, z) で返す。

    中国剰余定理。z = lcm(m) で 0 <= y < z。m[i] は互いに素でなくてよい（1 <= m[i]）。
    r[i] は負でもよい。解がなければ (0, 0)、空の列なら (0, 1)。
    O(len(m) log lcm(m))（多倍長演算のコストを除く）。

    Examples:
        >>> crt([2, 3, 2], [3, 5, 7])
        (23, 105)
        >>> crt([1, 3], [4, 6])
        (9, 12)
        >>> crt([0, 1], [2, 4])
        (0, 0)
    """
    r0 = 0
    m0 = 1
    # 0 <= r0 < m0 を保ちながら 1 本ずつ合成する
    for r1, m1 in zip(r, m, strict=True):
        g = gcd(m0, m1)
        if (r1 - r0) % g:
            return (0, 0)
        u1 = m1 // g
        # r0 + t * m0 ≡ r1 (mod m1) を t について解く
        r0 += (r1 - r0) // g * pow(m0 // g, -1, u1) % u1 * m0
        m0 *= u1
    return (r0, m0)


def ext_gcd(a: int, b: int) -> tuple[int, int, int]:
    """a * x + b * y == g となる (g, x, y) を返す（g = gcd(a, b)）。

    拡張ユークリッドの互除法。1 次不定方程式 a * x + b * y = c を解く問題に使う。
    c が g で割り切れるときに限り解があり、(x * (c // g), y * (c // g)) が
    その 1 つになる。
    a, b は負や 0 でもよい。g は常に 0 以上で、gcd(0, 0) = 0 とする。
    O(log min(|a|, |b|))。
    |x| <= max(|b| // g, 1)、|y| <= max(|a| // g, 1) を満たす（g > 0 のとき）。

    Examples:
        >>> ext_gcd(12, 18)
        (6, -1, 1)
        >>> ext_gcd(-4, 6), ext_gcd(0, -5), ext_gcd(0, 0)
        ((2, 1, 1), (5, 0, -1), (0, 1, 0))
    """
    x0, y0, x1, y1 = 1, 0, 0, 1
    while b:
        q = a // b
        a, b = b, a - q * b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    if a < 0:
        return (-a, -x0, -y0)
    return (a, x0, y0)
