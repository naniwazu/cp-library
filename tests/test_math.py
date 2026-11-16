import random
from math import comb, factorial, gcd, isqrt, lcm, perm

import pytest

from cplib.math.comb import Comb
from cplib.math.convolution import convolution
from cplib.math.floor_sum import floor_sum
from cplib.math.matrix import mat_mul, mat_pow
from cplib.math.mod import crt, ext_gcd, inv_mod
from cplib.math.prime import (
    divisors,
    factorize,
    is_prime,
    min_factor_table,
    primes,
)

MOD = 998244353


def test_comb() -> None:
    c = Comb(MOD, 10)
    for n in range(-2, 60):
        for r in range(-2, 62):
            ok = 0 <= r <= n
            assert c.comb(n, r) == (comb(n, r) % MOD if ok else 0)
            assert c.perm(n, r) == (perm(n, r) % MOD if ok else 0)
    rng = random.Random(0)
    for _ in range(100):
        n = rng.randint(1, 3000)
        assert c.fac(n) == factorial(n) % MOD
        assert c.finv(n) * c.fac(n) % MOD == 1
        assert c.inv(n) * n % MOD == 1


def test_comb_small_mod() -> None:
    c = Comb(13)
    for n in range(13):
        for r in range(n + 1):
            assert c.comb(n, r) == comb(n, r) % 13


@pytest.mark.parametrize("seed", range(20))
def test_floor_sum(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(200):
        n = rng.randint(0, 30)
        lim = rng.choice((30, 10**9))
        m = rng.randint(1, lim)
        a = rng.randint(-2 * lim, 2 * lim)
        b = rng.randint(-2 * lim, 2 * lim)
        assert floor_sum(n, m, a, b) == sum((a * i + b) // m for i in range(n))


def naive_is_prime(n: int) -> bool:
    return n >= 2 and all(n % d for d in range(2, int(n**0.5) + 1))


def test_primes_and_min_factor_table() -> None:
    for n in range(0, 200):
        assert primes(n) == [p for p in range(n + 1) if naive_is_prime(p)]
    t = min_factor_table(1000)
    assert t[:2] == [0, 1]
    for x in range(2, 1001):
        assert t[x] == next(d for d in range(2, x + 1) if x % d == 0)


def test_is_prime() -> None:
    sieve = set(primes(10**5))
    assert [n for n in range(-5, 10**5 + 1) if is_prime(n)] == sorted(sieve)
    # 強擬素数（2 などを底にすると素数と誤判定されやすい合成数）
    for n in (3215031751, 3825123056546413051):
        assert not is_prime(n)
    for n in (998244353, 10**9 + 7, 2**61 - 1, 18446744073709551557):
        assert is_prime(n)


def naive_factorize(n: int) -> dict[int, int]:
    res: dict[int, int] = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            res[d] = res.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        res[n] = res.get(n, 0) + 1
    return res


@pytest.mark.parametrize("seed", range(10))
def test_factorize(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(100):
        n = rng.randint(1, 10**7)
        got = factorize(n)
        assert got == naive_factorize(n)
        assert list(got) == sorted(got)
    ps = [p for p in (rng.randint(10**4, 10**6) for _ in range(30)) if is_prime(p)]
    for _ in range(20):
        expected: dict[int, int] = {}
        n = 1
        for p in rng.sample(ps, min(len(ps), 3)):
            e = rng.randint(1, 3)
            expected[p] = e
            n *= p**e
        assert factorize(n) == dict(sorted(expected.items()))


def test_factorize_large() -> None:
    p, q = 999999937, 1000000007
    assert factorize(p * q) == {p: 1, q: 1}
    assert factorize(p * p) == {p: 2}
    assert factorize(2**63) == {2: 63}
    assert factorize(1) == {}


def test_inv_mod() -> None:
    for m in range(1, 60):
        for x in range(-2 * m, 2 * m):
            if gcd(x, m) == 1:
                y = inv_mod(x, m)
                assert 0 <= y < m and x * y % m == 1 % m
            else:
                with pytest.raises(ValueError):
                    inv_mod(x, m)
    m = 10**18 + 9
    assert inv_mod(-(10**30), m) * -(10**30) % m == 1


def naive_crt(r: list[int], m: list[int]) -> tuple[int, int]:
    z = lcm(*m)
    for y in range(z):
        if all((y - ri) % mi == 0 for ri, mi in zip(r, m, strict=True)):
            return (y, z)
    return (0, 0)


@pytest.mark.parametrize("seed", range(20))
def test_crt(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(200):
        k = rng.randint(0, 4)
        m = [rng.randint(1, 12) for _ in range(k)]
        r = [rng.randint(-30, 30) for _ in range(k)]
        assert crt(r, m) == naive_crt(r, m)


def test_crt_large() -> None:
    ps = [998244353, 10**9 + 7, 10**9 + 9]
    x = 123456789123456789123456789 % (ps[0] * ps[1] * ps[2])
    assert crt([x % p for p in ps], ps) == (x, ps[0] * ps[1] * ps[2])
    m = [2**40 * 3, 2**20 * 5**10]
    x = 10**20
    assert crt([x % mi for mi in m], m) == (x % lcm(*m), lcm(*m))
    assert crt([x % mi + 1 for mi in m], m) == (x % lcm(*m) + 1, lcm(*m))
    assert crt([0, 1], m) == (0, 0)


def naive_convolution(mod: int, a: list[int], b: list[int]) -> list[int]:
    c = [0] * (len(a) + len(b) - 1) if a and b else []
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            c[i + j] += x * y
    return [x % mod for x in c]


@pytest.mark.parametrize("seed", range(10))
def test_convolution(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(30):
        # 愚直に切り替える境目（短い側の長さ 60）の前後を含める
        n = rng.choice((0, 1, 2, rng.randint(1, 200), rng.randint(55, 70)))
        m = rng.choice((0, 1, 2, rng.randint(1, 200), rng.randint(55, 70)))
        lim = rng.choice((10, MOD, 10**18))
        a = [rng.randint(-lim, lim) for _ in range(n)]
        b = [rng.randint(-lim, lim) for _ in range(m)]
        a0 = a[:]
        b0 = b[:]
        assert convolution(MOD, a, b) == naive_convolution(MOD, a, b)
        assert (a, b) == (a0, b0)


@pytest.mark.parametrize(
    ("mod", "n", "m"),
    [
        (MOD, 2000, 3000),
        (MOD, 1 << 11, 1 << 11),
        (167772161, 1000, 1025),
        (469762049, 700, 300),
        (754974721, 512, 513),
        # 長さ n + m - 1 を 2 の冪に切り上げたものが mod - 1 を割り切る上限ちょうど
        (257, 128, 128),
        (7681, 300, 200),
    ],
)
def test_convolution_ntt(mod: int, n: int, m: int) -> None:
    rng = random.Random(n * m)
    a = [rng.randrange(mod) for _ in range(n)]
    b = [rng.randrange(mod) for _ in range(m)]
    assert convolution(mod, a, b) == naive_convolution(mod, a, b)


def test_divisors() -> None:
    for n in range(1, 2000):
        assert divisors(n) == [d for d in range(1, n + 1) if n % d == 0]
    rng = random.Random(0)
    for _ in range(50):
        n = rng.randint(1, 10**12)
        expected = sorted(
            {x for d in range(1, isqrt(n) + 1) if n % d == 0 for x in (d, n // d)}
        )
        assert divisors(n) == expected


def test_divisors_large() -> None:
    # 10^18 以下で約数が最も多い数
    n = 897612484786617600
    got = divisors(n)
    assert len(got) == 103680
    assert got == sorted(set(got)) and all(n % d == 0 for d in got)
    p, q = 999999937, 1000000007
    assert divisors(p * q) == [1, p, q, p * q]
    assert divisors(2**63) == [2**i for i in range(64)]


def check_ext_gcd(a: int, b: int) -> None:
    g, x, y = ext_gcd(a, b)
    assert g == gcd(a, b)
    assert a * x + b * y == g
    if g:
        assert abs(x) <= max(abs(b) // g, 1) and abs(y) <= max(abs(a) // g, 1)


def test_ext_gcd() -> None:
    for a in range(-40, 41):
        for b in range(-40, 41):
            check_ext_gcd(a, b)
    rng = random.Random(0)
    for _ in range(1000):
        lim = rng.choice((10**9, 10**18, 10**30))
        check_ext_gcd(rng.randint(-lim, lim), rng.randint(-lim, lim))


def naive_mat_mul(a: list[list[int]], b: list[list[int]], mod: int) -> list[list[int]]:
    p = len(b[0]) if b else 0
    return [
        [sum(a[i][k] * b[k][j] for k in range(len(b))) % mod for j in range(p)]
        for i in range(len(a))
    ]


@pytest.mark.parametrize("seed", range(20))
def test_mat_mul(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(20):
        # 8 項ごとに剰余を取る境目の前後を含める
        n, m, p = (rng.choice((1, 2, 7, 8, 9, rng.randint(1, 30))) for _ in range(3))
        mod = rng.choice((1, 2, MOD, 10**9 + 7, 2**61 - 1))
        lim = rng.choice((mod - 1, 10**18))
        a = [[rng.randint(-lim, lim) for _ in range(m)] for _ in range(n)]
        b = [[rng.randint(-lim, lim) for _ in range(p)] for _ in range(m)]
        assert mat_mul(a, b, mod) == naive_mat_mul(a, b, mod)


def test_mat_mul_empty() -> None:
    assert mat_mul([], [], MOD) == []
    assert mat_mul([[], []], [], MOD) == [[], []]
    assert mat_mul([], [[1, 2]], MOD) == []


@pytest.mark.parametrize("seed", range(20))
def test_mat_pow(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(10):
        n = rng.randint(0, 6)
        mod = rng.choice((1, 7, MOD))
        a = [[rng.randint(-(10**9), 10**9) for _ in range(n)] for _ in range(n)]
        expected = [[int(i == j) % mod for j in range(n)] for i in range(n)]
        for k in range(20):
            assert mat_pow(a, k, mod) == expected
            expected = naive_mat_mul(expected, a, mod)


def test_mat_pow_large() -> None:
    # 2 × 2 のフィボナッチ行列の k 乗は [[F(k+1), F(k)], [F(k), F(k-1)]]
    k = 10**18
    f = mat_pow([[1, 1], [1, 0]], k, MOD)
    f2 = mat_pow([[1, 1], [1, 0]], 2 * k, MOD)
    # F(2k) = F(k) (2 F(k+1) - F(k))
    assert f2[0][1] == f[0][1] * (2 * f[0][0] - f[0][1]) % MOD
    a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert mat_pow(a, 10**18 + 1, MOD) == mat_mul(mat_pow(a, 10**18, MOD), a, MOD)
