import random
from math import comb, factorial, perm

import pytest

from cplib.math.comb import Comb
from cplib.math.floor_sum import floor_sum
from cplib.math.prime import factorize, is_prime, min_factor_table, primes

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
