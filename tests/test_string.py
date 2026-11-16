import random
from itertools import pairwise

import pytest

from cplib.string.manacher import manacher
from cplib.string.rolling_hash import RollingHash
from cplib.string.suffix_array import lcp_array, suffix_array
from cplib.string.trie import Trie
from cplib.string.z_algorithm import z_algorithm


@pytest.mark.parametrize("seed", range(20))
def test_rolling_hash(seed: int) -> None:
    rng = random.Random(seed)
    s = "".join(rng.choice("ab") for _ in range(rng.randint(0, 30)))
    a = [ord(c) for c in s]
    t = a[::-1]
    hs = RollingHash(s)
    ht = RollingHash(t)
    n = len(s)
    for _ in range(300):
        k = rng.randint(0, n)
        i = rng.randint(0, n - k)
        j = rng.randint(0, n - k)
        assert (hs.get(i, i + k) == hs.get(j, j + k)) == (s[i : i + k] == s[j : j + k])
        assert (hs.get(i, i + k) == ht.get(j, j + k)) == (a[i : i + k] == t[j : j + k])


def test_rolling_hash_large_values() -> None:
    big = (1 << 61) - 2
    h = RollingHash([big, big, 0, big, big])
    assert h.get(0, 2) == h.get(3, 5)
    assert h.get(0, 2) != h.get(1, 3)


@pytest.mark.parametrize("seed", range(20))
def test_z_algorithm(seed: int) -> None:
    rng = random.Random(seed)
    s = "".join(
        rng.choice("abc"[: rng.randint(1, 3)]) for _ in range(rng.randint(0, 40))
    )
    expected: list[int] = []
    for i in range(len(s)):
        k = 0
        while i + k < len(s) and s[k] == s[i + k]:
            k += 1
        expected.append(k)
    assert z_algorithm(s) == expected
    assert z_algorithm([ord(c) for c in s]) == expected


def naive_lcp_array(a: list[int], sa: list[int]) -> list[int]:
    lcp: list[int] = []
    for x, y in pairwise(sa):
        k = 0
        while x + k < len(a) and y + k < len(a) and a[x + k] == a[y + k]:
            k += 1
        lcp.append(k)
    return lcp


@pytest.mark.parametrize("seed", range(200))
def test_suffix_array_and_lcp_array(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.choice([0, 1, rng.randint(2, 10), rng.randint(2, 200)])
    upper = rng.choice([0, 1, 2, 25, 10**9])
    a = [rng.randint(0, upper) for _ in range(n)]
    expected = sorted(range(n), key=lambda i: a[i:])
    expected_lcp = naive_lcp_array(a, expected)
    assert suffix_array(a) == expected
    if upper < 100:
        assert suffix_array(a, upper) == expected
        s = "".join(chr(ord("a") + x) for x in a)
        assert suffix_array(s) == expected
        assert lcp_array(s, expected) == expected_lcp
    assert lcp_array(a, expected) == expected_lcp


@pytest.mark.parametrize("n", range(50))
def test_suffix_array_same_characters(n: int) -> None:
    assert suffix_array("a" * n) == list(range(n - 1, -1, -1))
    assert lcp_array("a" * n, list(range(n - 1, -1, -1))) == list(range(1, n))


def test_suffix_array_non_ascii() -> None:
    s = "あいあ"
    assert suffix_array(s) == sorted(range(len(s)), key=lambda i: s[i:])


@pytest.mark.parametrize("seed", range(50))
def test_manacher(seed: int) -> None:
    rng = random.Random(seed)
    n = rng.choice([0, 1, 2, rng.randint(3, 40)])
    s = "".join(rng.choice("abc"[: rng.randint(1, 3)]) for _ in range(n))
    expected = [0] * (2 * n + 1)
    for l in range(n + 1):
        for r in range(l, n + 1):
            if s[l:r] == s[l:r][::-1]:
                expected[l + r] = max(expected[l + r], r - l)
    a = manacher(s)
    assert a == expected
    assert manacher([ord(c) for c in s]) == expected
    for l in range(n + 1):
        for r in range(l, n + 1):
            assert (r - l <= a[l + r]) == (s[l:r] == s[l:r][::-1])


def test_trie_empty() -> None:
    t = Trie()
    assert t.count("") == t.count_prefix("") == 0
    assert t.count("a") == t.count_prefix("a") == 0
    assert t.child(0, "a") == -1
    t.add("")
    assert t.count("") == t.count_prefix("") == 1
    assert t.count("a") == t.count_prefix("a") == 0
    assert t.prefix_counts == t.counts == [1]


def _trie_walk(t: Trie, s: str | list[int]) -> int:
    v = 0
    for c in s:
        v = t.child(v, c)
        if v < 0:
            break
    return v


@pytest.mark.parametrize("seed", range(20))
def test_trie(seed: int) -> None:
    rng = random.Random(seed)
    alphabet = "abc"[: rng.randint(1, 3)]

    def random_string() -> str:
        return "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 6)))

    t = Trie()
    ti = Trie()
    added: list[str] = []
    for _ in range(100):
        s = random_string()
        t.add(s)
        # 整数列は負の値や 32 ビットを超える値でも同じに振る舞う
        ti.add([(ord(c) - 98) * 10**12 for c in s])
        added.append(s)
        q = random_string()
        qi = [(ord(c) - 98) * 10**12 for c in q]
        expected = added.count(q)
        expected_prefix = sum(x.startswith(q) for x in added)
        assert t.count(q) == ti.count(qi) == expected
        assert t.count_prefix(q) == ti.count_prefix(qi) == expected_prefix
        v = _trie_walk(t, q)
        vi = _trie_walk(ti, qi)
        if expected_prefix:
            assert t.prefix_counts[v] == ti.prefix_counts[vi] == expected_prefix
            assert t.counts[v] == ti.counts[vi] == expected
        else:
            assert v == vi == -1
    assert len(t.prefix_counts) == len(t.counts) == len(ti.counts)
