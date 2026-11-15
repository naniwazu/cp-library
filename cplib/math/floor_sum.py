def floor_sum(n: int, m: int, a: int, b: int) -> int:
    """sum_{i=0}^{n-1} floor((a * i + b) / m) を O(log m) で返す（0 <= n、1 <= m）。

    格子点の数え上げや、等差数列を m で割った商・余りの総和を求める問題に使う。
    a、b は負でもよい。

    Examples:
        >>> floor_sum(4, 10, 6, 3)
        3
        >>> floor_sum(3, 2, -1, 0)
        -2
    """
    # Python の // は床関数なので、負の a, b もここで [0, m) に寄せられる
    ans = (a // m) * (n * (n - 1) // 2) + (b // m) * n
    a %= m
    b %= m
    # 以後のループでは常に 0 <= b < m が保たれ、a だけ寄せ直せばよい
    while True:
        y = a * n + b
        if y < m:
            return ans
        u = y // m
        v = u * m - b
        ans += (n - (v + a - 1) // a) * u
        n, m, a, b = u, a, m, (a - v) % a
        ans += (a // m) * (n * (n - 1) // 2)
        a %= m
