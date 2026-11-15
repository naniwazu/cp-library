def _convolution_rates(mod: int, z: int) -> tuple[int, int, list[int], list[int]]:
    # 基数 4 のバタフライで使う 1 の 4 乗根とその逆数、ブロック間の回転因子の比を返す
    rank2 = ((mod - 1) & (1 - mod)).bit_length() - 1
    assert z <= 1 << rank2, "mod - 1 が 2 の冪で十分に割り切れない"
    # NTT 向きの素数では mod - 1 の奇数部分が小さいので、試し割りで足りる
    x = (mod - 1) >> rank2
    ps = [2]
    d = 3
    while d * d <= x:
        if x % d == 0:
            ps.append(d)
            while x % d == 0:
                x //= d
        d += 2
    if x > 1:
        ps.append(x)
    g = 2
    while any(pow(g, (mod - 1) // p, mod) == 1 for p in ps):
        g += 1
    # root[i] は 1 の原始 2^i 乗根
    root = [0] * (rank2 + 1)
    iroot = [0] * (rank2 + 1)
    root[rank2] = pow(g, (mod - 1) >> rank2, mod)
    iroot[rank2] = pow(root[rank2], mod - 2, mod)
    for i in range(rank2 - 1, -1, -1):
        root[i] = root[i + 1] * root[i + 1] % mod
        iroot[i] = iroot[i + 1] * iroot[i + 1] % mod
    rate3 = [0] * (rank2 - 2)
    irate3 = [0] * (rank2 - 2)
    prod = iprod = 1
    for i in range(rank2 - 2):
        rate3[i] = root[i + 3] * prod % mod
        irate3[i] = iroot[i + 3] * iprod % mod
        prod = prod * iroot[i + 3] % mod
        iprod = iprod * root[i + 3] % mod
    return root[2], iroot[2], rate3, irate3


def _convolution_butterfly(a: list[int], mod: int, imag: int, rate3: list[int]) -> None:
    # 周波数間引きの NTT。結果はビット反転順に並ぶ
    n = len(a)
    h = (n - 1).bit_length()
    le = 0
    if h & 1:
        # 基数 2 の段を回転因子が 1 になる先頭に置き、残りを基数 4 で処理する
        p = n >> 1
        for i in range(p):
            l = a[i]
            r = a[i + p]
            a[i] = (l + r) % mod
            a[i + p] = (l - r) % mod
        le = 1
    while le < h:
        p = 1 << (h - le - 2)
        rot = 1
        for s in range(1 << le):
            if s:
                rot = rot * rate3[(s & -s).bit_length() - 1] % mod
            rot2 = rot * rot % mod
            rot3 = rot2 * rot % mod
            offset = s << (h - le)
            for i in range(offset, offset + p):
                j = i + p
                k = j + p
                t = k + p
                # 積は 2^60 未満なので、4 つ足しても PyPy の機械語整数に収まる
                a0 = a[i]
                a1 = a[j] * rot
                a2 = a[k] * rot2
                a3 = a[t] * rot3
                x = (a1 - a3) % mod * imag
                a[i] = (a0 + a2 + a1 + a3) % mod
                a[j] = (a0 + a2 - a1 - a3) % mod
                a[k] = (a0 - a2 + x) % mod
                a[t] = (a0 - a2 - x) % mod
        le += 2


def _convolution_butterfly_inv(
    a: list[int], mod: int, iimag: int, irate3: list[int]
) -> None:
    # _convolution_butterfly の逆変換（n 倍される）。段を逆順にたどる
    n = len(a)
    h = (n - 1).bit_length()
    le = h
    while le > 1:
        p = 1 << (h - le)
        irot = 1
        for s in range(1 << (le - 2)):
            if s:
                irot = irot * irate3[(s & -s).bit_length() - 1] % mod
            irot2 = irot * irot % mod
            irot3 = irot2 * irot % mod
            offset = s << (h - le + 2)
            for i in range(offset, offset + p):
                j = i + p
                k = j + p
                t = k + p
                a0 = a[i]
                a1 = a[j]
                a2 = a[k]
                a3 = a[t]
                x = (a2 - a3) * iimag % mod
                a[i] = (a0 + a1 + a2 + a3) % mod
                a[j] = (a0 - a1 + x) * irot % mod
                a[k] = (a0 + a1 - a2 - a3) * irot2 % mod
                a[t] = (a0 - a1 - x) * irot3 % mod
        le -= 2
    if le:
        p = n >> 1
        for i in range(p):
            l = a[i]
            r = a[i + p]
            a[i] = (l + r) % mod
            a[i + p] = (l - r) % mod


def convolution(mod: int, a: list[int], b: list[int]) -> list[int]:
    """c[k] = sum_{i + j = k} a[i] * b[j] mod mod の列 c（長さ n + m - 1）を返す。

    多項式の積や、2 つの列の「和が k になる組」の数え上げに使う。NTT で O((n + m)
    log (n + m))。mod は 998244353 のように、2 の冪 z >= n + m - 1 について
    z が mod - 1 を割り切る素数であること。どちらかが空なら空リストを返す。
    a, b の要素は負や mod 以上でもよく、a, b は書き換えない。
    ac-library-python の convolution と同じ API。

    Examples:
        >>> convolution(998244353, [1, 2, 3], [4, 5])
        [4, 13, 22, 15]
        >>> convolution(998244353, [-1], [2, 3]), convolution(998244353, [], [1])
        ([998244351, 998244350], [])
    """
    n = len(a)
    m = len(b)
    if not n or not m:
        return []
    if min(n, m) <= 60:
        # 短い側が短ければ愚直のほうが速い（境目は PyPy での計測による）
        if n < m:
            n, m, a, b = m, n, b, a
        a2 = [x % mod for x in a]
        c = [0] * (n + m - 1)
        for j in range(m):
            bj = b[j] % mod
            for i in range(n):
                c[i + j] = (c[i + j] + a2[i] * bj) % mod
        return c
    z = 1 << (n + m - 2).bit_length()
    imag, iimag, rate3, irate3 = _convolution_rates(mod, z)
    a2 = [0] * z
    for i in range(n):
        a2[i] = a[i] % mod
    b2 = [0] * z
    for i in range(m):
        b2[i] = b[i] % mod
    _convolution_butterfly(a2, mod, imag, rate3)
    _convolution_butterfly(b2, mod, imag, rate3)
    iz = pow(z, mod - 2, mod)
    for i in range(z):
        a2[i] = a2[i] * b2[i] % mod * iz % mod
    _convolution_butterfly_inv(a2, mod, iimag, irate3)
    del a2[n + m - 1 :]
    return a2
