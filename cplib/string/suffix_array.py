from collections.abc import Sequence


def _sa_induce(
    s: list[int], ls: list[bool], sum_s: list[int], sum_l: list[int], lms: list[int]
) -> list[int]:
    # lms の順に LMS 接尾辞を置き、そこから L 型、S 型の接尾辞を順に誘導する
    n = len(s)
    sa = [-1] * n
    buf = sum_s[:]
    for d in lms:
        c = s[d]
        sa[buf[c]] = d
        buf[c] += 1
    buf = sum_l[:]
    c = s[n - 1]
    sa[buf[c]] = n - 1
    buf[c] += 1
    for i in range(n):
        v = sa[i] - 1
        if v >= 0 and not ls[v]:
            c = s[v]
            sa[buf[c]] = v
            buf[c] += 1
    buf = sum_l[:]
    for i in range(n - 1, -1, -1):
        v = sa[i] - 1
        if v >= 0 and ls[v]:
            c = s[v] + 1
            buf[c] -= 1
            sa[buf[c]] = v
    return sa


def _sa_is(s: list[int], upper: int) -> list[int]:
    # SA-IS。各要素は 0 以上 upper 以下
    n = len(s)
    if n < 2:
        return list(range(n))
    # ls[i] は s[i:] が S 型（s[i:] < s[i + 1:]）かどうか
    ls = [False] * n
    for i in range(n - 2, -1, -1):
        a = s[i]
        b = s[i + 1]
        ls[i] = ls[i + 1] if a == b else a < b
    # 文字 c のバケットは、L 型の区間 [sum_s[c], sum_l[c + 1]) と
    # S 型の区間 [sum_l[c + 1], sum_s[c + 1]) からなる
    sum_l = [0] * (upper + 1)
    sum_s = [0] * (upper + 1)
    for i in range(n):
        if ls[i]:
            sum_l[s[i] + 1] += 1
        else:
            sum_s[s[i]] += 1
    for i in range(upper + 1):
        sum_s[i] += sum_l[i]
        if i < upper:
            sum_l[i + 1] += sum_s[i]
    lms_map = [-1] * n
    lms: list[int] = []
    for i in range(1, n):
        if ls[i] and not ls[i - 1]:
            lms_map[i] = len(lms)
            lms.append(i)
    m = len(lms)
    sa = _sa_induce(s, ls, sum_s, sum_l, lms)
    if m == 0:
        return sa
    # 誘導後の LMS 部分文字列の順に名前を振り、
    # 名前の列の接尾辞配列から LMS 接尾辞の順を確定させる
    sorted_lms = [v for v in sa if lms_map[v] != -1]
    rec_s = [0] * m
    rec_upper = 0
    for i in range(1, m):
        left = sorted_lms[i - 1]
        right = sorted_lms[i]
        jl = lms_map[left] + 1
        jr = lms_map[right] + 1
        end_l = lms[jl] if jl < m else n
        end_r = lms[jr] if jr < m else n
        same = False
        if end_l - left == end_r - right:
            while left < end_l and s[left] == s[right]:
                left += 1
                right += 1
            same = left < n and s[left] == s[right]
        if not same:
            rec_upper += 1
        rec_s[jr - 1] = rec_upper
    # 再帰の深さは O(log n)（m <= n / 2）なので、規約の例外として再帰で書く
    rec_sa = _sa_is(rec_s, rec_upper)
    return _sa_induce(s, ls, sum_s, sum_l, [lms[i] for i in rec_sa])


def suffix_array(s: str | Sequence[int], upper: int | None = None) -> list[int]:
    """s の接尾辞配列（接尾辞を辞書順に並べたときの開始位置のリスト）を返す。

    lcp_array と組み合わせて、異なる部分文字列の個数（n(n+1)/2 - sum(lcp)）、
    パターンの出現位置の二分探索、区切り文字で連結した 2 文字列の
    最長共通部分文字列などを求める。
    SA-IS で、O(n + upper)。
    s が整数列のとき、upper を渡すと各要素が 0 以上 upper 以下である必要があり、
    省略すると座標圧縮してから計算する（O(n log n)）。

    Examples:
        >>> suffix_array("abracadabra")
        [10, 7, 0, 3, 5, 8, 1, 4, 6, 9, 2]
        >>> suffix_array([3, 1, 3, 1]), suffix_array([2, 0, 2, 0], upper=2)
        ([3, 1, 2, 0], [3, 1, 2, 0])
    """
    if isinstance(s, str):
        a = [ord(c) for c in s]
        return _sa_is(a, max(a, default=0))
    if upper is None:
        rank = {v: i for i, v in enumerate(sorted(set(s)))}
        return _sa_is([rank[v] for v in s], max(len(rank) - 1, 0))
    assert 0 <= upper
    assert all(0 <= v <= upper for v in s)
    return _sa_is(list(s), upper)


def lcp_array(s: str | Sequence[int], sa: Sequence[int]) -> list[int]:
    """接尾辞配列 sa で隣り合う接尾辞どうしの最長共通接頭辞の長さのリストを返す。

    長さは len(s) - 1 で、i 番目の値は s[sa[i]:] と s[sa[i + 1]:] の
    最長共通接頭辞の長さ。
    使いどころは suffix_array を参照。Kasai 法で、O(n)。

    Examples:
        >>> s = "abab"
        >>> sa = suffix_array(s)
        >>> lcp = lcp_array(s, sa)
        >>> sa, lcp
        ([2, 0, 3, 1], [2, 0, 1])
        >>> len(s) * (len(s) + 1) // 2 - sum(lcp)  # 異なる部分文字列の個数
        7
    """
    a = [ord(c) for c in s] if isinstance(s, str) else s
    n = len(a)
    rnk = [0] * n
    for i, v in enumerate(sa):
        rnk[v] = i
    lcp = [0] * max(n - 1, 0)
    h = 0
    # s[i:] と辞書順で 1 つ前の接尾辞の LCP が h なら、s[i + 1:] についての LCP は
    # h - 1 以上なので、h を持ち越して比較を省く（合計 O(n)）
    for i in range(n):
        if h:
            h -= 1
        r = rnk[i]
        if r == 0:
            continue
        j = sa[r - 1]
        while i + h < n and j + h < n and a[i + h] == a[j + h]:
            h += 1
        lcp[r - 1] = h
    return lcp
