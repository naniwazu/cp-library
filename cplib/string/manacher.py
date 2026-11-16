from collections.abc import Sequence


def manacher(s: Sequence[object]) -> list[int]:
    """各中心について最長の回文の長さを並べた長さ 2|s| + 1 のリストを O(|s|) で返す。

    戻り値を a とすると、a[i] は l + r == i を満たす回文 s[l:r] の長さの最大値。
    i が奇数なら s[(i - 1) // 2] を中心とする奇数長の回文、偶数なら s[i // 2 - 1] と
    s[i // 2] の間を中心とする偶数長の回文（両端の a[0] と a[2|s|] は 0）に当たる。
    s[l:r] が回文であることは r - l <= a[l + r] と同値なので、前計算のあとは
    任意の s[l:r] が回文かを O(1) で判定できる（l == r の空列も含む）。

    最長の回文である部分文字列の長さは max(a)、回文である空でない部分文字列の個数
    （位置が異なれば別に数える）は sum((x + 1) // 2 for x in a) で求まる。

    Examples:
        >>> manacher("abaab")
        [0, 1, 0, 3, 0, 1, 4, 1, 0, 1, 0]
        >>> s = "abaab"
        >>> a = manacher(s)
        >>> 4 <= a[1 + 5], 3 <= a[0 + 3]  # s[1:5] == "baab", s[0:3] == "aba"
        (True, True)
        >>> 2 <= a[0 + 2]  # s[0:2] == "ab"
        False
        >>> manacher("")
        [0]
    """
    n = len(s)
    m = 2 * n + 1
    a = [0] * m
    # 区切りを挟んだ列 t（t[2j + 1] == s[j]、偶数番目は区切り）の上の奇数長 Manacher。
    # a[i] は t[i - k:i + k + 1] が回文になる最大の k で、区切りどうしは常に一致するので
    # 比較するのは s の文字どうしだけでよい
    i = 0
    k = 0
    while i < m:
        while i - k > 0 and i + k < 2 * n:
            if not (i + k) & 1 and s[(i - k - 2) >> 1] != s[(i + k) >> 1]:
                break
            k += 1
        a[i] = k
        # 中心 i の回文の内側では、左右対称の位置の値を写せる
        j = 1
        while j <= i and j + a[i - j] < k:
            a[i + j] = a[i - j]
            j += 1
        i += j
        k -= j
    return a
