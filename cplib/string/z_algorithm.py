from collections.abc import Sequence


def z_algorithm(s: Sequence[object]) -> list[int]:
    """各 i について s と s[i:] の最長共通接頭辞の長さを並べたリストを O(|s|) で返す。

    パターン p の文字列 t 中の出現位置を、p + 区切り + t の Z 配列から求めるのに使う。
    ac-library と同じく先頭の値は len(s)。

    Examples:
        >>> z_algorithm("abacaba")
        [7, 0, 1, 0, 3, 0, 1]
        >>> z_algorithm([])
        []
    """
    n = len(s)
    z = [0] * n
    if n == 0:
        return z
    z[0] = n
    i = 1
    j = 0
    # z[i..i+k) のうち、既知の一致区間 s[i:i+j] に収まるものは z[1..k) から写せる
    while i < n:
        while i + j < n and s[j] == s[i + j]:
            j += 1
        z[i] = j
        if j == 0:
            i += 1
            continue
        k = 1
        while i + k < n and k + z[k] < j:
            z[i + k] = z[k]
            k += 1
        i += k
        j -= k
    return z
