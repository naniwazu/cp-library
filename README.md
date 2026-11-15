# cp-library

競技プログラミング用のライブラリとテンプレート。

| ディレクトリ | 中身 |
| --- | --- |
| `cplib/` | Python (PyPy) 用ライブラリ本体（`ds` / `graph` / `math` / `string` / `misc` に分類） |
| `python/` | Python 用テンプレート |
| `tests/` | ライブラリと展開スクリプトのテスト |
| `cpp/` | C++ 用テンプレート |
| `snippets/` | VSCode 用の入力スニペット |

Python では `from cplib.ds.dsu import DSU` のように import して書き、提出前に
`uv run cplib-expand main.py > submit.py` でライブラリ本体を 1 ファイルに展開する。

開発時のチェック:

```sh
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest
```

AtCoder の生成AI対策ルールに従い、生成AIを使って書いたコードもここで事前に公開し、
提出時にはこのリポジトリの URL を記載している。

ac-library-python と同じ機能は、名前と引数をそれに合わせて自前で実装している。
未実装のもの（CRT、suffix array など）は `atcoder` を使う。
