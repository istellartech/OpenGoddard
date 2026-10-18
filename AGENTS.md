# OpenGoddard

擬スペクトル法による最適軌道計算のPythonライブラリ。Python 3.15、uvで管理。

## 構成

- `src/OpenGoddard/`：ライブラリ本体（import名 `OpenGoddard`）。公開APIは `__init__.py` の `__all__`。
- `examples/NN_<id>/main.py`：15の例題。番号は学習順、`catalog.py` のIDは安定識別子。
  生成物は各例題の `output/`（Git対象外）。共通処理は `_shared.py`。
- `tests/`：unittest。
- `website/`：Astro + Starlight + MDX の解説サイト。APIリファレンスは
  starlight-pydocs（Griffe）がdocstring（Google形式）から生成する。
- `website/scripts/build_media.py`：例題を解いてMatplotlibで動画・図を `website/public/media/` に生成する。

## コマンド

```sh
uv sync --extra autodiff
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run python -m unittest discover -s tests
uv run python website/scripts/build_media.py [ids...]   # 動画・図
cd website && npm ci && npm run build                     # サイト
```

CI（`.github/workflows/`）は上記を実行し、`master` のサイトをGitHub Pagesへ公開する。

## 方針

- API仕様はdocstringに書き、手書きで二重管理しない。
- READMEは入口に留め、詳細はサイトに書く。数式・コード・結果の対応が分かる説明を重視する。
- 動きは事前生成したMP4と静止画で見せる。React等のブラウザ側可視化や、閲覧者による
  再計算は必要になるまで追加しない。
- 例題はパラメータ・求解・結果データ・描画を分離し、動画生成でも再利用する。
- サイトは `base: '/OpenGoddard'` で公開される。リンクやメディアのパスはこれに合わせる。
