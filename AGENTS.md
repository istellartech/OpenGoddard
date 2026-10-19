# OpenGoddard

擬スペクトル法による最適軌道計算のPythonライブラリ。Python 3.11、uvで管理。

## 構成

- `src/OpenGoddard/`：ライブラリ本体（import名 `OpenGoddard`）。公開APIは `__init__.py` の `__all__`。
- `examples/NN_<id>.py`：15の例題。番号は学習順、`catalog.py` のIDは安定識別子。
  生成物は `examples/output/<id>/`（Git対象外）。学習者向け本体は普通の関数で書く。
- `examples/data/`：例題14が使う入力CSV。
- `docs/`：日本語READMEと過去の参考図（`figures/<id>/`）。解説サイトのソースは `website/`。
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
- 例題はパラメータ・モデル構築・求解・描画を分離し、動画生成でも再利用する。
- 1例題は1つの物理モデルと1つの目的関数に絞る。プリセットや目的関数・モデルの切り替えは設けない。
- 例題本体には数式・制約・目的関数・初期推定を明記する。初級ではデコレータ・
  データクラス・自動微分用の記法・CLI処理を使わず、文法も段階的に導入する。
- 番号付き例題は補助ファイルをimportせず、そのファイルだけで条件設定から描画まで読める形にする。
  コメントには変数の意味・単位・具体的な計算を書く。
- サイトは `base: '/OpenGoddard'` で公開される。リンクやメディアのパスはこれに合わせる。
- サイトの日本語はルート、英語は `/en/`。`website/src/content/docs/` の本文を変更するときは
  対応する `en/` のページも更新する。APIは両言語とも同じdocstringから生成する。
  図・動画のラベルは `website/scripts/i18n.py` で翻訳し、計算結果は両言語で共有する。
  サイトのビルド後は `cd website && npm run check:site` で言語切り替え・リンク・メディアを検証する。
