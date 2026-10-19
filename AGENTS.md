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
- `website/scripts/build_media.py`：ローカルで例題を解き、Matplotlibで日英の動画・図と
  計算結果JSONを `website/public/media/` に生成する。生成物もGitで管理する。
- `website/scripts/check_media.py`：標準ライブラリだけで生成物の不足・更新漏れ・求解失敗を検出する。

## コマンド

```sh
uv sync --extra autodiff
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run python -m unittest discover -s tests
# 図・動画の入力を変更した場合だけ実行（ffmpegと日本語フォントが必要）
uv run python website/scripts/build_media.py [ids...]
cd website && npm ci && npm run check:media && npm run build && npm run check:site
```

ライブラリのCI（`.github/workflows/ci.yml`）はlint・型・テスト・パッケージビルドを実行する。
公開ワークフロー（`docs.yml`）はコミット済みメディアの検査・サイトのビルド・リンク検査を行い、
`master` のサイトをGitHub Pagesへ公開する。Actionsでは数値計算・動画生成を行わない。

## 方針

- API仕様はdocstringに書き、手書きで二重管理しない。
- READMEは入口に留め、詳細はサイトに書く。数式・コード・結果の対応が分かる説明を重視する。
- 動きは事前生成したMP4と静止画で見せる。React等のブラウザ側可視化や、閲覧者による
  再計算は必要になるまで追加しない。
- 例題・入力CSV・ライブラリ・描画処理・翻訳ラベル・`uv.lock` を変更したら、ローカルで
  メディアを再生成し、`website/public/media/` の図・動画・JSON・ハッシュをソースと一緒に
  コミットする。文章だけの変更では再生成しない。ハッシュを手で更新して検査を回避しない。
  対象IDの指定で一部を生成できるが、公開前は `npm run check:media` で全体の更新漏れを検査する。
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
