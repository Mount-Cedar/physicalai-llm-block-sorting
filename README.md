# AI×ロボティクス課題: 自然言語指示によるブロック仕分け

Genesis シミュレータ上の Franka Panda アームが、日本語の指示文を
LLM(Claude API)に解釈させて得た実行計画に従い、色付きブロックを
同色のトレイへ仕分けるデモです。

## セットアップ

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## 実行

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # Claude API キー
./venv/bin/python src/sorting_task.py
```

- 動画は `output/video.mp4`、LLM が生成した計画は `output/plan.json` に保存されます。
- API キーなしで動作確認する場合: `./venv/bin/python src/sorting_task.py --offline`
  (AI 不使用の固定ルール。提出用の実行には使いません)

## ファイル構成

- `src/sorting_task.py` — シミュレーション本体(シーン構築・制御・録画)
- `src/llm_planner.py` — Claude API による自然言語→実行計画の推論
- `report.md` — 課題レポート
