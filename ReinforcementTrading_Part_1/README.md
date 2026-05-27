# ReinforcementTrading Part 1

Pipeline chính của dự án trading bot nằm trong thư mục này. Bản nâng cấp đã được gom lại để không tạo thêm một dự án riêng ở root: UI, trainer, live signal, model artifacts và báo cáo bổ sung đều nằm dưới `ReinforcementTrading_Part_1/`.

> Dự án dùng cho nghiên cứu, backtest và paper-trading. Code không đặt lệnh thật và không phải lời khuyên tài chính.

## Cấu Trúc Gọn

```text
ReinforcementTrading_Part_1/
├── trading_bot/                         # Pipeline PPO chung cho BTC/PAXG/coin Binance USDT
│   ├── data.py                          # Lấy dữ liệu Binance API
│   ├── features.py                      # Technical + volume/order-flow features
│   ├── env.py                           # TradingEnv chung, risk-normalized
│   ├── trainer.py                       # Train/evaluate/save artifact
│   ├── evaluation.py                    # Metrics, baseline, stress test, chart
│   ├── live.py                          # Latest signal từ model đã lưu
│   └── ui/app.py                        # Dashboard Streamlit
├── artifacts/models/<SYMBOL>/<TF>/<RUN>/ # Model, metrics, chart theo từng tài sản
├── reports/TTCS_MOCK2_BoSung.docx       # DOCX bổ sung báo cáo
├── streamlit_app.py                     # Entry UI quen thuộc
├── training_runner.py                   # Wrapper tương thích cho code cũ
├── run_ui.sh                            # Chạy UI nhanh
├── requirements.txt
├── model_btc_best.zip                   # Model BTC 10M cũ vẫn giữ
└── ReinforcementTrading_Gold/model_gold_best.zip
```

Folder `ReinforcementTrading_Gold/` và các script cũ vẫn được giữ để không mất model 10M/reference cũ. Pipeline mới không cần tách BTC/Gold nữa: `BTCUSDT`, `PAXGUSDT`, `NEARUSDT`, `ETHUSDT`, `SOLUSDT`... đều chạy qua `trading_bot/`.

## Cài Đặt

Chạy từ đúng thư mục này:

```bash
cd ReinforcementTrading_Part_1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Chạy Dashboard

Cách nhanh:

```bash
./run_ui.sh
```

Cách thủ công:

```bash
cd ReinforcementTrading_Part_1
source .venv/bin/activate
export PYTHONPATH="$PWD"
python -m streamlit run streamlit_app.py
```

Lưu ý lỗi bạn gặp trước đó: lệnh đúng là `streamlit`, không phải `treamlit`.

## Train Từ UI

Dashboard hỗ trợ:

- Fetch danh sách cặp `USDT` đang trading từ Binance `exchangeInfo`.
- Card watchlist: `BTCUSDT`, `ETHUSDT`, `NEARUSDT`, `SOLUSDT`, `BNBUSDT`, `XRPUSDT`, `ADAUSDT`, `DOGEUSDT`, `PAXGUSDT`.
- Tài sản đã có model sẽ có viền sáng và hiện return/drawdown mới nhất.
- Chọn symbol, timeframe, số timesteps, reward mode, policy type, Optuna trials rồi bấm `Run`.
- Sau train sẽ hiện metrics và hình: equity curve, drawdown curve, baseline comparison, stress-test comparison.

Smoke test nên dùng:

```text
Symbol: BTCUSDT
Timeframe: 1h
Timesteps: 10,000 đến 50,000
Lookback days: 180 đến 730
Reward mode: pnl_drawdown
Policy type: mlp
Optuna trials: 0
```

Run BTC chính cho báo cáo:

```text
Symbol: BTCUSDT
Timeframe: 1h
Timesteps: 2,000,000
Lookback days: 730
Reward mode: pnl_drawdown
Policy type: mlp
```

## Train Từ CLI

```bash
cd ReinforcementTrading_Part_1
source .venv/bin/activate
export PYTHONPATH="$PWD"
python -m trading_bot.cli train --symbol BTCUSDT --timeframe 1h --timesteps 50000 --lookback-days 365
```

Run BTC 2M:

```bash
python -m trading_bot.cli train \
  --symbol BTCUSDT \
  --timeframe 1h \
  --timesteps 2000000 \
  --lookback-days 730 \
  --reward-mode pnl_drawdown \
  --policy-type mlp \
  --run-name btc_2m_risk_normalized_20260527
```

## Latest Signal

Lệnh này đọc artifact mới nhất, fetch nến Binance mới và in khuyến nghị. Nó không đặt lệnh thật.

```bash
cd ReinforcementTrading_Part_1
source .venv/bin/activate
export PYTHONPATH="$PWD"
python -m trading_bot.cli signal --symbol BTCUSDT --timeframe 1h
```

## Artifact

Mỗi lần train lưu vào:

```text
artifacts/models/<SYMBOL>/<TIMEFRAME>/<RUN_ID>/
├── model.zip
├── train_config.json
├── train_stats.npz
├── metrics.json
├── baseline_metrics.json
├── walk_forward_metrics.json
├── stress_test_metrics.json
├── overfit_report.json
├── equity_curve.png
├── drawdown_curve.png
├── baseline_comparison.png
└── stress_test_comparison.png
```

## Chống Overfit

Bản nâng cấp ghi lại:

- Split train/validation/test theo thời gian, không shuffle.
- Chọn best checkpoint theo validation, không lấy checkpoint cuối một cách mù quáng.
- So với Buy & Hold, MA crossover, RSI rule và random policy.
- Walk-forward-style segment metrics.
- Stress test khi tăng spread/slippage/fee.
- PBO-style warning report dựa trên ranking OOS.
- Train-only normalization trong `train_stats.npz`.
- Risk-normalized sizing: SL/TP theo basis points, mỗi lệnh rủi ro tối đa `1%` equity, giới hạn notional và không cho equity âm.

## Kết Quả BTC 2M Hiện Tại

Artifact chính:

```text
artifacts/models/BTCUSDT/1h/btc_2m_risk_normalized_20260527/
```

PPO test-only:

```text
Final equity: 9,766.97
Return: -2.33%
Max drawdown: -14.92%
Sharpe simple: -0.014
Trades: 343
Win rate: 35.28%
```

Baseline cùng test period:

```text
Buy & Hold: +8.26%
MA crossover: +4.01%
RSI rule: +9.44%
Random policy: -3.87%
```

Kết luận trung thực: model mới đã sửa vấn đề exposure/risk của bản cũ và có đủ kiểm định chống overfit, nhưng PPO hiện chưa thắng baseline đơn giản trên OOS window này. Đây là bằng chứng nghiên cứu tốt hơn, không phải claim SOTA.

## Lỗi Thường Gặp

`zsh: command not found: treamlit`

Bạn gõ nhầm. Dùng:

```bash
python -m streamlit run streamlit_app.py
```

`No module named streamlit`

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`No module named trading_bot`

Bạn đang chạy sai thư mục hoặc thiếu `PYTHONPATH`:

```bash
cd ReinforcementTrading_Part_1
export PYTHONPATH="$PWD"
```

`recurrent_lstm requires sb3-contrib`

```bash
python -m pip install sb3-contrib
```
