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
├── artifacts/models/<SYMBOL>/<TF>/best/ # Best model, metrics, chart theo từng tài sản/timeframe
├── artifacts/legacy_models/              # Model 10M cũ được giữ lại để backup
├── reports/TTCS_MOCK2_BoSung.docx       # DOCX bổ sung báo cáo
├── streamlit_app.py                     # Entry UI quen thuộc
├── training_runner.py                   # Wrapper tương thích cho code cũ
├── run_ui.sh                            # Chạy UI nhanh
└── requirements.txt
```

Các script cũ BTC/Gold, CSV cũ, chart cũ và folder `training_runs/` đã được dọn để tránh nhầm lẫn. Hai model 10M cũ vẫn được giữ ở `artifacts/legacy_models/` để không mất công train trước đây. Pipeline mới không cần tách BTC/Gold nữa: `BTCUSDT`, `PAXGUSDT`, `NEARUSDT`, `ETHUSDT`, `SOLUSDT`... đều chạy qua `trading_bot/`.

## Vai Trò Từng File Chính

- `streamlit_app.py`: entrypoint tương thích cũ; cách chạy chính hiện nay là `trading_bot/ui/app.py`.
- `run_ui.sh`: chọn Python 3.9-3.12, tạo `.venv`, cài dependency, chạy trực tiếp `trading_bot/ui/app.py`.
- `requirements.txt`: dependency tối thiểu cho pipeline mới; không còn cần `pandas_ta_classic`.
- `training_runner.py`: wrapper tương thích nếu code cũ import `run_training`; logic thật nằm trong `trading_bot/trainer.py`.
- `trading_bot/data.py`: gọi Binance API `exchangeInfo` và `klines`.
- `trading_bot/features.py`: tự tính RSI, ATR, MA, MACD, Bollinger width, StochRSI, MFI, volume/order-flow bằng pandas/numpy.
- `trading_bot/env.py`: môi trường PPO chung, action `HOLD/CLOSE/OPEN`, SL/TP theo basis points, risk sizing theo equity.
- `trading_bot/modeling.py`: tạo PPO `mlp`, `cnn1d`, hoặc `recurrent_lstm`.
- `trading_bot/trainer.py`: fetch API, split train/validation/test, HPO Optuna, train PPO, chọn best validation checkpoint, evaluate OOS, lưu artifact.
- `trading_bot/evaluation.py`: baseline, walk-forward report, stress test, chart equity/drawdown/baseline.
- `trading_bot/live.py`: latest signal từ artifact đã lưu, dùng đúng scaler/config lúc train.
- `trading_bot/ui/app.py`: dashboard, form train, tab latest signal, tab artifact.

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

Script này sẽ dùng `venv/` nếu folder đó đã tồn tại và còn dùng được. Nếu `venv/` cũ thiếu Streamlit hoặc bị lỗi Torch, script sẽ tự tạo `.venv/` sạch rồi cài lại dependency.
Nếu máy đang có một Streamlit khác chạy ở `8501`, Streamlit có thể tự mở port kế tiếp như `8502`; hãy nhìn dòng `Local URL` trong terminal.

Nếu máy có nhiều Python, script sẽ tránh Python `3.14` vì nhiều thư viện ML chưa ổn định trên phiên bản này. Có thể ép Python cụ thể bằng:

```bash
PYTHON_BIN=/usr/bin/python3 ./run_ui.sh
```

Cách thủ công:

```bash
cd ReinforcementTrading_Part_1
source .venv/bin/activate
export PYTHONPATH="$PWD"
python -m streamlit run trading_bot/ui/app.py
```

Lưu ý lỗi bạn gặp trước đó: lệnh đúng là `streamlit`, không phải `treamlit`.

## Train Từ UI

Dashboard hỗ trợ:

- Fetch danh sách cặp `USDT` đang trading từ Binance `exchangeInfo`.
- Card watchlist: `BTCUSDT`, `ETHUSDT`, `NEARUSDT`, `SOLUSDT`, `BNBUSDT`, `XRPUSDT`, `ADAUSDT`, `DOGEUSDT`, `PAXGUSDT`.
- Tài sản đã có model sẽ có viền sáng và hiện return/drawdown mới nhất.
- Chọn symbol, timeframe, số timesteps, model rồi bấm `Run`.
- UI đã bỏ các setting rườm rà khỏi màn hình chính. Mặc định dùng lookback `730` ngày, reward chống overfit `pnl_drawdown`, seed `42`, và tắt Optuna trên UI để tiến trình đi thẳng vào timesteps train.
- Khi bấm `Run`, UI khởi động training dưới dạng background job, hiện stage, progress bar, current step, target steps, remaining steps, artifact folder và tail log. Vì vậy trình duyệt không còn bị màn hình đen/kẹt khi train lâu.
- Dashboard không fetch Binance `exchangeInfo` lúc mở trang nữa để tránh trắng màn hình khi API/mạng chậm; dữ liệu train vẫn fetch trực tiếp từ Binance sau khi bấm `Run`.
- Sau train sẽ hiện metrics, selected strategy, và hình: equity curve, drawdown curve, baseline comparison, stress-test comparison.
- Sau khi train xong, hệ thống so sánh candidate với model hiện tại bằng score: `positive_return_bonus + return_pct + 2*sharpe - 0.25*abs(max_drawdown_pct)`. Nếu candidate tốt hơn thì promote vào `artifacts/models/<SYMBOL>/<TIMEFRAME>/best`; nếu không tốt hơn thì giữ best cũ.
- Mỗi symbol/timeframe chỉ giữ một folder `best`. Các folder run tạm được xóa sau khi promote/so sánh để tránh nhiều folder BTC 4h gây nhầm.
- Sau màn hình kết quả có nút `Back to main screen` để ẩn kết quả train và quay lại dashboard chính.

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

Run kiểm tra cấu hình CNN/Optuna bạn yêu cầu. Đây cũng là cấu hình mặc định đang được đặt sẵn trên UI để bạn mở lên là thấy ngay:

```text
Symbol: BTCUSDT
Timeframe: 4h
Timesteps: 1,000,000
Lookback days: 730
Reward mode: pnl_drawdown
Policy type: cnn1d
Optuna trials: 1
```

Mình đã chạy kiểm thử rút gọn cùng cấu hình này với `100,000` timesteps để xác nhận pipeline không lỗi. Kết quả PPO OOS hiện tại là `0.00%` vì model chọn không vào lệnh; hệ thống vì vậy đánh dấu cảnh báo anti-overfit và chọn baseline RSI `+19.13%` trong `selected_strategy.json`. Muốn kiểm tra PPO lâu hơn, giữ nguyên UI và bấm `Run` với `1,000,000` timesteps.

Progress job tạm thời được ghi vào `artifacts/jobs/` và đã được ignore khỏi Git. Model/metrics/chart thật được chuẩn hóa vào `artifacts/models/<SYMBOL>/<TIMEFRAME>/best/`.

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

Run BTC 4h CNN/Optuna:

```bash
python -m trading_bot.cli train \
  --symbol BTCUSDT \
  --timeframe 4h \
  --timesteps 1000000 \
  --lookback-days 730 \
  --reward-mode pnl_drawdown \
  --policy-type cnn1d \
  --hpo-trials 1 \
  --run-name btc_4h_1m_cnn1d_optuna1
```

## Latest Signal

Lệnh này đọc best artifact, fetch nến Binance mới và in khuyến nghị. Nó không đặt lệnh thật. Nếu không truyền timeframe, hệ thống chọn best model tốt nhất của symbol theo score OOS; nếu truyền timeframe thì chọn `artifacts/models/<SYMBOL>/<TIMEFRAME>/best`.

```bash
cd ReinforcementTrading_Part_1
source .venv/bin/activate
export PYTHONPATH="$PWD"
python -m trading_bot.cli signal --symbol BTCUSDT
```

## Artifact

Mỗi lần train tạo candidate tạm, sau đó chỉ giữ best model tại:

```text
artifacts/models/<SYMBOL>/<TIMEFRAME>/best/
├── model.zip
├── train_config.json
├── train_stats.npz
├── metrics.json
├── baseline_metrics.json
├── walk_forward_metrics.json
├── stress_test_metrics.json
├── overfit_report.json
├── selected_strategy.json
├── best_selection.json
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
- `selected_strategy.json`: nếu PPO không thắng baseline OOS, hệ thống ghi cảnh báo và chọn candidate OOS tốt nhất để tránh overfit theo cảm tính.
- UI dùng nút `Run` đơn giản; live signal luôn ưu tiên best model thay vì folder mới nhất.

## Kết Quả BTC 2M Hiện Tại

Artifact chính:

```text
artifacts/models/BTCUSDT/1h/best/
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
