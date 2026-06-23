# ReinforcementTrading Part 1 - AI Handoff Context

Đây là README duy nhất cần đọc để tiếp quản dự án. File này được viết như một bản context cho người mới hoặc AI khác: đọc xong phải biết dự án đang làm gì, chạy ở đâu, sửa file nào, artifact nằm đâu, và những quyết định kỹ thuật quan trọng nào không nên phá.

Pipeline chính của dự án trading bot nằm trong thư mục `ReinforcementTrading_Part_1/`. Bản nâng cấp đã được gom lại để không tạo thêm một dự án riêng ở root: UI, trainer, live signal, model artifacts và báo cáo bổ sung đều nằm dưới thư mục này.

> Dự án dùng cho nghiên cứu, backtest và paper-trading. Code không đặt lệnh thật và không phải lời khuyên tài chính.

## Handoff Nhanh Cho AI Khác

- Luôn làm việc trong `ReinforcementTrading_Part_1/`, không tạo thêm project song song ở root.
- Chỉ có một README chính: `ReinforcementTrading_Part_1/README.md`. Root `README.md` ngắn đã bị xóa theo yêu cầu để tránh nhầm.
- UI chính chạy bằng `./run_ui.sh` hoặc `python -m streamlit run trading_bot/ui/app.py`.
- Dữ liệu train lấy trực tiếp từ Binance API, không dùng CSV tải sẵn làm nguồn chính.
- Multi-asset nghĩa là nhiều model riêng cho nhiều symbol/timeframe, dùng chung một pipeline PPO.
- Best model chuẩn nằm tại `artifacts/models/<SYMBOL>/<TIMEFRAME>/best/`.
- Live signal phải dùng best artifact qua `trading_bot/live.py`, không lấy đại folder mới nhất.
- Không xóa model/artifact chưa rõ nguồn gốc. Các model train lâu có thể rất quý với user.
- Các thư mục cache/job/venv không nên đưa vào Git; model/artifact quan trọng có thể track nếu dung lượng cho phép.
- Nếu cần thay đổi UI, ưu tiên giữ đơn giản: symbol, timeframe, start date, end date, timesteps, model, Run.
- Nếu cần đánh giá overfitting, dùng train/test return, drawdown, baseline, stress test, walk-forward và selected strategy đã có.

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

## Trạng Thái Hiện Tại

- Branch chính đang dùng: `main`.
- Repo remote: `https://github.com/Nhat-School/TradingPPO-ReinformentLearning.git`.
- UI đã chạy được ở `localhost:8501` khi dùng `./run_ui.sh`.
- UI đã bỏ auto-refresh để tránh đang xem metrics bị nhảy lên đầu trang. Muốn cập nhật progress thì bấm `Refresh progress`.
- UI hiện progress training gồm stage, current step, target steps, remaining steps, log tail và artifact path.
- UI có nút `Back to main screen` sau khi train xong.
- UI có `Start date` và `End date`; backend chặn ngày trước `2017-07-01`.
- Sau train, UI hiển thị riêng `Train return`, `Train max DD`, `Test return`, `Test max DD`.
- Hiện có thể tồn tại artifact chưa được track Git như `artifacts/models/BTCUSDT/1d/` hoặc `artifacts/models/ETHUSDT/`. Đừng xóa nếu chưa hỏi user.

## Vai Trò Từng File Chính

- `streamlit_app.py`: entrypoint tương thích cũ; cách chạy chính hiện nay là `trading_bot/ui/app.py`.
- `run_ui.sh`: chọn Python 3.9-3.12, tạo `.venv`, cài dependency, chạy trực tiếp `trading_bot/ui/app.py`.
- `requirements.txt`: dependency tối thiểu cho pipeline mới; không còn cần `pandas_ta_classic`.
- `training_runner.py`: wrapper tương thích nếu code cũ import `run_training`; logic thật nằm trong `trading_bot/trainer.py`.
- `trading_bot/data.py`: gọi Binance API `exchangeInfo` và `klines`.
- `trading_bot/features.py`: tự tính RSI, ATR, MA, MACD, Bollinger width, StochRSI, MFI, volume/order-flow bằng pandas/numpy.
- `trading_bot/env.py`: môi trường PPO chung, action `HOLD/CLOSE/OPEN`, SL/TP theo basis points, risk sizing theo equity.
- `trading_bot/modeling.py`: tạo PPO `mlp`, `cnn1d`, hoặc `recurrent_lstm`.
- `trading_bot/trainer.py`: fetch API, split 70% train/30% test, HPO Optuna nếu bật, train PPO, evaluate train/test, lưu artifact.
- `trading_bot/evaluation.py`: baseline, walk-forward report, stress test, chart equity/drawdown/baseline.
- `trading_bot/live.py`: latest signal từ artifact đã lưu, dùng đúng scaler/config lúc train.
- `trading_bot/ui/app.py`: dashboard, form train, tab latest signal, tab artifact.

## Luồng Hoạt Động Chính

```text
UI/CLI input
  -> trading_bot/data.py fetch Binance klines
  -> trading_bot/features.py build feature set
  -> trading_bot/trainer.py split 70/30 theo thời gian
  -> train PPO trên 70% đầu
  -> evaluate train/test
  -> compare baseline/stress/walk-forward/PBO
  -> promote candidate nếu tốt hơn best hiện tại
  -> save best artifact
  -> trading_bot/live.py đọc best artifact để tạo latest signal
```

Split mặc định:

```text
70% train
30% test/OOS để báo cáo kết quả cuối
```

Selection rule cho best model:

```text
score = positive_return_bonus + return_pct + 2 * sharpe - 0.25 * abs(max_drawdown_pct)
```

Ý nghĩa: ưu tiên model có OOS return dương, Sharpe tốt và drawdown thấp. Không chọn model chỉ vì train đẹp.

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
- Chọn symbol, timeframe, khoảng ngày train, số timesteps, model rồi bấm `Run`.
- UI đã bỏ các setting rườm rà khỏi màn hình chính. Dữ liệu lấy theo `Start date` và `End date` bạn chọn, reward chống overfit `pnl_drawdown`, seed `42`, và tắt Optuna trên UI để tiến trình đi thẳng vào timesteps train.
- Hệ thống tự split theo thời gian: `70%` đầu để train, `30%` cuối để backtest test/OOS bằng model vừa train. Sau train UI hiện riêng `Train return`, `Train max DD`, `Test return`, `Test max DD` và các chart có cả đường train/test để đánh giá overfit.
- UI chặn ngày trước `2017-07-01` vì Binance spot API không có dữ liệu phù hợp cho training trước mốc này. Nếu một coin niêm yết muộn hoặc khoảng ngày quá ngắn, backend sẽ báo lỗi rõ để bạn chọn lại khoảng ngày.
- Khi bấm `Run`, UI khởi động training dưới dạng background job, hiện stage, progress bar, current step, target steps, remaining steps, artifact folder và tail log. Vì vậy trình duyệt không còn bị màn hình đen/kẹt khi train lâu.
- Dashboard không fetch Binance `exchangeInfo` lúc mở trang nữa để tránh trắng màn hình khi API/mạng chậm; dữ liệu train vẫn fetch trực tiếp từ Binance sau khi bấm `Run`.
- Sau train sẽ hiện metrics, selected strategy, và hình: equity curve, drawdown curve, baseline comparison, stress-test comparison.
- Sau khi train xong, hệ thống so sánh candidate với model hiện tại bằng score: `positive_return_bonus + return_pct + 2*sharpe - 0.25*abs(max_drawdown_pct)`. Nếu candidate tốt hơn thì promote vào `artifacts/models/<SYMBOL>/<TIMEFRAME>/best`; nếu không tốt hơn thì giữ best cũ.
- Mỗi symbol/timeframe chỉ giữ một folder `best`. Các folder run tạm được xóa sau khi promote/so sánh để tránh nhiều folder BTC 4h gây nhầm.
- Sau màn hình kết quả có nút `Back to main screen` để ẩn kết quả train và quay lại dashboard chính.

Điều cần nhớ khi sửa UI:

- Không đưa lại quá nhiều setting rườm rà lên màn hình chính.
- Không dùng auto-refresh liên tục vì user đã phản ánh bị nhảy lên đầu trang.
- Nếu thêm tùy chọn mới, cân nhắc để CLI hỗ trợ trước, UI chỉ giữ các trường hay dùng.
- Khi bấm `Run`, phải có phản hồi rõ ràng: job đã bắt đầu, đang fetch/train/evaluate/saving, số step hiện tại.

Smoke test nên dùng:

```text
Symbol: BTCUSDT
Timeframe: 1h
Start date: 2020-01-01
End date: 2023-02-02
Timesteps: 10,000 đến 50,000
Reward mode: pnl_drawdown
Policy type: mlp
Optuna trials: 0
```

Run BTC chính cho báo cáo:

```text
Symbol: BTCUSDT
Timeframe: 1h
Start date: 2020-01-01
End date: 2023-02-02
Timesteps: 2,000,000
Reward mode: pnl_drawdown
Policy type: mlp
```

Run kiểm tra cấu hình CNN. UI hiện tại cố ý bỏ Optuna khỏi màn hình chính để đỡ rườm rà; nếu cần Optuna thì chạy bằng CLI như ví dụ bên dưới.

```text
Symbol: BTCUSDT
Timeframe: 4h
Start date: 2020-01-01
End date: 2023-02-02
Timesteps: 1,000,000
Reward mode: pnl_drawdown
Policy type: cnn1d
```

Mình đã chạy kiểm thử rút gọn để xác nhận pipeline không lỗi. Kết quả PPO ngắn chỉ dùng để smoke test, không dùng để kết luận hiệu suất cuối cùng. Muốn kiểm tra PPO lâu hơn, giữ nguyên UI và bấm `Run` với `1,000,000` timesteps.

Progress job tạm thời được ghi vào `artifacts/jobs/` và đã được ignore khỏi Git. Model/metrics/chart thật được chuẩn hóa vào `artifacts/models/<SYMBOL>/<TIMEFRAME>/best/`.

## Train Từ CLI

```bash
cd ReinforcementTrading_Part_1
source .venv/bin/activate
export PYTHONPATH="$PWD"
python -m trading_bot.cli train \
  --symbol BTCUSDT \
  --timeframe 1h \
  --timesteps 50000 \
  --start-date 2020-01-01 \
  --end-date 2023-02-02
```

Nếu muốn dùng lại kiểu cũ theo số ngày thay vì date range:

```bash
python -m trading_bot.cli train \
  --symbol BTCUSDT \
  --timeframe 1h \
  --timesteps 50000 \
  --lookback-days 730
```

Ưu tiên hiện tại vẫn là `--start-date` và `--end-date` vì user muốn kiểm soát rõ khoảng train/test.

Run BTC 2M:

```bash
python -m trading_bot.cli train \
  --symbol BTCUSDT \
  --timeframe 1h \
  --timesteps 2000000 \
  --start-date 2020-01-01 \
  --end-date 2023-02-02 \
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
  --start-date 2020-01-01 \
  --end-date 2023-02-02 \
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
├── train_metrics.json
├── metrics.json
├── test_metrics.json
├── baseline_metrics.json
├── walk_forward_metrics.json
├── stress_test_metrics.json
├── overfit_report.json
├── selected_strategy.json
├── best_selection.json
├── train_equity_curve.png
├── equity_curve.png
├── drawdown_curve.png
├── baseline_comparison.png
└── stress_test_comparison.png
```

Ý nghĩa các file quan trọng:

- `model.zip`: PPO model đã lưu.
- `train_config.json`: symbol, timeframe, date range, feature columns, split rows, env settings.
- `train_stats.npz`: mean/std fit trên train-only để tránh leakage.
- `train_metrics.json`: return/drawdown/trades trên tập train.
- `metrics.json` và `test_metrics.json`: metrics trên tập test/OOS.
- `baseline_metrics.json`: Buy & Hold, MA crossover, RSI rule, random policy.
- `stress_test_metrics.json`: metrics khi tăng transaction cost.
- `walk_forward_metrics.json`: segment report theo thời gian.
- `overfit_report.json`: cảnh báo kiểu PBO/ranking.
- `selected_strategy.json`: PPO có thắng baseline OOS không.
- `best_selection.json`: candidate có được promote thành best không.

Quy tắc dọn artifact:

- Trong mỗi `<SYMBOL>/<TIMEFRAME>/`, chỉ nên giữ folder `best`.
- Folder run tạm được xóa sau khi promote/compare.
- Không tự xóa `artifacts/legacy_models/` vì chứa model cũ train lâu.
- Không xóa artifact chưa rõ user có cần không, nhất là model BTC/ETH mới train.

## Chống Overfit

Bản nâng cấp ghi lại:

- Split 70% train/30% test theo thời gian, không shuffle.
- Backtest cả train và test bằng cùng model vừa train để so sánh overfit trực tiếp.
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

## Báo Cáo Bổ Sung

Tài liệu bổ sung nằm ở:

```text
reports/TTCS_MOCK2_BoSung.docx
```

Vai trò của DOCX:

- Bổ sung cho PDF gốc, không thay thế PDF.
- Giải thích survey trading bot/RL trading, baseline, SOTA liên quan.
- Giải thích các kỹ thuật chống overfitting đã áp dụng.
- Mô tả các nâng cấp mục 11: UI train, API-first, multi-asset, artifact best model, latest signal, baseline/stress/walk-forward.
- Không cần nhồi hướng dẫn chạy quá chi tiết vào DOCX; hướng dẫn chạy nằm ở README này.

Nếu sửa DOCX, nên giữ tinh thần: báo cáo bổ sung, không viết lại toàn bộ báo cáo cũ.

## Git Và Backup

User muốn push thẳng lên `main` khi cần backup, không thích tạo nhiều branch.

Quy tắc nên theo:

- Trước khi commit, chạy `git status --short`.
- Chỉ stage file mình sửa hoặc file user yêu cầu đưa lên Git.
- Không stage `.venv`, `__pycache__`, `artifacts/jobs`, cache hoặc file quá nặng không cần thiết.
- Với model quan trọng, có thể track nếu Git/GitHub cho phép dung lượng. Nếu quá nặng, cần Git LFS hoặc backup ngoài.
- Không dùng lệnh destructive như `git reset --hard` hoặc `git checkout --` trừ khi user yêu cầu rõ.
- Không xóa artifact/model cũ để "dọn" nếu chưa chắc chắn.

Lệnh thường dùng:

```bash
git status --short
git add <files>
git commit -m "message"
git push origin main
```

## Checklist Khi AI Khác Tiếp Tục Làm

- Đọc README này trước.
- Chạy từ `ReinforcementTrading_Part_1/`, không chạy từ root nếu lệnh cần package `trading_bot`.
- Nếu UI lỗi trắng/đen, kiểm tra terminal log và chạy `python -m streamlit run trading_bot/ui/app.py`.
- Nếu sửa train logic, test bằng `compileall`, bad date check và smoke train nhỏ.
- Nếu sửa UI, test bằng Streamlit AppTest hoặc mở `localhost:8501`.
- Nếu sửa live signal, kiểm tra `python -m trading_bot.cli signal --symbol BTCUSDT`.
- Khi train thật, đừng hứa return dương tuyệt đối; hệ thống chỉ đo và giảm overfit, không đảm bảo thắng thị trường.
- Nếu PPO không thắng baseline, ghi trung thực vào báo cáo/thảo luận thay vì che giấu.

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
