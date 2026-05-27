# Multi-Asset PPO Trading Bot

Research project for training PPO reinforcement-learning trading agents on Binance market data. The current pipeline supports multiple Binance `USDT` spot symbols, saves one model artifact per asset/timeframe/run, and includes anti-overfit evaluation with baselines, walk-forward-style reports, stress tests, and charts.

> This repository is for research, backtesting, and paper-trading experiments. It is not financial advice and does not place real orders.

## Project Layout

```text
TradingBot/
├── src/trading_bot/                 # New shared multi-asset pipeline
│   ├── data.py                      # Binance API data
│   ├── features.py                  # Technical + volume/order-flow features
│   ├── env.py                       # Shared PPO trading environment
│   ├── trainer.py                   # Train/evaluate/save artifacts
│   ├── evaluation.py                # Metrics, baselines, stress tests, charts
│   ├── live.py                      # Latest signal from saved model
│   └── ui/app.py                    # Streamlit dashboard
├── artifacts/models/                # Saved model runs
│   └── <SYMBOL>/<TIMEFRAME>/<RUN_ID>/
├── reports/                         # Supplementary report documents
├── legacy/                          # Notes about the old project layout
├── ReinforcementTrading_Part_1/     # Original BTC/Gold scripts and 10M models
├── requirements.txt
└── run_ui.sh
```

The original 10M-step models are still tracked:

```text
ReinforcementTrading_Part_1/model_btc_best.zip
ReinforcementTrading_Part_1/ReinforcementTrading_Gold/model_gold_best.zip
```

## Setup From Project Root

Use Python 3.11 or 3.12 if possible. Some ML packages may not support very new Python versions.

```bash
cd TradingBot
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run The Dashboard

Fast path:

```bash
./run_ui.sh
```

`run_ui.sh` creates `.venv` if needed, installs missing dependencies from `requirements.txt`, sets `PYTHONPATH`, and starts Streamlit.

Manual path:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
python -m streamlit run src/trading_bot/ui/app.py
```

Do not run `treamlit`; the correct command is `streamlit`. Using `python -m streamlit ...` is safer because it uses the active virtual environment.

## Train From UI

The dashboard can:

- Fetch Binance `exchangeInfo` and list available `USDT` symbols.
- Show default watchlist cards: `BTCUSDT`, `ETHUSDT`, `NEARUSDT`, `SOLUSDT`, `BNBUSDT`, `XRPUSDT`, `ADAUSDT`, `DOGEUSDT`, `PAXGUSDT`.
- Highlight symbols that already have a saved artifact.
- Train a model after choosing symbol, timeframe, timesteps, reward mode, policy type, and optional Optuna trials.
- Display final equity, return %, max drawdown, Sharpe, total trades, win rate, and artifact path.
- Display generated charts: equity curve, drawdown curve, baseline comparison, and stress-test comparison.

Recommended smoke test:

```text
Symbol: BTCUSDT
Timeframe: 1h
Timesteps: 10,000 to 50,000
Lookback days: 180 to 730
Reward mode: pnl_drawdown
Policy type: mlp
Optuna trials: 0
```

Main BTC run for the report:

```text
Symbol: BTCUSDT
Timeframe: 1h
Timesteps: 2,000,000
Lookback days: 730 or more
Reward mode: pnl_drawdown
Policy type: mlp
Optuna trials: 0 to 5
```

## Train From CLI

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
python -m trading_bot.cli train --symbol BTCUSDT --timeframe 1h --timesteps 50000 --lookback-days 365
```

Train the BTC main run:

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

The latest-signal mode reads the newest saved artifact for a symbol, fetches fresh Binance candles, and prints the model recommendation. It does not place orders.

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
python -m trading_bot.cli signal --symbol BTCUSDT --timeframe 1h
```

## Artifact Outputs

Each training run writes:

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

## Anti-Overfit Checks

The upgraded pipeline records:

- Train/validation/test split by time, never shuffled.
- Validation checkpoint selection.
- Buy & Hold, MA crossover, RSI rule, and random policy baselines.
- Walk-forward-style segment metrics.
- Stress test with higher transaction cost and slippage.
- PBO-style warning report based on candidate ranking and OOS dispersion.
- Train-only normalization saved in `train_stats.npz`.

## Risk-Normalized Environment

The original prototype could accidentally create unrealistic exposure, especially on BTC where one full coin per trade is too large for a small account. The shared environment now uses risk-normalized sizing:

- Stop-loss and take-profit distances are interpreted as basis points by default.
- Each trade risks at most `1%` of current equity before slippage/spread.
- Position notional is capped by current equity.
- Equity cannot go negative, and an episode stops if equity falls below the configured safety threshold.
- Live/latest-signal mode reads the saved training config and normalization stats instead of recalculating them from a short live window.

This does not guarantee profitability, but it makes the backtest less fragile and much closer to a realistic research setup.

## Current BTC 2M Result

The current BTC report run is stored at:

```text
artifacts/models/BTCUSDT/1h/btc_2m_risk_normalized_20260527/
```

Test-only PPO metrics from this run:

```text
Final equity: 9,766.97
Return: -2.33%
Max drawdown: -14.92%
Sharpe simple: -0.014
Trades: 343
Win rate: 35.28%
```

Baseline comparison on the same test period:

```text
Buy & Hold: +8.26%
MA crossover: +4.01%
RSI rule: +9.44%
Random policy: -3.87%
```

Stress test with doubled trading cost:

```text
PPO return: -9.55%
```

Interpretation: this BTC 2M artifact is useful as an honest anti-overfit research result, not as a claim of state-of-the-art profitability. The new evaluation correctly shows that PPO is currently more stable than the broken exposure version, but it still underperforms simple baselines on this OOS window. The next research step should be HPO, recurrent/CNN candidates, feature/reward refinement, and additional seeds before claiming an edge.

## Common Errors

`zsh: command not found: treamlit`

You typed `treamlit`. Use:

```bash
python -m streamlit run src/trading_bot/ui/app.py
```

`No module named streamlit`

Activate the virtual environment and install requirements:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`No module named trading_bot`

Run from the project root and set `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/src"
```

`recurrent_lstm requires sb3-contrib`

Install optional dependency:

```bash
python -m pip install sb3-contrib
```

Binance API/rate-limit errors

Wait a few minutes, lower lookback days, or retry with fewer experiments.
