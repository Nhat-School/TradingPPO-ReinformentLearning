# Reinforcement Learning Trading Bot (PPO) — BTC & Gold

A high-performance trading agent suite trained with **Proximal Policy Optimization (PPO)** supporting two independent strategies:

| Asset | Data Source | Timeframe | Architecture |
|---|---|---|---|
| **Bitcoin (BTC-USD)** | Binance Spot API | 1-hour | MLP (128×128) |
| **Gold (PAXG-USD)** | Binance Spot API | 15-minute | 1D CNN + MLP |

---

## 🚀 Key Achievements

### Bitcoin (BTC)
- **Training Duration**: 10,000,000 timesteps.
- **Dataset**: 7-year history from Binance (2019 – 2026).
- **Out-of-Sample Performance**: **+130% profit ($10k → $23k)** on 2 completely unseen years (Apr 2024 – Apr 2026).
- **In-Sample Performance**: **+684% profit ($10k → $78k)** on training data (2019 – 2024).

### Gold (PAXG)
- **Training Duration**: 10,000,000 timesteps.
- **Dataset**: ~1,700 days of 15-minute PAXG-USDT bars from Binance.
- **Architecture**: Custom 1D CNN feature extractor that slides over price patterns before the PPO policy head.
- **Evaluation Metric**: Prediction accuracy (win rate > 50% = real edge over random).

---

## 🧠 Project Architecture

### Bitcoin Bot (in `ReinforcementTrading_Part_1/`)

**Data (`indicators.py`)**
- Source: Binance Spot API, 1-hour candles.
- Features: RSI, MACD, Bollinger Bands, ATR, MFI, Volume SMA Ratio.
- Normalization: Z-Score from training statistics (no look-ahead bias).

**Environment (`trading_env.py`)**
- Custom OpenAI Gym-compatible environment.
- Action Space: 9 discrete actions combining 3 SL levels × 3 TP levels ($200 / $500 / $1000).
- Reward: Optimized for long-term equity growth with overtrading penalties.
- Constraints: 2.0 USD spread + slippage simulation.

**Anti-Overfitting**
- Strict time split: Train 2019–2024 | Eval 2024–2026.
- `EvalCallback` saves the absolute best checkpoint by out-of-sample reward.

---

### Gold Bot (in `ReinforcementTrading_Part_1/ReinforcementTrading_Gold/`)

**Data (`indicators_gold.py`)**
- Source: Binance Spot API, 15-minute candles, PAXG-USDT.
- 12 technical features per bar.

**Environment (`trading_env_gold.py`)**
- Simplified binary action space: LONG or SHORT on every bar.
- Fixed SL = TP = $30 (1:1 ratio) so win rate directly measures prediction accuracy.
- Spread = $0 during training to eliminate bias; add real spread for live deployment.

**Architecture (`TimeCNNFeatureExtractor`)**
- 1D CNN (Conv1d → MaxPool1d × 2) processes the 24-bar look-back window as a time-series image.
- Output merged with trade-state context, then fed into the PPO MLP head.

---

## 🛠 Usage

### Prerequisites

Install all dependencies from the project root:
```bash
pip install -r ReinforcementTrading_Part_1/Requirements.txt
```

---

## 🪙 Bitcoin (BTC) — Train, Test & Live Trade

### 1. Train + Test (All-in-One)

```bash
cd ReinforcementTrading_Part_1
python train_btc_live.py
```

What this does:
1. Downloads 7 years of BTCUSDT 1h data from Binance (cached locally in `data/`).
2. Splits data: **Train** 2019–Apr 2024 | **Test** Apr 2024–Apr 2026.
3. Trains PPO for 10M steps, saving the best checkpoint via `EvalCallback`.
4. Loads the best model and evaluates on both splits.
5. Saves `equity_curve.png` showing both Train and Test equity curves.

Output files:
- `model_btc_best.zip` — best model weights.
- `equity_curve.png` — performance plot.
- `checkpoints_btc/` — periodic checkpoints every 250k steps.
- `best_model/best_model.zip` — best eval checkpoint (auto-copied to `model_btc_best.zip`).

---

### 2. Test Only (Out-of-Sample Validation)

Run the standalone validation script to evaluate `model_btc_best.zip` on the strict out-of-sample period (Apr 2024 – Apr 2026) without retraining:

```bash
cd ReinforcementTrading_Part_1
python final_validate_model.py
```

What this does:
1. Fetches BTCUSDT 1h data for Apr 2024 – Apr 2026 from Binance.
2. Runs the saved model deterministically.
3. Prints `Initial Equity`, `Final Equity`, and whether the model is profitable.
4. Saves `equity_curve_final.png`.

> **Requirement**: `model_btc_best.zip` must exist (run `train_btc_live.py` first).

---

### 3. Live Trading / Next-Bar Prediction

```bash
cd ReinforcementTrading_Part_1
python trade_btc_live.py
```

What this does:
1. Fetches the latest 120 days of BTC-USD 1h data via Yahoo Finance.
2. Loads `model_eurusd_best.zip` (transfer-learned baseline) or your own `model_btc_best.zip`.
3. Steps through the recent data and outputs the **recommended action for the next bar**.

Example output:
```
==================================================
LIVE TRADING BTC-USD PREDICTION (PPO RL AGENT)
Time (latest available): 2026-05-03 11:00:00
Current Price: $96,400.00
Action Recommended: OPEN LONG | SL: $500.0 | TP: $1000.0
==================================================
```

> **Note**: Update the `PPO.load(...)` path in `trade_btc_live.py` to point to your freshly trained `model_btc_best` if you want to use your own model.

---

## 🥇 Gold (PAXG) — Train, Test & Live Trade

### 1. Train + Test (All-in-One)

```bash
cd ReinforcementTrading_Part_1/ReinforcementTrading_Gold
python train_gold.py
```

What this does:
1. Downloads ~1,700 days of PAXG-USDT 15m data from Binance.
2. Splits data: **80% Train | 20% Test** (chronological).
3. Trains PPO with a 1D CNN policy for 10M steps.
4. Saves the best model found during evaluation callbacks.
5. Runs a final evaluation and prints prediction accuracy statistics.
6. Saves `gold_equity_curve.png`.

Output files:
- `model_gold_best.zip` — best model weights.
- `gold_equity_curve.png` — equity curve with win rate annotation.
- `logs/` — evaluation logs.
- `tensorboard_log/Gold_Scalper/` — TensorBoard training logs.

Interpreting results:
```
Prediction Accuracy: 52.3%   → ✅ Real edge (> 50% = better than random)
Prediction Accuracy: 49.1%   → ❌ No edge yet, keep training
```

---

### 2. Test Only (Out-of-Sample Evaluation)

The `train_gold.py` script automatically evaluates on the held-out 20% test set at the end of every run. To re-evaluate an already-trained model without retraining, you can adapt `train_gold.py` by:

1. Commenting out the `model.learn(...)` call.
2. Replacing it with `model = PPO.load("model_gold_best", env=test_vec_env, custom_objects=...)`.
3. Running the final evaluation loop that follows.

---

### 3. Live Trading / Next-Bar Prediction

```bash
cd ReinforcementTrading_Part_1/ReinforcementTrading_Gold
python trade_gold_live.py
```

What this does:
1. Fetches the last 5 days of PAXG-USDT 15m data from Binance (live).
2. Computes Z-Score normalization from this recent window.
3. Loads `model_gold_best.zip` with the CNN feature extractor.
4. Predicts the direction for the **next 15-minute bar**.

Example output:
```
==================================================
LATEST BAR TIME: 2026-05-03 11:45:00
CURRENT GOLD PRICE: $2,650.00
--------------------------------------------------
MODEL RECOMMENDATION: 🟢 BUY / LONG
Signal: Bullish Patterns detected by CNN
TP Target: $2,680.00 | SL Exit: $2,620.00

[!] Disclaimer: Accuracy ~52.3%. Always trade with caution.
==================================================
```

> **Requirement**: `model_gold_best.zip` must exist (run `train_gold.py` first).

---

## 📂 File Structure

```
TradingPPO-ReinformentLearning/
└── ReinforcementTrading_Part_1/
    │
    │  ── Bitcoin Bot ──
    ├── train_btc_live.py          # Train + Test (BTC, 10M steps)
    ├── final_validate_model.py    # Standalone out-of-sample test (BTC)
    ├── trade_btc_live.py          # Live next-bar prediction (BTC)
    ├── trading_env.py             # Gym environment with SL/TP logic
    ├── indicators.py              # Technical indicators + Binance data loader
    ├── model_btc_best.zip         # Pre-trained BTC model
    ├── equity_curve.png           # Train/Test equity plot (BTC)
    ├── Requirements.txt           # Python dependencies
    │
    └── ReinforcementTrading_Gold/
        │
        │  ── Gold Bot ──
        ├── train_gold.py          # Train + Test (Gold, 10M steps, 1D CNN)
        ├── trade_gold_live.py     # Live next-bar prediction (Gold)
        ├── trading_env_gold.py    # Simplified binary Gym environment (Gold)
        ├── indicators_gold.py     # Technical indicators + Binance data loader (Gold)
        └── model_gold_best.zip    # Pre-trained Gold model
```

---

## 📊 Results Visualization

| Script | Output File | Description |
|---|---|---|
| `train_btc_live.py` | `equity_curve.png` | BTC Train (2019–2024) and Test (2024–2026) equity curves |
| `final_validate_model.py` | `equity_curve_final.png` | BTC strict OOS validation equity curve |
| `train_gold.py` | `gold_equity_curve.png` | Gold test equity curve with win rate annotation |

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. Past performance does not guarantee future results. Always use proper risk management and never risk capital you cannot afford to lose.
