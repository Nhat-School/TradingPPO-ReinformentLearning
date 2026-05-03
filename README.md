# 🏎️ Reinforcement Learning Trading Bot (PPO) — BTC & Gold

A trading agent suite trained with **Proximal Policy Optimization (PPO)** supporting two independent strategies:

| Asset | Data Source | Timeframe | Architecture |
|:---|:---|:---|:---|
| Bitcoin (BTC-USD) | Binance Spot API | 1-hour | MLP (128×128) |
| Gold (PAXG-USD) | Binance Spot API | 15-minute | 1D CNN + MLP |

---

## 🚀 Key Achievements

### Bitcoin (BTC)
- **Training**: 10,000,000 timesteps on 7-year Binance history (2019–2026).
- **Out-of-Sample**: +130% profit ($10k → $23k) on 2 completely unseen years.
- **In-Sample**: +684% profit ($10k → $78k) on training data.

### Gold (PAXG)
- **Training**: 10,000,000 timesteps on ~1,700 days of 15-minute PAXG bars.
- **Architecture**: Custom 1D CNN that slides over price patterns before the PPO policy head.
- **Metric**: Prediction accuracy (win rate > 50% = real edge over random).

---

## 🛠 Setup (Step-by-Step)

### 1. Create a Virtual Environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate

# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1
```

Your terminal prompt will show `(venv)` when active.

### 2. Install Dependencies

```bash
pip install -r ReinforcementTrading_Part_1/Requirements.txt
```

---

## 🪙 Bitcoin (BTC) — How to Use

All BTC commands are run from the project root. The scripts auto-resolve their own paths, so **you don't need to `cd` anywhere**.

### Train + Test (Full Pipeline)
```bash
python ReinforcementTrading_Part_1/train_btc_live.py
```
- Downloads 7 years of BTCUSDT 1h data from Binance (cached locally).
- Trains PPO for 10M steps with EvalCallback.
- Evaluates on both Train and Test splits.
- **Output**: `model_btc_best.zip`, `equity_curve.png`

### Test Only (Out-of-Sample Validation)
```bash
python ReinforcementTrading_Part_1/final_validate_model.py
```
- Evaluates the saved model on unseen data (Apr 2024 – Apr 2026).
- **Requires**: `model_btc_best.zip` (run training first).
- **Output**: `equity_curve_final.png`

### Live Prediction (Next-Bar Signal)
```bash
python ReinforcementTrading_Part_1/trade_btc_live.py
```
- Fetches the latest 10 days of BTC data from Binance.
- Outputs the recommended action for the next bar.
- **Requires**: `model_btc_best.zip`

**Example output:**
```
==================================================
LATEST DATA POINT: 2026-05-03 12:00:00
CURRENT PRICE: $96,400.00
--------------------------------------------------
MODEL RECOMMENDATION: 🟢 LONG
Stop-Loss: $95,400.00 (-1000.0 USD)
Take-Profit: $96,900.00 (+500.0 USD)
==================================================
```

---

## 🥇 Gold (PAXG) — How to Use

All Gold commands are also run from the project root.

### Train + Test (Full Pipeline)
```bash
python ReinforcementTrading_Part_1/ReinforcementTrading_Gold/train_gold.py
```
- Downloads ~1,700 days of PAXG-USDT 15m data from Binance.
- Trains PPO with a 1D CNN policy for 10M steps.
- **Output**: `model_gold_best.zip`, `gold_equity_curve.png`

### Live Prediction (Next-Bar Signal)
```bash
python ReinforcementTrading_Part_1/ReinforcementTrading_Gold/trade_gold_live.py
```
- Fetches the last 5 days of PAXG-USDT 15m data from Binance.
- Predicts the direction for the next 15-minute bar.
- **Requires**: `model_gold_best.zip`

**Example output:**
```
==================================================
LATEST BAR TIME: 2026-05-03 11:45:00
CURRENT GOLD PRICE: $4,615.00
--------------------------------------------------
MODEL RECOMMENDATION: 🟢 BUY / LONG
Signal: Bullish Patterns detected by CNN
TP Target: $4,645.00 | SL Exit: $4,585.00

[!] Disclaimer: Accuracy ~52.3%. Always trade with caution.
==================================================
```

---

## 📂 File Structure

```
TradingBot/
└── ReinforcementTrading_Part_1/
    │
    │  ── Bitcoin Bot ──
    ├── train_btc_live.py          # Train + Test (10M steps)
    ├── final_validate_model.py    # Out-of-sample validation
    ├── trade_btc_live.py          # Live next-bar prediction
    ├── trading_env.py             # Gym environment (SL/TP logic)
    ├── indicators.py              # Technical indicators + Binance loader
    ├── model_btc_best.zip         # Pre-trained BTC model
    ├── Requirements.txt           # Python dependencies
    │
    └── ReinforcementTrading_Gold/
        │
        │  ── Gold Bot ──
        ├── train_gold.py          # Train + Test (1D CNN, 10M steps)
        ├── trade_gold_live.py     # Live next-bar prediction
        ├── trading_env_gold.py    # Binary Gym environment
        ├── indicators_gold.py     # Order flow indicators + Binance loader
        └── model_gold_best.zip    # Pre-trained Gold model
```

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. Past performance does not guarantee future results. Always use proper risk management and never risk capital you cannot afford to lose.
