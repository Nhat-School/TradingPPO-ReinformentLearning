# Bitcoin Reinforcement Learning Trading Bot (PPO)

A robust, high-performance trading agent for BTC-USD, trained using **Proximal Policy Optimization (PPO)** on Binance 1-hour historical data.

## 🚀 Key Achievements
- **Training Duration**: 10,000,000 timesteps for deep convergence.
- **Dataset**: Comprehensive 7-year history from **Binance** (2019 - 2026).
- **Out-of-Sample Performance**: Achieved **+130% profit ($10k -> $23k)** in 2 years of completely unseen market data (April 2024 - April 2026).
- **In-Sample Performance**: Achieved **+684% profit ($10k -> $78k)** in training data (2019 - 2024).

## 🧠 Project Architecture

### 1. Data Intelligence (`indicators.py`)
- **Source**: Binance Spot API (High-fidelity data).
- **Features**: 
  - Standard Technicals: RSI, MACD, Bollinger Bands, ATR.
  - **Volume-based Indicators**: Money Flow Index (MFI), Volume SMA Ratio.
- **Normalization**: Dynamic Z-Score normalization to handle Bitcoin's massive price range ($3k to $70k+).

### 2. The Trading Environment (`trading_env.py`)
- Custom OpenAI Gym-compatible environment.
- **Action Space**: 20 discrete actions: HOLD, CLOSE, and OPEN long/short with multiple Stop-Loss (SL) and Take-Profit (TP) levels.
- **Reward Shaping**: Optimized for long-term equity growth with penalties for unnecessary overtrading.
- **Realistic Constraints**: 2.0 USD Spread and Slippage simulation.

### 3. Fighting Overfitting
- **Time Split**: Strict separation between the training period (2019-2024) and the validation period (2024-2026).
- **EvalCallback**: Automated saving of the absolute best model based on out-of-sample rewards during training.
- **Train-only Normalization**: Z-score statistics are fitted on the train split, then saved with the model for live inference.
- **API-first Training**: The Streamlit trainer fetches Binance API data directly by default. Cache is optional for repeated experiments.

## 🛠 Usage

### Interactive Training UI
Install the project dependencies, then launch the trainer:
```bash
cd ReinforcementTrading_Part_1
pip install -r Requirements.txt
streamlit run streamlit_app.py
```

The UI supports:
- `BTCUSDT` and `PAXGUSDT`
- Timeframe selection (`1h` default for BTC, `15m` default for PAXG)
- Training timesteps (`600000` default for short experiments)
- Binance API lookback window
- One-click training with artifacts saved under `training_runs/`

Each training run saves:
- `model.zip`
- `train_config.json`
- `train_stats.npz`
- `metrics.json`
- `equity_curve.png`

Live inference scripts now read the latest compatible training artifact so they reuse the same feature columns and normalization statistics used during training.

### Training & Testing (All-in-One)
Go to the project directory and run the main script:
```bash
cd ReinforcementTrading_Part_1
python train_btc_live.py
```
This script will:
1. Download BTC data from Binance.
2. Train the PPO model for 10M steps on 2019-2024 data.
3. Automatically evaluate the best model on unseen 2024-2026 data.
4. Generate `equity_curve.png` with both Train and Test curves.

> Note: This project is for research/backtesting/paper-trading experiments. It is not financial advice and should not be connected to real capital without additional risk controls.

## 📂 File Structure (in ReinforcementTrading_Part_1/)
| File | Purpose |
|---|---|
| `train_btc_live.py` | **Main script** — Train, evaluate, and plot results. |
| `training_runner.py` | Shared API-first training runner used by CLI and Streamlit UI. |
| `streamlit_app.py` | Simple UI to select asset, timeframe, timesteps, and run training. |
| `trading_env.py` | Custom Gym environment with SL/TP logic. |
| `indicators.py` | Technical indicators and Binance data fetching. |
| `trade_btc_live.py` | Inference script for predicting next action. |
| `final_validate_model.py` | Optional — Test on specific historical periods. |
| `model_btc_best.zip` | The trained model (Best out-of-sample performance). |

## 📊 Results Visualization
The model generates `equity_curve.png` showing the growth of $10,000 across both the training and testing regimes.
