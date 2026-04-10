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
- **Action Space**: 9 Discrete actions combining multiple Stop-Loss (SL) and Take-Profit (TP) levels.
- **Reward Shaping**: Optimized for long-term equity growth with penalties for unnecessary overtrading.
- **Realistic Constraints**: 2.0 USD Spread and Slippage simulation.

### 3. Fighting Overfitting
- **Time Split**: Strict separation between the training period (2019-2024) and the validation period (2024-2026).
- **EvalCallback**: Automated saving of the absolute best model based on out-of-sample rewards during training.

## 🛠 Usage

### Training & Testing (All-in-One)
Go to the project directory and run the main script:
```bash
cd ReinforcementTrading_Part_1
python train_btc_live.py
```
This script will:
1. Download 7 years of BTC data from Binance (cached locally).
2. Train the PPO model for 10M steps on 2019-2024 data.
3. Automatically evaluate the best model on unseen 2024-2026 data.
4. Generate `equity_curve.png` with both Train and Test curves.

## 📂 File Structure (in ReinforcementTrading_Part_1/)
| File | Purpose |
|---|---|
| `train_btc_live.py` | **Main script** — Train, evaluate, and plot results. |
| `trading_env.py` | Custom Gym environment with SL/TP logic. |
| `indicators.py` | Technical indicators and Binance data fetching. |
| `trade_btc_live.py` | Inference script for predicting next action. |
| `final_validate_model.py` | Optional — Test on specific historical periods. |
| `model_btc_best.zip` | The trained model (Best out-of-sample performance). |

## 📊 Results Visualization
The model generates `equity_curve.png` showing the growth of $10,000 across both the training and testing regimes.
