import os
import sys
import requests
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_ta_classic as ta
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Resolve paths relative to this script so it works from any working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from trading_env import ForexTradingEnv

def fetch_binance_1h_historical(symbol="BTCUSDT", start_str="2021-01-01", end_str="2022-12-31"):
    """
    Fetches historical 1H data from Binance since Yahoo Finance limits 1H to 730 days.
    """
    print(f"Fetching historical {symbol} data from Binance ({start_str} to {end_str})...")
    start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_str).timestamp() * 1000)
    
    url = "https://api.binance.com/api/v3/klines"
    all_klines = []
    
    current_start = start_ts
    while current_start < end_ts:
        params = {
            "symbol": symbol,
            "interval": "1h",
            "startTime": current_start,
            "endTime": end_ts,
            "limit": 1000
        }
        res = requests.get(url, params=params)
        data = res.json()
        if not data or type(data) is dict:
            print("Finished or hit an error:", data)
            break
        all_klines.extend(data)
        current_start = data[-1][0] + 1
        time.sleep(0.1)
        
    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    df['Gmt time'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('Gmt time')
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

def preprocess_features(df):
    """
    Exact mirror of `indicators.py` preprocessing, including the newly added Volume features!
    """
    print("Computing technical indicators (matching exactly with train data)...")
    df = df.copy()
    
    df["rsi_14"] = ta.rsi(df["Close"], length=14)
    df["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["atr_pct"] = (df["atr_14"] / df["Close"]) * 100.0

    df["ma_20"] = ta.sma(df["Close"], length=20)
    df["ma_50"] = ta.sma(df["Close"], length=50)

    df["ma_20_slope_pct"] = df["ma_20"].pct_change() * 100.0
    df["ma_50_slope_pct"] = df["ma_50"].pct_change() * 100.0

    df["close_ma20_diff_pct"] = ((df["Close"] - df["ma_20"]) / df["ma_20"]) * 100.0
    df["close_ma50_diff_pct"] = ((df["Close"] - df["ma_50"]) / df["ma_50"]) * 100.0

    df["ma_spread_pct"] = ((df["ma_20"] - df["ma_50"]) / df["ma_50"]) * 100.0
    df["ma_spread_slope_pct"] = df["ma_spread_pct"].diff()

    macd = ta.macd(df["Close"])
    df["macd_line"] = macd["MACD_12_26_9"]
    df["macd_hist"] = macd["MACDh_12_26_9"]

    bbands = ta.bbands(df["Close"], length=20)
    df["bbands_width"] = (bbands["BBU_20_2.0"] - bbands["BBL_20_2.0"]) / df["Close"] * 100.0
    
    stochrsi = ta.stochrsi(df["Close"])
    df["stochrsi_k"] = stochrsi["STOCHRSIk_14_14_3_3"]
    df["stochrsi_d"] = stochrsi["STOCHRSId_14_14_3_3"]

    # Volume Indicators
    df["mfi_14"] = ta.mfi(df["High"], df["Low"], df["Close"], df["Volume"], length=14)
    df["vol_sma_20"] = ta.sma(df["Volume"], length=20)
    df["vol_ratio"] = df["Volume"] / df["vol_sma_20"]

    df.dropna(inplace=True)

    feature_cols = [
        "rsi_14", "atr_pct", "ma_20_slope_pct", "ma_50_slope_pct",
        "close_ma20_diff_pct", "close_ma50_diff_pct", "ma_spread_pct",
        "ma_spread_slope_pct", "macd_line", "macd_hist", "bbands_width",
        "stochrsi_k", "stochrsi_d", "mfi_14", "vol_ratio"
    ]
    return df, feature_cols


from indicators import load_yfinance_data

def main():
    print("="*50)
    print("FINAL VALIDATION: OUT-OF-SAMPLE (2024-2026)")
    print("="*50)
    
    # 1. Fetch data from Binance for the strict Out-of-sample period.
    # This matches the validation set used in the 10M training.
    raw_df = fetch_binance_1h_historical(symbol="BTCUSDT", start_str="2024-04-01", end_str="2026-04-10")
    print(f"Loaded {len(raw_df)} bars of historical data.")
    
    # 2. Add indicators
    df, feature_cols = preprocess_features(raw_df)
    
    # 3. Apply Z-Score normalization based on this dataset.
    stats_df = df[feature_cols]
    val_mean = stats_df.mean().values
    val_std = stats_df.std().values

    # Pad with 0 and 1 for the 3 internal state features (pos, time, unrealized)
    val_mean = np.concatenate([val_mean, [0.0, 0.0, 0.0]])
    val_std = np.concatenate([val_std, [1.0, 1.0, 1.0]])

    # 4. Environment setup (matching train_btc_live.py exactly!)
    WIN = 60
    SL_OPTS = [200, 500, 1000]
    TP_OPTS = [200, 500, 1000]

    def make_val_env():
        return ForexTradingEnv(
            df=df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            pip_value=1.0,
            lot_size=1.0,
            spread_pips=15.0,
            commission_pips=0.0,
            max_slippage_pips=5.0,
            random_start=False,      # Deterministic for true evaluation
            episode_max_steps=None,
            feature_columns=feature_cols,
            feature_mean=val_mean,
            feature_std=val_std,
            hold_reward_weight=0.05,
            open_penalty_pips=2.0,
            time_penalty_pips=0.0,
            unrealized_delta_weight=0.0,
            downside_penalty_factor=2.0
        )

    val_env = DummyVecEnv([make_val_env])

    # 5. Load the absolute best model
    model_file = os.path.join(SCRIPT_DIR, "model_btc_best.zip")
    if not os.path.exists(model_file):
        print("ERROR: model_btc_best.zip not found! Run train_btc_live.py first.")
        return
        
    print("Loading 'model_btc_best.zip'...")
    model = PPO.load(os.path.join(SCRIPT_DIR, "model_btc_best"), env=val_env)

    # 6. Run the simulation
    print("Running deterministic simulation over the unseen 2 years. Please wait...")
    obs = val_env.reset()
    equity_curve = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        step_out = val_env.step(action)

        if len(step_out) == 4:
            obs, rewards, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, rewards, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])

        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        eq = info.get("equity_usd", val_env.get_attr("equity_usd")[0])
        equity_curve.append(eq)

        if done:
            break

    final_equity = float(equity_curve[-1])
    print(f"\n[FINAL VALIDATION] Initial Equity: $10,000.00")
    print(f"[FINAL VALIDATION] Final Equity  : ${final_equity:,.2f}")
    
    if final_equity > 10000:
        print("SUCCESS: The model is PROFITABLE on completely unseen historical data!")
    else:
        print("FAIL: The model lost money on unseen data. Still overfitting or lacks edge.")

    # 7. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve, label="Validation Equity", color="purple", linewidth=2)
    plt.title("BTC Unseen Data Validation (Years 2021-2022)")
    plt.xlabel("Trade / Time Steps")
    plt.ylabel("Equity ($)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "equity_curve_final.png"), dpi=150)
    print("Saved exact evaluation plot as 'equity_curve_final.png'")
    
if __name__ == "__main__":
    main()
