import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Resolve paths relative to this script so it works from any working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from indicators import load_binance_data, preprocess_technical_indicators
from trading_env import ForexTradingEnv

def main():
    print("="*50)
    print("BTC-USD LIVE PREDICTION - 10M STEPS MODEL")
    print("="*50)

    # 1. Fetch latest data from Binance (last 10 days)
    print("Fetching latest BTC-USD data from Binance...")
    # Calculate timestamps for the last 10 days
    now = pd.Timestamp.now(tz='UTC')
    start_time = now - pd.Timedelta(days=10)
    
    start_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    end_str = now.strftime('%Y-%m-%d %H:%M:%S')
    
    # load_binance_data already calls preprocess_technical_indicators and returns (df, feature_cols)
    df, feature_cols = load_binance_data(symbol="BTCUSDT", start_str=start_str, end_str=end_str)
    
    # Use the last 100 bars for the environment
    test_df = df.tail(100).copy()

    # ---- Env settings (MUST MATCH train_btc_live.py) ----
    SL_OPTS = [200, 500, 1000]
    TP_OPTS = [200, 500, 1000]
    WIN = 60
    PIP_VALUE = 1.0
    LOT_SIZE = 1.0
    SPREAD_PIPS = 2.0

    # Calculate normalization stats using this live chunk
    # (In a production system, you'd use the train_mean/train_std saved during training)
    stats_df = test_df[feature_cols]
    val_mean = stats_df.mean().values
    val_std = stats_df.std().values
    val_mean = np.concatenate([val_mean, [0.0, 0.0, 0.0]])
    val_std = np.concatenate([val_std, [1.0, 1.0, 1.0]])

    def make_live_env():
        return ForexTradingEnv(
            df=test_df, window_size=WIN, sl_options=SL_OPTS, tp_options=TP_OPTS,
            pip_value=PIP_VALUE, lot_size=LOT_SIZE, spread_pips=SPREAD_PIPS,
            commission_pips=0.0, max_slippage_pips=1.0, random_start=False,
            episode_max_steps=None, feature_columns=feature_cols,
            feature_mean=val_mean, feature_std=val_std, hold_reward_weight=0.0,
            open_penalty_pips=0.0, time_penalty_pips=0.0, unrealized_delta_weight=0.0,
            downside_penalty_factor=1.0
        )

    live_vec_env = DummyVecEnv([make_live_env])

    print("Loading the best BTC model...")
    model_path = os.path.join(SCRIPT_DIR, "model_btc_best")
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: {model_path}.zip not found! Please run train_btc_live.py first.")
        return
        
    model = PPO.load(model_path, env=live_vec_env)

    # Reset environment to get current observation
    obs = live_vec_env.reset()
    
    # Predict ONLY the very last action
    # We step through the last available bars to arrive at the current state
    last_action = 0
    for i in range(len(test_df) - WIN):
        action, _ = model.predict(obs, deterministic=True)
        last_action = action[0]
        obs, rewards, dones, infos = live_vec_env.step(action)
        if dones[0]: break
            
    # Remap the action to a human-readable format
    # Index 0: HOLD, 1: CLOSE, then others...
    action_map = [("HOLD", None, None, None), ("CLOSE", None, None, None)]
    for direction in [0, 1]:  # 0=short, 1=long
        for sl in SL_OPTS:
            for tp in TP_OPTS:
                action_map.append(("OPEN", direction, float(sl), float(tp)))
                
    mapped_action = action_map[last_action]
    
    current_price = test_df.iloc[-1]['Close']
    current_time = test_df.index[-1]
    
    print("\n" + "="*50)
    print(f"LATEST DATA POINT: {current_time}")
    print(f"CURRENT PRICE: ${current_price:,.2f}")
    print("-" * 50)
    
    if mapped_action[0] == "OPEN":
        direction = "🟢 LONG" if mapped_action[1] == 1 else "🔴 SHORT"
        sl_val = mapped_action[2]
        tp_val = mapped_action[3]
        print(f"MODEL RECOMMENDATION: {direction}")
        print(f"Stop-Loss: ${current_price - sl_val if mapped_action[1]==1 else current_price + sl_val:,.2f} (-{sl_val} USD)")
        print(f"Take-Profit: ${current_price + tp_val if mapped_action[1]==1 else current_price - tp_val:,.2f} (+{tp_val} USD)")
    elif mapped_action[0] == "CLOSE":
        print("MODEL RECOMMENDATION: ⚠️ CLOSE POSITION")
    else:
        print("MODEL RECOMMENDATION: ⏳ HOLD / WAIT")
    print("="*50)

if __name__ == '__main__':
    main()
