import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from indicators_gold import load_binance_5m_data
from trading_env_gold import GoldScalpingEnv

def get_live_data():
    # Only need recent history for indicators
    df, feature_cols = load_binance_5m_data("PAXGUSDT", start_str="1 days ago UTC", end_str="now")
    
    # Extract stats to normalize the live data (In reality, you'd save train_mean, train_std and load them here)
    # Since this is a test script, we normalize on recent data block
    stats = df[feature_cols]
    mean = stats.mean().values
    std = stats.std().values
    return df.iloc[-60:], feature_cols, mean, std

def main():
    print("="*50)
    print("GOLD (PAXG) SCALPING LIVE PREDICTION - 5m TIMEFRAME")
    print("="*50)

    # 1. Fetch latest data from Binance
    print("Fetching latest PAXG-USDT 5m data from Binance...")
    test_df, feature_cols, val_mean, val_std = get_live_data()

    # ---- Env settings (MUST MATCH train_gold.py) ----
    WINDOW = 60 
    SL_USD = 5.0 
    TP_USD = 10.0 
    SPREAD_USD = 0.5 

    def make_live_env():
        return GoldScalpingEnv(
            df=test_df, window_size=WINDOW, 
            sl_usd=SL_USD, tp_usd=TP_USD, spread_usd=SPREAD_USD,
            feature_columns=feature_cols,
            feature_mean=val_mean, feature_std=val_std,
            hold_penalty=0.0
        )

    live_vec_env = DummyVecEnv([make_live_env])

    print("Loading the best Gold Scalping model...")
    model_path = "model_gold_best.zip"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found! Please run train_gold.py first.")
        return
        
    model = PPO.load("model_gold_best", env=live_vec_env)

    # Reset environment to get current observation
    obs = live_vec_env.reset()
    
    # Predict ONLY the very last action
    last_action = 0
    env_inst = live_vec_env.envs[0]
    
    for i in range(len(test_df) - WINDOW):
        action, _ = model.predict(obs, deterministic=True)
        last_action = action[0]
        obs, rewards, dones, infos = live_vec_env.step(action)
        if dones[0]: break
            
    # Remap the action
    # 0 = HOLD, 1 = LONG, 2 = SHORT
    action_map = {0: "HOLD", 1: "LONG", 2: "SHORT"}
    mapped_action = action_map[last_action]
    
    current_price = test_df.iloc[-1]['Close']
    current_time = test_df.index[-1]
    
    # Order Flow Context mapping
    current_cvd = test_df.iloc[-1]['Cvd']
    current_delta = test_df.iloc[-1]['Delta']
    vwap_diff = test_df.iloc[-1]['Vwap_diff']
    
    print("\n" + "="*50)
    print(f"LATEST 1S TICK: {current_time}")
    print(f"CURRENT PRICE: ${current_price:,.2f}")
    print(f"ORDER FLOW: Delta = {current_delta:,.4f} | CVD = {current_cvd:,.4f} | VWAP Diff = {vwap_diff:,.2f}")
    print("-" * 50)
    
    if mapped_action == "LONG":
        print("MODEL RECOMMENDATION: 🟢 BUY/LONG")
        print(f"Entry: ${current_price+SPREAD_USD:,.2f} | SL: ${current_price+SPREAD_USD - SL_USD:,.2f} | TP: ${current_price+SPREAD_USD + TP_USD:,.2f}")
    elif mapped_action == "SHORT":
        print("MODEL RECOMMENDATION: 🔴 SELL/SHORT")
        print(f"Entry: ${current_price-SPREAD_USD:,.2f} | SL: ${current_price-SPREAD_USD + SL_USD:,.2f} | TP: ${current_price-SPREAD_USD - TP_USD:,.2f}")
    else:
        print("MODEL RECOMMENDATION: ⏳ HOLD (Wait for setup)")
    print("="*50)

if __name__ == '__main__':
    main()
