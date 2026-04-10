import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from indicators import load_yfinance_data
from trading_env import ForexTradingEnv

def main():
    print("Fetching live BTC-USD data...")
    df, feature_cols = load_yfinance_data(symbol="BTC-USD", period="120d", interval="1h")

    # Split data to Train and Test (last 20%)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Ensure Action Space Matches the EURUSD trained model exactly!
    WIN = 30
    SL_OPTS = [5, 10, 15, 25, 30, 60, 90, 120]
    TP_OPTS = [5, 10, 15, 25, 30, 60, 90, 120]

    def make_live_env():
        # Environment for predicting step-by-step
        return ForexTradingEnv(
            df=test_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            pip_value=10.0,          # 1 pip = $10 move in BTC to reuse the [5..120] options
            lot_size=0.1,            # Scales the USD value properly
            spread_pips=1.5,         # 1.5 pips = $15 spread
            commission_pips=0.0,
            max_slippage_pips=0.5,   # Up to $5 slippage
            random_start=False,
            episode_max_steps=None,
            feature_columns=feature_cols,
            hold_reward_weight=0.0,
            open_penalty_pips=0.0,
            time_penalty_pips=0.0,
            unrealized_delta_weight=0.0,
            downside_penalty_factor=2.0
        )

    live_vec_env = DummyVecEnv([make_live_env])

    print("Loading the best pre-trained EURUSD model to trade BTC...")
    model = PPO.load("model_eurusd_best", env=live_vec_env)

    # Run the live environment to the end
    obs = live_vec_env.reset()
    last_action = None
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        last_action = action[0]
        step_out = live_vec_env.step(action)
        
        if len(step_out) == 4:
            obs, rewards, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, rewards, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])
            
        if done:
            break
            
    # Remap the action to a human-readable format
    action_map = [("HOLD", None, None, None), ("CLOSE", None, None, None)]
    for direction in [0, 1]:  # 0=short, 1=long
        for sl in SL_OPTS:
            for tp in TP_OPTS:
                action_map.append(("OPEN", direction, float(sl), float(tp)))
                
    mapped_action = action_map[last_action]
    
    current_price = test_df.iloc[-1]['Close']
    current_time = test_df.index[-1]
    
    print("\n" + "="*50)
    print("LIVE TRADING BTC-USD PREDICTION (PPO RL AGENT)")
    print(f"Time (latest available): {current_time}")
    print(f"Current Price: ${current_price:.2f}")
    if mapped_action[0] == "OPEN":
        direction = "LONG" if mapped_action[1] == 1 else "SHORT"
        sl_usd = mapped_action[2] * 10.0
        tp_usd = mapped_action[3] * 10.0
        print(f"Action Recommended: {mapped_action[0]} {direction} | SL: ${sl_usd:.1f} | TP: ${tp_usd:.1f}")
    else:
        print(f"Action Recommended: {mapped_action[0]}")
    print("="*50)

if __name__ == '__main__':
    main()
