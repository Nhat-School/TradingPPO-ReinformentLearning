import os
import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

from indicators_gold import load_binance_data
from trading_env_gold import GoldScalpingEnv

# --- MUST REDEFINE THE CNN BRAIN SO PPO CAN LOAD IT ---
class TimeCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128, window_size: int = 24, num_features: int = 12):
        super().__init__(observation_space, features_dim)
        self.window_size = window_size
        self.num_features = num_features
        self.extra_dim = 3
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )
        cnn_out_dim = 64 * (window_size // 4)
        self.linear = nn.Sequential(
            nn.Linear(cnn_out_dim + self.extra_dim, features_dim),
            nn.ReLU()
        )
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        window_flat = observations[:, :-self.extra_dim]
        extra_data = observations[:, -self.extra_dim:]
        batch_size = observations.shape[0]
        window_reshaped = window_flat.view(batch_size, self.window_size, self.num_features)
        window_transposed = window_reshaped.transpose(1, 2)
        cnn_output = self.cnn(window_transposed)
        combined = torch.cat([cnn_output, extra_data], dim=1)
        return self.linear(combined)

def get_live_data():
    # Only need recent history for indicators (last several days)
    df, feature_cols = load_binance_data("PAXGUSDT", interval="15m", start_str="5 days ago UTC", end_str="now")
    
    # Simple Z-score normalization based on recent context
    stats = df[feature_cols]
    mean = stats.mean().values
    std = stats.std().values
    return df, feature_cols, mean, std

def main():
    print("="*50)
    print("GOLD (PAXG) ORACLE LIVE PREDICTION - 15m CNN")
    print("="*50)

    # 1. Fetch latest data from Binance
    print("Fetching latest PAXG-USDT 15m data from Binance...")
    df, feature_cols, val_mean, val_std = get_live_data()

    # ---- Env settings (MUST MATCH train_gold.py) ----
    WINDOW = 24
    SL_USD = 30.0
    TP_USD = 30.0
    SPREAD_USD = 0.0 

    def make_live_env():
        return GoldScalpingEnv(
            df=df.tail(WINDOW+10), window_size=WINDOW, 
            sl_usd=SL_USD, tp_usd=TP_USD, spread_usd=SPREAD_USD,
            feature_columns=feature_cols,
            feature_mean=val_mean, feature_std=val_std
        )

    live_vec_env = DummyVecEnv([make_live_env])

    print("Loading the Oracle CNN model...")
    model_path = "model_gold_best.zip"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found! Please run train_gold.py first.")
        return
        
    custom_objects = {
        "features_extractor_class": TimeCNNFeatureExtractor,
        "features_extractor_kwargs": dict(features_dim=128, window_size=WINDOW, num_features=len(feature_cols))
    }
    
    model = PPO.load("model_gold_best", env=live_vec_env, custom_objects=custom_objects)

    # Get the very latest observation
    obs = live_vec_env.reset()
    
    # Predict the last action
    action, _states = model.predict(obs, deterministic=True)
    last_action = action[0]
    
    action_map = {0: "LONG", 1: "SHORT"}
    mapped_action = action_map[last_action]
    
    current_price = df.iloc[-1]['Close']
    current_time = df.index[-1]
    
    print("\n" + "="*50)
    print(f"LATEST BAR TIME: {current_time}")
    print(f"CURRENT GOLD PRICE: ${current_price:,.2f}")
    print("-" * 50)
    
    if mapped_action == "LONG":
        print("MODEL RECOMMENDATION: 🟢 BUY / LONG")
        print(f"Signal: Bullish Patterns detected by CNN")
        print(f"TP Target: ${current_price + TP_USD:,.2f} | SL Exit: ${current_price - SL_USD:,.2f}")
    else:
        print("MODEL RECOMMENDATION: 🔴 SELL / SHORT")
        print(f"Signal: Bearish Patterns detected by CNN")
        print(f"TP Target: ${current_price - TP_USD:,.2f} | SL Exit: ${current_price + SL_USD:,.2f}")
    
    print("\n[!] Disclaimer: Accuracy ~52.3%. Always trade with caution.")
    print("="*50)

if __name__ == '__main__':
    main()
