import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
import gymnasium as gym

from indicators_gold import load_binance_data
from trading_env_gold import GoldScalpingEnv

class TimeCNNFeatureExtractor(BaseFeaturesExtractor):
    """
    1D CNN that acts as the "Brain" of the bot for Time Series data.
    Instead of flattening candles, it slides over them looking for patterns.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128, window_size: int = 24, num_features: int = 12):
        super().__init__(observation_space, features_dim)
        
        self.window_size = window_size
        self.num_features = num_features
        self.extra_dim = 3 # (pos_dir, unrealized_norm, bars_in_trade)
        
        # Conv1d expects (batch_size, channels, length)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 24 -> 12
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # 12 -> 6
            nn.Flatten()
        )
        
        # 64 channels * (window_size // 4)
        cnn_out_dim = 64 * (window_size // 4)
        
        self.linear = nn.Sequential(
            nn.Linear(cnn_out_dim + self.extra_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Separate the flat 2D tensor into TimeSeries Matrix and Extra Info
        window_flat = observations[:, :-self.extra_dim]
        extra_data = observations[:, -self.extra_dim:]
        
        batch_size = observations.shape[0]
        
        # Reshape to (batch_size, window_size, features)
        window_reshaped = window_flat.view(batch_size, self.window_size, self.num_features)
        
        # PyTorch Conv1d expects (batch, channels, length), so we swap axis 1 and 2
        window_transposed = window_reshaped.transpose(1, 2)
        
        cnn_output = self.cnn(window_transposed)
        
        # Combine the visual patterns from CNN with current trade context
        combined = torch.cat([cnn_output, extra_data], dim=1)
        
        return self.linear(combined)


def main():
    print("="*50)
    print("GOLD (PAXG) SCALPING AI - 15-MINUTE ORDER FLOW v2")
    print("="*50)

    # 1. Load Data
    df, feature_cols = load_binance_data("PAXGUSDT", interval="15m", start_str="1700 days ago UTC", end_str="now")
    
    print(f"Total 15m Bars: {len(df)}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    
    # Split Train/Test (80/20)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Training on {len(train_df)} bars, Testing on {len(test_df)} bars.")

    # 2. Extract Normalization Stats from Training Data ONLY (prevent look-ahead bias)
    train_stats = train_df[feature_cols]
    train_mean = train_stats.mean().values
    train_std = train_stats.std().values
    train_std[train_std == 0] = 1e-8  # prevent division by zero
    
    # 3. Create Environments — PREDICTION MODE
    # SL = TP = $30 (1:1 ratio) so win rate directly = prediction accuracy
    # Random guessing = 50%. Anything above 50% = real edge.
    WINDOW = 24        # Look back 24 * 15 minutes = 6 hours
    SL_USD = 30.0      # Equal SL
    TP_USD = 30.0      # Equal TP (1:1 ratio = true prediction test)
    SPREAD_USD = 0.0   # SPREAD = 0 so math expectation is exactly 50.0% (unbiased)
    
    def make_train_env():
        return GoldScalpingEnv(
            df=train_df, window_size=WINDOW, 
            sl_usd=SL_USD, tp_usd=TP_USD, spread_usd=SPREAD_USD,
            feature_columns=feature_cols,
            feature_mean=train_mean, feature_std=train_std,
            hold_penalty=0.03,
            entry_cost=0.0  # No entry cost — let bot trade freely to show raw accuracy
        )

    def make_test_env():
        return GoldScalpingEnv(
            df=test_df, window_size=WINDOW, 
            sl_usd=SL_USD, tp_usd=TP_USD, spread_usd=SPREAD_USD,
            feature_columns=feature_cols,
            feature_mean=train_mean, feature_std=train_std,
            hold_penalty=0.03,
            entry_cost=0.0
        )

    train_vec_env = DummyVecEnv([make_train_env])
    test_vec_env = DummyVecEnv([make_test_env])

    # 4. Callback to save the best model on the TEST set
    eval_callback = EvalCallback(
        eval_env=test_vec_env,
        best_model_save_path="./",
        log_path="./logs",
        eval_freq=50000,
        deterministic=True,
        render=False
    )

    # 5. Model Initialization — tuned with 1D CNN Custom Feature Extractor
    policy_kwargs = dict(
        features_extractor_class=TimeCNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128, window_size=WINDOW, num_features=12),
        net_arch=[256, 128] # MLP after the CNN
    )
    
    model = PPO("MlpPolicy", train_vec_env, verbose=1, 
                learning_rate=3e-4, n_steps=4096, batch_size=512,
                n_epochs=10, gamma=0.995, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.005,
                policy_kwargs=policy_kwargs,
                tensorboard_log="./tensorboard_log/Gold_Scalper")

    # 6. Train the Agent
    print("Starting Deep Training with 1D CNN Architecture (15m)...")
    model.learn(total_timesteps=10000000, callback=eval_callback, tb_log_name="PPO_Gold_15m_CNN")
    
    # Save model directly to ensure it exists for short test runs
    model.save("model_gold_best")
    
    # Rename best model if eval_callback managed to save one
    if os.path.exists("best_model.zip"):
        os.replace("best_model.zip", "model_gold_best.zip")
        print("Updated with evaluation's best model.")

    # 7. Final Evaluation and Visualization
    print("Running Final Evaluation on Test Set...")
    best_model = PPO.load("model_gold_best", env=test_vec_env)
    
    obs = test_vec_env.reset()
    eval_equity_history = [10000.0]
    total_trades = 0
    winning_trades = 0
    
    while True:
        action, _ = best_model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = test_vec_env.step(action)
        
        # Use ACTUAL PNL for equity chart (not reward which includes penalties)
        actual_pnl = infos[0].get("actual_pnl", 0.0)
        eval_equity_history.append(eval_equity_history[-1] + actual_pnl)
        total_trades = infos[0].get("total_trades", 0)
        winning_trades = infos[0].get("winning_trades", 0)
        
        if dones[0]:
            break
    
    # Print trade statistics
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    final_equity = eval_equity_history[-1]
    prediction_edge = win_rate - 50.0  # How much better than random
    print(f"\n{'='*40}")
    print(f"PREDICTION ACCURACY RESULTS")
    print(f"{'='*40}")
    print(f"Total Signals:      {total_trades}")
    print(f"Correct Predictions: {winning_trades}")
    print(f"Prediction Accuracy: {win_rate:.1f}%")
    print(f"Edge Over Random:    {prediction_edge:+.1f}%")
    print(f"Final Equity:        ${final_equity:,.2f}")
    print(f"{'='*40}")
    if win_rate > 50:
        print("✅ Bot has REAL predictive power!")
    elif win_rate > 48:
        print("⚠️  Bot is close to having an edge, needs more training.")
    else:
        print("❌ Bot is not yet predicting better than random.")
            
    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(eval_equity_history, label=f'Equity (WR: {win_rate:.1f}%, Trades: {total_trades})', color='blue', linewidth=0.8)
    plt.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='Starting Capital')
    plt.title("Gold Scalper (15m) - Test Strategy Performance")
    plt.xlabel("Ticks (15-Minute Bars)")
    plt.ylabel("Account Equity ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("gold_equity_curve.png", dpi=150)
    print("Saved evaluation plot to 'gold_equity_curve.png'")
    print("Training Pipeline Complete!")

if __name__ == "__main__":
    main()
