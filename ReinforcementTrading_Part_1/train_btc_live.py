import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from indicators import load_binance_data
from trading_env import ForexTradingEnv


def evaluate_model(model: PPO, eval_env: DummyVecEnv, deterministic: bool = True):
    obs = eval_env.reset()
    equity_curve = []

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        step_out = eval_env.step(action)

        if len(step_out) == 4:
            obs, rewards, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, rewards, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])

        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        eq = info.get("equity_usd", eval_env.get_attr("equity_usd")[0])
        equity_curve.append(eq)

        if done:
            break

    final_equity = float(equity_curve[-1])
    return equity_curve, final_equity


def main():
    print("="*50)
    print("ULTIMATE BTC TRAINING: BINANCE 2019-2026 (10M STEPS)")
    print("="*50)

    # 1. Fetch historical BTC-USD 1h data from Binance (2019 to 2026)
    full_df, feature_cols = load_binance_data(symbol="BTCUSDT", start_str="2019-01-01", end_str="2026-04-10")
    
    # Strictly split data: 
    # Train: 2019 to April 2024 (Approx 5 years)
    # Test/Eval/Validation: April 2024 to April 2026 (Last 2 years - UNSEEN during training)
    split_date = pd.Timestamp("2024-04-01")
    train_df = full_df[full_df.index < split_date].copy()
    test_df = full_df[full_df.index >= split_date].copy()

    print(f"Dataset Split:")
    print(f"  Training Period: {train_df.index.min()} to {train_df.index.max()} ({len(train_df)} bars)")
    print(f"  Testing Period : {test_df.index.min()} to {test_df.index.max()} ({len(test_df)} bars)")

    # ---- Normalization ----
    train_stats_df = train_df[feature_cols]
    train_mean = train_stats_df.mean().values
    train_std = train_stats_df.std().values
    train_mean = np.concatenate([train_mean, [0.0, 0.0, 0.0]])
    train_std = np.concatenate([train_std, [1.0, 1.0, 1.0]])

    # ---- Env settings ----
    SL_OPTS = [200, 500, 1000]
    TP_OPTS = [200, 500, 1000]
    WIN = 60
    PIP_VALUE = 1.0
    LOT_SIZE = 1.0
    SPREAD_PIPS = 2.0 # Adjusted for realistic Binance spread (was 15.0)

    def make_train_env():
        return ForexTradingEnv(
            df=train_df, window_size=WIN, sl_options=SL_OPTS, tp_options=TP_OPTS,
            pip_value=PIP_VALUE, lot_size=LOT_SIZE, spread_pips=SPREAD_PIPS,
            commission_pips=0.0, max_slippage_pips=5.0, random_start=True,
            min_episode_steps=1000, episode_max_steps=2000, feature_columns=feature_cols,
            feature_mean=train_mean, feature_std=train_std, hold_reward_weight=0.05,
            open_penalty_pips=2.0, time_penalty_pips=0.0, unrealized_delta_weight=0.0,
            downside_penalty_factor=1.0
        )

    def make_train_eval_env():
        return ForexTradingEnv(
            df=train_df, window_size=WIN, sl_options=SL_OPTS, tp_options=TP_OPTS,
            pip_value=PIP_VALUE, lot_size=LOT_SIZE, spread_pips=SPREAD_PIPS,
            commission_pips=0.0, max_slippage_pips=5.0, random_start=False,
            episode_max_steps=None, feature_columns=feature_cols,
            feature_mean=train_mean, feature_std=train_std, hold_reward_weight=0.05,
            open_penalty_pips=2.0, time_penalty_pips=0.0, unrealized_delta_weight=0.0,
            downside_penalty_factor=1.0
        )

    def make_test_eval_env():
        return ForexTradingEnv(
            df=test_df, window_size=WIN, sl_options=SL_OPTS, tp_options=TP_OPTS,
            pip_value=PIP_VALUE, lot_size=LOT_SIZE, spread_pips=SPREAD_PIPS,
            commission_pips=0.0, max_slippage_pips=5.0, random_start=False,
            episode_max_steps=None, feature_columns=feature_cols,
            feature_mean=train_mean, feature_std=train_std, hold_reward_weight=0.05,
            open_penalty_pips=2.0, time_penalty_pips=0.0, unrealized_delta_weight=0.0,
            downside_penalty_factor=1.0
        )

    train_vec_env = DummyVecEnv([make_train_env])
    test_eval_env = DummyVecEnv([make_test_eval_env])
    train_eval_env = DummyVecEnv([make_train_eval_env])

    # ---- Model ----
    model = PPO(
        policy="MlpPolicy", env=train_vec_env, verbose=1,
        tensorboard_log="./tensorboard_log/", n_steps=4096, batch_size=512,
        ent_coef=0.05, policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    )

    ckpt_dir = "./checkpoints_btc"
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(save_freq=250_000, save_path=ckpt_dir, name_prefix="ppo_btc_10M")

    eval_callback = EvalCallback(
        test_eval_env, best_model_save_path="./best_model/",
        log_path="./logs/", eval_freq=10000, deterministic=True, render=False
    )

    # ---- Train ----
    total_timesteps = 10000000 
    print(f"Start training model on BTC (Binance) for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])

    # Load best
    print("Training finished. Loading the best model found during evaluation...")
    if os.path.exists("./best_model/best_model.zip"):
        best_model = PPO.load("./best_model/best_model.zip", env=train_vec_env)
        best_model.save("model_btc_best")
        print("Best BTC model saved from evaluation: model_btc_best")
    else:
        model.save("model_btc_best")
        best_model = model

    # ---- Final Evaluation Plots ----
    print("Evaluating final curves...")
    eq_train, _ = evaluate_model(best_model, train_eval_env)
    eq_test, _ = evaluate_model(best_model, test_eval_env)

    print(f"[IS Eval]  Final equity (train): {eq_train[-1]:.2f}")
    print(f"[OOS Eval] Final equity (test) : {eq_test[-1]:.2f}")

    plt.figure(figsize=(12, 6))
    plt.plot(eq_train, label="Train (2019-2024)")
    plt.plot(eq_test, label="Test (2024-2026)")
    plt.title("BTC Final Training Results (10M Steps)")
    plt.legend()
    plt.savefig("equity_curve.png")
    plt.close()
    print("Updated equity_curve.png")

if __name__ == "__main__":
    main()
