from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
os.chdir(SCRIPT_DIR)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from indicators_gold import load_binance_data
from trading_env_gold import GoldScalpingEnv
from training_runner import TimeCNNFeatureExtractor, find_latest_run, load_artifact


def _latest_observation(env: GoldScalpingEnv):
    _obs, _info = env.reset()
    env.current_step = len(env.df) - 1
    return env._get_obs()


def main():
    print("=" * 50)
    print("PAXGUSDT LIVE PREDICTION - ARTIFACT-BASED CNN PPO")
    print("=" * 50)

    run_dir = find_latest_run("PAXGUSDT", output_root=PROJECT_DIR / "training_runs")
    if run_dir is None:
        print("No PAXGUSDT training artifact found in training_runs/.")
        print("Run the Streamlit trainer first so live inference can reuse saved train_stats.npz.")
        return

    config, stats = load_artifact(run_dir)
    timeframe = config.get("timeframe", "15m")
    feature_cols = config["feature_columns"]
    window = int(config.get("window_size", 24))
    sl_usd = float(config.get("sl_usd", 30.0))
    tp_usd = float(config.get("tp_usd", 30.0))

    print(f"Using artifact: {run_dir}")
    print(f"Fetching latest PAXGUSDT {timeframe} bars from Binance API...")
    df, live_feature_cols = load_binance_data(
        "PAXGUSDT",
        interval=timeframe,
        start_str="30 days ago UTC",
        end_str="now",
    )
    if live_feature_cols != feature_cols:
        raise ValueError("Live feature columns do not match the saved training artifact.")

    live_df = df.tail(max(window + 120, window + 5)).copy()
    env = GoldScalpingEnv(
        df=live_df,
        window_size=window,
        sl_usd=sl_usd,
        tp_usd=tp_usd,
        spread_usd=0.0,
        feature_columns=feature_cols,
        feature_mean=stats["feature_mean"],
        feature_std=stats["feature_std"],
    )

    custom_objects = {
        "features_extractor_class": TimeCNNFeatureExtractor,
        "features_extractor_kwargs": dict(features_dim=128, window_size=window, num_features=len(feature_cols)),
    }
    model = PPO.load(str(Path(run_dir) / "model"), custom_objects=custom_objects)

    obs = _latest_observation(env)
    action, _states = model.predict(np.expand_dims(obs, axis=0), deterministic=True)
    mapped_action = "LONG" if int(action[0]) == 0 else "SHORT"

    current_price = float(live_df.iloc[-1]["Close"])
    current_time = live_df.index[-1]

    print("\n" + "=" * 50)
    print(f"LATEST BAR TIME: {current_time}")
    print(f"CURRENT GOLD PRICE: ${current_price:,.2f}")
    print("-" * 50)
    if mapped_action == "LONG":
        print("MODEL RECOMMENDATION: BUY / LONG")
        print(f"TP Target: ${current_price + tp_usd:,.2f} | SL Exit: ${current_price - sl_usd:,.2f}")
    else:
        print("MODEL RECOMMENDATION: SELL / SHORT")
        print(f"TP Target: ${current_price - tp_usd:,.2f} | SL Exit: ${current_price + sl_usd:,.2f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
