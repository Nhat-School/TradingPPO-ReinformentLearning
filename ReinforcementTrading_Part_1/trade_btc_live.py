from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from indicators import load_binance_data
from trading_env import ForexTradingEnv
from training_runner import find_latest_run, load_artifact


def _latest_flat_observation(env: ForexTradingEnv):
    reset_out = env.reset()
    _obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    env.current_step = len(env.df) - 1
    env.position = 0
    env.entry_price = None
    env.sl_price = None
    env.tp_price = None
    env.time_in_trade = 0
    env.prev_unrealized_pips = 0.0
    return env._get_observation()


def main():
    print("=" * 50)
    print("BTCUSDT LIVE PREDICTION - ARTIFACT-BASED PPO")
    print("=" * 50)

    run_dir = find_latest_run("BTCUSDT")
    if run_dir is None:
        print("No BTCUSDT training artifact found in training_runs/.")
        print("Run the Streamlit trainer first so live inference can reuse saved train_stats.npz.")
        return

    config, stats = load_artifact(run_dir)
    timeframe = config.get("timeframe", "1h")
    feature_cols = config["feature_columns"]
    window = int(config.get("window_size", 60))
    sl_options = config.get("sl_options", [200, 500, 1000])
    tp_options = config.get("tp_options", [200, 500, 1000])

    print(f"Using artifact: {run_dir}")
    print(f"Fetching latest BTCUSDT {timeframe} bars from Binance API...")
    df, live_feature_cols = load_binance_data(
        symbol="BTCUSDT",
        interval=timeframe,
        start_str="90 days ago UTC",
        end_str="now",
        use_cache=False,
    )
    if live_feature_cols != feature_cols:
        raise ValueError("Live feature columns do not match the saved training artifact.")

    live_df = df.tail(max(window + 120, window + 5)).copy()
    env = ForexTradingEnv(
        df=live_df,
        window_size=window,
        sl_options=sl_options,
        tp_options=tp_options,
        pip_value=1.0,
        lot_size=1.0,
        spread_pips=2.0,
        commission_pips=0.0,
        max_slippage_pips=0.0,
        random_start=False,
        episode_max_steps=None,
        feature_columns=feature_cols,
        feature_mean=stats["feature_mean"],
        feature_std=stats["feature_std"],
        hold_reward_weight=0.0,
        open_penalty_pips=0.0,
        time_penalty_pips=0.0,
        unrealized_delta_weight=0.0,
        downside_penalty_factor=1.0,
    )

    model = PPO.load(str(Path(run_dir) / "model"))
    obs = _latest_flat_observation(env)
    action, _ = model.predict(np.expand_dims(obs, axis=0), deterministic=True)
    mapped_action = env.action_map[int(action[0])]

    current_price = float(live_df.iloc[-1]["Close"])
    current_time = live_df.index[-1]

    print("\n" + "=" * 50)
    print(f"LATEST BAR TIME: {current_time}")
    print(f"CURRENT BTC PRICE: ${current_price:,.2f}")
    print("-" * 50)
    if mapped_action[0] == "OPEN":
        direction = "LONG" if mapped_action[1] == 1 else "SHORT"
        sl_val = float(mapped_action[2])
        tp_val = float(mapped_action[3])
        sl_price = current_price - sl_val if mapped_action[1] == 1 else current_price + sl_val
        tp_price = current_price + tp_val if mapped_action[1] == 1 else current_price - tp_val
        print(f"MODEL RECOMMENDATION: OPEN {direction}")
        print(f"Stop-Loss: ${sl_price:,.2f} ({sl_val:.0f} USD)")
        print(f"Take-Profit: ${tp_price:,.2f} ({tp_val:.0f} USD)")
    else:
        print(f"MODEL RECOMMENDATION: {mapped_action[0]}")
    print("=" * 50)


if __name__ == "__main__":
    main()
