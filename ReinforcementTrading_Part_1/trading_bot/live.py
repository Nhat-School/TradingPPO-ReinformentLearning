from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from .config import EnvSettings, default_env_settings
from .data import fetch_klines
from .env import MultiAssetTradingEnv
from .features import add_features
from .modeling import TimeCNNFeatureExtractor
from .trainer import latest_artifact


def latest_signal(symbol: str, timeframe: str | None = None) -> dict:
    run_dir = latest_artifact(symbol, timeframe)
    if run_dir is None:
        raise FileNotFoundError(f"No saved model found for {symbol}. Train it first.")

    config = json.loads((run_dir / "train_config.json").read_text(encoding="utf-8"))
    stats = np.load(run_dir / "train_stats.npz")
    symbol = config["symbol"]
    timeframe = config["timeframe"]
    feature_cols = config["feature_columns"]
    settings = default_env_settings(symbol, timeframe)
    saved_settings = config.get("env_settings")
    if isinstance(saved_settings, dict):
        allowed = {field.name for field in fields(EnvSettings)}
        merged = {**settings.__dict__, **{key: value for key, value in saved_settings.items() if key in allowed}}
        settings = EnvSettings(**merged)

    raw = fetch_klines(symbol, timeframe, start="90 days ago UTC", end="now")
    df, live_features = add_features(raw)
    missing_features = [col for col in feature_cols if col not in live_features]
    if missing_features:
        raise ValueError(f"Live feature columns are missing saved model features: {missing_features}")

    live_df = df.tail(max(settings.window_size + 120, settings.window_size + 5)).copy()
    env = MultiAssetTradingEnv(
        df=live_df,
        feature_columns=feature_cols,
        window_size=settings.window_size,
        sl_options=settings.sl_options,
        tp_options=settings.tp_options,
        pip_value=settings.pip_value,
        lot_size=settings.lot_size,
        spread_pips=settings.spread_pips,
        commission_pips=settings.commission_pips,
        max_slippage_pips=0.0,
        initial_equity_usd=settings.initial_equity_usd,
        risk_fraction_per_trade=settings.risk_fraction_per_trade,
        max_notional_fraction=settings.max_notional_fraction,
        min_equity_fraction=settings.min_equity_fraction,
        reward_scale=settings.reward_scale,
        feature_mean=stats["feature_mean"],
        feature_std=stats["feature_std"],
        reward_mode=config.get("reward_mode", "pnl_drawdown"),
        random_start=False,
        episode_max_steps=None,
    )
    env.reset()
    env.current_step = len(env.df) - 1
    obs = env._get_observation()

    custom_objects = None
    if config.get("policy_type") == "cnn1d":
        custom_objects = {"features_extractor_class": TimeCNNFeatureExtractor}
    model = PPO.load(str(run_dir / "model"), custom_objects=custom_objects)
    action, _ = model.predict(np.expand_dims(obs, axis=0), deterministic=True)
    action_tuple = env.action_map[int(action[0])]

    current_price = float(live_df.iloc[-1]["Close"])
    result = {
        "symbol": symbol,
        "timeframe": timeframe,
        "run_dir": str(run_dir),
        "latest_time": str(live_df.index[-1]),
        "current_price": current_price,
        "action": action_tuple[0],
    }
    if action_tuple[0] == "OPEN":
        direction = "LONG" if action_tuple[1] == 1 else "SHORT"
        sl = float(action_tuple[2])
        tp = float(action_tuple[3])
        sl_distance = env._price_distance(sl)
        tp_distance = env._price_distance(tp)
        result.update(
            {
                "direction": direction,
                "stop_loss": current_price - sl_distance if direction == "LONG" else current_price + sl_distance,
                "take_profit": current_price + tp_distance if direction == "LONG" else current_price - tp_distance,
                "sl_distance": sl,
                "tp_distance": tp,
            }
        )
    return result
