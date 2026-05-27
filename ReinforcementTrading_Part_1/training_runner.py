from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

SCRIPT_DIR = Path(__file__).resolve().parent
GOLD_DIR = SCRIPT_DIR / "ReinforcementTrading_Gold"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(GOLD_DIR) not in sys.path:
    sys.path.insert(0, str(GOLD_DIR))

from indicators import load_binance_data as load_btc_binance_data  # noqa: E402
from trading_env import ForexTradingEnv  # noqa: E402
from indicators_gold import load_binance_data as load_gold_binance_data  # noqa: E402
from trading_env_gold import GoldScalpingEnv  # noqa: E402


DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "training_runs"


@dataclass
class TrainingConfig:
    asset: str = "BTCUSDT"
    timeframe: str = "1h"
    total_timesteps: int = 600_000
    start_str: str = "730 days ago UTC"
    end_str: str = "now"
    use_cache: bool = False
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    run_name: str | None = None


class TimeCNNFeatureExtractor(BaseFeaturesExtractor):
    """Small 1D CNN feature extractor used by the Gold/PAXG scalping agent."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        window_size: int = 24,
        num_features: int = 12,
    ):
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
            nn.Flatten(),
        )

        cnn_out_dim = 64 * (window_size // 4)
        self.linear = nn.Sequential(
            nn.Linear(cnn_out_dim + self.extra_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        window_flat = observations[:, :-self.extra_dim]
        extra_data = observations[:, -self.extra_dim:]
        batch_size = observations.shape[0]

        window_reshaped = window_flat.view(batch_size, self.window_size, self.num_features)
        window_transposed = window_reshaped.transpose(1, 2)
        cnn_output = self.cnn(window_transposed)
        return self.linear(torch.cat([cnn_output, extra_data], dim=1))


def _json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _make_run_dir(config: TrainingConfig) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = config.run_name or f"{config.asset}_{config.timeframe}_{stamp}"
    run_dir = Path(config.output_root) / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _split_train_test(df, train_ratio: float = 0.8):
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    if len(train_df) < 100 or len(test_df) < 100:
        raise ValueError("Not enough bars after indicator preprocessing. Increase API lookback or lower timeframe.")
    return train_df, test_df


def _periods_per_year(timeframe: str) -> int:
    unit = timeframe[-1]
    value = int(timeframe[:-1] or "1")
    if unit == "m":
        return int(365 * 24 * 60 / value)
    if unit == "h":
        return int(365 * 24 / value)
    if unit == "d":
        return int(365 / value)
    return 365


def calculate_metrics(equity_curve: list[float], initial_equity: float, timeframe: str, extra: dict[str, Any] | None = None):
    equity = np.asarray(equity_curve, dtype=np.float64)
    returns = np.diff(equity) / np.maximum(equity[:-1], 1e-9)
    running_peak = np.maximum.accumulate(equity)
    drawdowns = equity / np.maximum(running_peak, 1e-9) - 1.0
    sharpe = 0.0
    if returns.size > 1 and returns.std() > 0:
        sharpe = float(np.sqrt(_periods_per_year(timeframe)) * returns.mean() / returns.std())

    metrics = {
        "initial_equity": float(initial_equity),
        "final_equity": float(equity[-1]) if equity.size else float(initial_equity),
        "return_pct": float(((equity[-1] / initial_equity) - 1.0) * 100.0) if equity.size else 0.0,
        "max_drawdown_pct": float(drawdowns.min() * 100.0) if drawdowns.size else 0.0,
        "sharpe_simple": sharpe,
        "num_points": int(equity.size),
    }
    if extra:
        metrics.update(extra)
    return metrics


def _save_equity_plot(path: Path, equity_curve: list[float], title: str) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label="Evaluation Equity", linewidth=1.2)
    plt.axhline(y=10_000, color="gray", linestyle="--", alpha=0.5, label="Initial Equity")
    plt.title(title)
    plt.xlabel("Evaluation Step")
    plt.ylabel("Equity ($)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _eval_freq(total_timesteps: int) -> int:
    return max(1_000, min(50_000, max(total_timesteps // 10, 1_000)))


def _tensorboard_log(run_dir: Path) -> str | None:
    if importlib.util.find_spec("tensorboard") is None:
        return None
    return str(run_dir / "tensorboard")


def _evaluate_btc(model: PPO, eval_env: DummyVecEnv, timeframe: str):
    obs = eval_env.reset()
    equity_curve: list[float] = []
    seen_closes = set()
    total_trades = 0
    winning_trades = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _rewards, dones, infos = eval_env.step(action)
        info = infos[0]
        equity_curve.append(float(info.get("equity_usd", 10_000.0)))

        trade_info = info.get("last_trade_info")
        if trade_info and trade_info.get("event") == "CLOSE":
            key = (trade_info.get("step"), trade_info.get("reason"), trade_info.get("exit_price"))
            if key not in seen_closes:
                seen_closes.add(key)
                total_trades += 1
                if float(trade_info.get("net_pips", 0.0)) > 0:
                    winning_trades += 1

        if bool(dones[0]):
            break

    win_rate = (winning_trades / total_trades * 100.0) if total_trades else 0.0
    metrics = calculate_metrics(
        equity_curve,
        initial_equity=10_000.0,
        timeframe=timeframe,
        extra={"total_trades": total_trades, "winning_trades": winning_trades, "win_rate_pct": win_rate},
    )
    return equity_curve, metrics


def _evaluate_gold(model: PPO, eval_env: DummyVecEnv, timeframe: str):
    obs = eval_env.reset()
    equity_curve = [10_000.0]
    total_trades = 0
    winning_trades = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _rewards, dones, infos = eval_env.step(action)
        info = infos[0]
        equity_curve.append(equity_curve[-1] + float(info.get("actual_pnl", 0.0)))
        total_trades = int(info.get("total_trades", total_trades))
        winning_trades = int(info.get("winning_trades", winning_trades))
        if bool(dones[0]):
            break

    win_rate = (winning_trades / total_trades * 100.0) if total_trades else 0.0
    metrics = calculate_metrics(
        equity_curve,
        initial_equity=10_000.0,
        timeframe=timeframe,
        extra={"total_trades": total_trades, "winning_trades": winning_trades, "win_rate_pct": win_rate},
    )
    return equity_curve, metrics


def run_btc_training(config: TrainingConfig) -> dict[str, Any]:
    run_dir = _make_run_dir(config)
    df, feature_cols = load_btc_binance_data(
        symbol="BTCUSDT",
        start_str=config.start_str,
        end_str=config.end_str,
        interval=config.timeframe,
        use_cache=config.use_cache,
        cache_dir=str(run_dir / "cache"),
    )
    train_df, test_df = _split_train_test(df)

    train_stats_df = train_df[feature_cols]
    train_mean = np.concatenate([train_stats_df.mean().values, [0.0, 0.0, 0.0]])
    train_std = np.concatenate([train_stats_df.std().replace(0, 1.0).values, [1.0, 1.0, 1.0]])

    sl_opts = [200, 500, 1000]
    tp_opts = [200, 500, 1000]
    window = 60
    env_kwargs = dict(
        window_size=window,
        sl_options=sl_opts,
        tp_options=tp_opts,
        pip_value=1.0,
        lot_size=1.0,
        spread_pips=2.0,
        commission_pips=0.0,
        max_slippage_pips=5.0,
        feature_columns=feature_cols,
        feature_mean=train_mean,
        feature_std=train_std,
        hold_reward_weight=0.05,
        open_penalty_pips=2.0,
        time_penalty_pips=0.0,
        unrealized_delta_weight=0.0,
        downside_penalty_factor=1.0,
    )

    train_vec_env = DummyVecEnv([lambda: ForexTradingEnv(df=train_df, random_start=True, min_episode_steps=1000, episode_max_steps=2000, **env_kwargs)])
    test_vec_env = DummyVecEnv([lambda: ForexTradingEnv(df=test_df, random_start=False, episode_max_steps=None, **env_kwargs)])

    model = PPO(
        policy="MlpPolicy",
        env=train_vec_env,
        verbose=1,
        tensorboard_log=_tensorboard_log(run_dir),
        n_steps=4096,
        batch_size=512,
        ent_coef=0.05,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
    )
    callback = EvalCallback(
        test_vec_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "logs"),
        eval_freq=_eval_freq(config.total_timesteps),
        deterministic=True,
        render=False,
    )
    model.learn(total_timesteps=config.total_timesteps, callback=callback)

    best_path = run_dir / "best_model" / "best_model.zip"
    best_model = PPO.load(str(best_path), env=test_vec_env) if best_path.exists() else model
    best_model.save(str(run_dir / "model"))

    equity_curve, metrics = _evaluate_btc(best_model, test_vec_env, config.timeframe)
    _save_equity_plot(run_dir / "equity_curve.png", equity_curve, f"BTCUSDT {config.timeframe} Evaluation")

    full_config = asdict(config)
    full_config.update({
        "asset": "BTCUSDT",
        "source": "Binance API",
        "feature_columns": feature_cols,
        "train_bars": len(train_df),
        "test_bars": len(test_df),
        "window_size": window,
        "sl_options": sl_opts,
        "tp_options": tp_opts,
        "model_type": "PPO MlpPolicy",
    })
    np.savez(run_dir / "train_stats.npz", feature_mean=train_mean, feature_std=train_std)
    _write_json(run_dir / "train_config.json", full_config)
    _write_json(run_dir / "metrics.json", metrics)
    return {"run_dir": str(run_dir), "metrics": metrics, "config": full_config}


def run_gold_training(config: TrainingConfig) -> dict[str, Any]:
    run_dir = _make_run_dir(config)
    df, feature_cols = load_gold_binance_data(
        symbol="PAXGUSDT",
        interval=config.timeframe,
        start_str=config.start_str,
        end_str=config.end_str,
    )
    train_df, test_df = _split_train_test(df)

    train_stats_df = train_df[feature_cols]
    train_mean = train_stats_df.mean().values
    train_std = train_stats_df.std().replace(0, 1e-8).values

    window = 24
    sl_usd = 30.0
    tp_usd = 30.0
    spread_usd = 0.0

    env_kwargs = dict(
        window_size=window,
        sl_usd=sl_usd,
        tp_usd=tp_usd,
        spread_usd=spread_usd,
        feature_columns=feature_cols,
        feature_mean=train_mean,
        feature_std=train_std,
        hold_penalty=0.03,
        entry_cost=0.0,
    )
    train_vec_env = DummyVecEnv([lambda: GoldScalpingEnv(df=train_df, **env_kwargs)])
    test_vec_env = DummyVecEnv([lambda: GoldScalpingEnv(df=test_df, **env_kwargs)])

    policy_kwargs = dict(
        features_extractor_class=TimeCNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128, window_size=window, num_features=len(feature_cols)),
        net_arch=[256, 128],
    )
    model = PPO(
        "MlpPolicy",
        train_vec_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=512,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        policy_kwargs=policy_kwargs,
        tensorboard_log=_tensorboard_log(run_dir),
    )
    callback = EvalCallback(
        test_vec_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "logs"),
        eval_freq=_eval_freq(config.total_timesteps),
        deterministic=True,
        render=False,
    )
    model.learn(total_timesteps=config.total_timesteps, callback=callback)

    best_path = run_dir / "best_model" / "best_model.zip"
    best_model = PPO.load(str(best_path), env=test_vec_env) if best_path.exists() else model
    best_model.save(str(run_dir / "model"))

    equity_curve, metrics = _evaluate_gold(best_model, test_vec_env, config.timeframe)
    _save_equity_plot(run_dir / "equity_curve.png", equity_curve, f"PAXGUSDT {config.timeframe} Evaluation")

    full_config = asdict(config)
    full_config.update({
        "asset": "PAXGUSDT",
        "source": "Binance API",
        "feature_columns": feature_cols,
        "train_bars": len(train_df),
        "test_bars": len(test_df),
        "window_size": window,
        "sl_usd": sl_usd,
        "tp_usd": tp_usd,
        "model_type": "PPO MlpPolicy + 1D CNN extractor",
    })
    np.savez(run_dir / "train_stats.npz", feature_mean=train_mean, feature_std=train_std)
    _write_json(run_dir / "train_config.json", full_config)
    _write_json(run_dir / "metrics.json", metrics)
    return {"run_dir": str(run_dir), "metrics": metrics, "config": full_config}


def run_training(config: TrainingConfig) -> dict[str, Any]:
    asset = config.asset.upper()
    if asset == "BTCUSDT":
        return run_btc_training(config)
    if asset == "PAXGUSDT":
        return run_gold_training(config)
    raise ValueError("Supported assets are BTCUSDT and PAXGUSDT.")


def find_latest_run(asset: str, output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> Path | None:
    root = Path(output_root)
    if not root.exists():
        return None
    candidates = [
        path for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith(asset.upper())
        and (path / "model.zip").exists()
        and (path / "train_stats.npz").exists()
        and (path / "train_config.json").exists()
    ]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def load_artifact(run_dir: str | Path) -> tuple[dict[str, Any], np.lib.npyio.NpzFile]:
    run_path = Path(run_dir)
    config = json.loads((run_path / "train_config.json").read_text(encoding="utf-8"))
    stats = np.load(run_path / "train_stats.npz")
    return config, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BTC/PAXG PPO bot from Binance API and save artifacts.")
    parser.add_argument("--asset", choices=["BTCUSDT", "PAXGUSDT"], default="BTCUSDT")
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--timesteps", type=int, default=600_000)
    parser.add_argument("--start", default="730 days ago UTC")
    parser.add_argument("--end", default="now")
    parser.add_argument("--use-cache", action="store_true")
    args = parser.parse_args()

    timeframe = args.timeframe or ("1h" if args.asset == "BTCUSDT" else "15m")
    result = run_training(TrainingConfig(
        asset=args.asset,
        timeframe=timeframe,
        total_timesteps=args.timesteps,
        start_str=args.start,
        end_str=args.end,
        use_cache=args.use_cache,
    ))
    print(json.dumps(result, indent=2, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
