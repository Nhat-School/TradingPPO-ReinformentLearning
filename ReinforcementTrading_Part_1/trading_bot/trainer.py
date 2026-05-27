from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .config import ARTIFACT_ROOT, EnvSettings, TrainingConfig, default_env_settings
from .data import fetch_klines
from .env import MultiAssetTradingEnv
from .evaluation import (
    baseline_curves,
    evaluate_model,
    metrics_from_equity,
    pbo_report,
    save_drawdown_chart,
    save_line_chart,
    walk_forward_report,
    write_json,
)
from .features import add_features
from .modeling import build_model


def split_train_val_test(df, train_ratio=0.70, val_ratio=0.15):
    train_end = int(len(df) * train_ratio)
    val_end = int(len(df) * (train_ratio + val_ratio))
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    if min(len(train), len(val), len(test)) < 100:
        raise ValueError("Not enough rows for train/validation/test split. Increase lookback days.")
    return train, val, test


def fit_stats(train_df, feature_cols):
    feature_mean = train_df[feature_cols].mean().values.astype(np.float32)
    feature_std = train_df[feature_cols].std().replace(0, 1.0).values.astype(np.float32)
    return (
        np.concatenate([feature_mean, np.asarray([0.0, 0.0, 0.0], dtype=np.float32)]),
        np.concatenate([feature_std, np.asarray([1.0, 1.0, 1.0], dtype=np.float32)]),
    )


def make_run_dir(config: TrainingConfig) -> Path:
    run_id = config.run_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.artifact_root) / config.symbol.upper() / config.timeframe / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def env_kwargs(settings: EnvSettings, feature_cols, feature_mean, feature_std, reward_mode: str):
    payload = asdict(settings)
    payload.update(
        {
            "feature_columns": feature_cols,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "reward_mode": reward_mode,
        }
    )
    return payload


def make_env(df, kwargs, random_start: bool, episode_max_steps: int | None):
    env = MultiAssetTradingEnv(
        df=df,
        random_start=random_start,
        episode_max_steps=episode_max_steps,
        min_episode_steps=min(1000, max(120, len(df) // 2)),
        **kwargs,
    )
    return Monitor(env)


def _eval_freq(total_timesteps: int) -> int:
    return max(1_000, min(50_000, max(total_timesteps // 10, 1_000)))


def run_hpo(config: TrainingConfig, train_df, val_df, env_params, run_dir: Path):
    if config.hpo_trials <= 0:
        return {}
    try:
        import optuna
    except ImportError:
        return {"warning": "Optuna is not installed; skipped HPO."}

    trial_steps = max(2_000, min(50_000, config.total_timesteps // 5))

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 8e-4, log=True),
            "ent_coef": trial.suggest_float("ent_coef", 0.001, 0.08, log=True),
            "n_steps": trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512]),
        }
        train_env = DummyVecEnv([lambda: make_env(train_df, env_params, True, 2000)])
        val_env = DummyVecEnv([lambda: make_env(val_df, env_params, False, None)])
        model = build_model(config.policy_type, train_env, run_dir, config.seed + trial.number, trial_steps, params)
        model.learn(total_timesteps=trial_steps)
        curve, metrics = evaluate_model(model, val_env, config.timeframe)
        return metrics["return_pct"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=config.hpo_trials)
    report = {"best_params": study.best_params, "best_value": study.best_value, "trials": config.hpo_trials}
    write_json(run_dir / "hpo_report.json", report)
    return report


def run_training(config: TrainingConfig) -> dict[str, Any]:
    symbol = config.symbol.upper()
    run_dir = make_run_dir(config)

    raw = fetch_klines(symbol, config.timeframe, start=f"{config.lookback_days} days ago UTC", end="now")
    df, feature_cols = add_features(raw)
    train_df, val_df, test_df = split_train_val_test(df)
    feature_mean, feature_std = fit_stats(train_df, feature_cols)
    settings = default_env_settings(symbol, config.timeframe)
    params = env_kwargs(settings, feature_cols, feature_mean, feature_std, config.reward_mode)

    hpo_report = run_hpo(config, train_df, val_df, params, run_dir)
    model_params = hpo_report.get("best_params", {}) if isinstance(hpo_report, dict) else {}

    train_env = DummyVecEnv([lambda: make_env(train_df, params, True, 2000)])
    val_env = DummyVecEnv([lambda: make_env(val_df, params, False, None)])
    test_env = DummyVecEnv([lambda: make_env(test_df, params, False, None)])

    model = build_model(config.policy_type, train_env, run_dir, config.seed, config.total_timesteps, model_params)
    callback = EvalCallback(
        val_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "logs"),
        eval_freq=_eval_freq(config.total_timesteps),
        deterministic=True,
        render=False,
    )
    model.learn(total_timesteps=config.total_timesteps, callback=callback)

    best_path = run_dir / "best_model" / "best_model.zip"
    model_cls = type(model)
    best_model = model_cls.load(str(best_path), env=test_env) if best_path.exists() else model
    best_model.save(str(run_dir / "model"))

    ppo_curve, ppo_metrics = evaluate_model(best_model, test_env, config.timeframe)
    base = baseline_curves(test_df, config.timeframe, seed=config.seed)

    stress_params = dict(params)
    stress_params["spread_pips"] = float(stress_params["spread_pips"]) * 2.0
    stress_params["max_slippage_pips"] = float(stress_params["max_slippage_pips"]) * 2.0
    stress_env = DummyVecEnv([lambda: make_env(test_df, stress_params, False, None)])
    stress_curve, stress_metrics = evaluate_model(best_model, stress_env, config.timeframe)

    candidate_metrics = {"ppo": ppo_metrics}
    candidate_metrics.update({name: item["metrics"] for name, item in base.items()})

    write_json(run_dir / "metrics.json", ppo_metrics)
    write_json(run_dir / "baseline_metrics.json", {name: item["metrics"] for name, item in base.items()})
    write_json(run_dir / "walk_forward_metrics.json", walk_forward_report(ppo_curve, config.timeframe))
    write_json(run_dir / "stress_test_metrics.json", {"cost_x2": stress_metrics})
    write_json(run_dir / "overfit_report.json", pbo_report(candidate_metrics))

    np.savez(run_dir / "train_stats.npz", feature_mean=feature_mean, feature_std=feature_std)
    full_config = asdict(config)
    full_config.update(
        {
            "source": "Binance API",
            "feature_columns": feature_cols,
            "train_rows": len(train_df),
            "validation_rows": len(val_df),
            "test_rows": len(test_df),
            "env_settings": asdict(settings),
            "hpo_report": hpo_report,
        }
    )
    write_json(run_dir / "train_config.json", full_config)

    save_line_chart(run_dir / "equity_curve.png", f"{symbol} PPO Equity Curve", {"PPO": ppo_curve})
    save_drawdown_chart(run_dir / "drawdown_curve.png", ppo_curve)
    save_line_chart(
        run_dir / "baseline_comparison.png",
        f"{symbol} PPO vs Baselines",
        {"PPO": ppo_curve, **{name: item["equity_curve"] for name, item in base.items()}},
    )
    save_line_chart(
        run_dir / "stress_test_comparison.png",
        f"{symbol} Stress Test",
        {"normal_cost": ppo_curve, "cost_x2": stress_curve},
    )

    return {
        "run_dir": str(run_dir),
        "metrics": ppo_metrics,
        "baseline_metrics": {name: item["metrics"] for name, item in base.items()},
        "stress_test_metrics": {"cost_x2": stress_metrics},
        "config": full_config,
    }


def list_artifacts(symbol: str | None = None, artifact_root: Path = ARTIFACT_ROOT) -> list[Path]:
    if not artifact_root.exists():
        return []
    roots = [artifact_root / symbol.upper()] if symbol else [p for p in artifact_root.iterdir() if p.is_dir()]
    runs: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        runs.extend(path for path in root.glob("*/*") if (path / "model.zip").exists())
    return sorted(runs, key=_artifact_sort_key, reverse=True)


def _artifact_sort_key(run_dir: Path) -> tuple[int, float]:
    config_path = run_dir / "train_config.json"
    timesteps = 0
    if config_path.exists():
        try:
            timesteps = int(json.loads(config_path.read_text(encoding="utf-8")).get("total_timesteps", 0))
        except Exception:
            timesteps = 0
    return timesteps, run_dir.stat().st_mtime


def latest_artifact(symbol: str, timeframe: str | None = None) -> Path | None:
    runs = list_artifacts(symbol)
    if timeframe:
        runs = [run for run in runs if run.parent.name == timeframe]
    return runs[0] if runs else None


def load_run_summary(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "train_config.json"
    return {
        "run_dir": str(run_dir),
        "metrics": json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {},
        "config": json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {},
    }
