from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .config import ARTIFACT_ROOT, EnvSettings, TrainingConfig, default_env_settings
from .data import fetch_klines, to_millis
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


MIN_BINANCE_SPOT_DATE = datetime(2017, 7, 1, tzinfo=timezone.utc)


def resolve_date_range(config: TrainingConfig) -> tuple[str, str]:
    start = config.start_date or f"{config.lookback_days} days ago UTC"
    end = config.end_date or "now"
    start_ms = to_millis(start)
    end_ms = to_millis(end)
    min_ms = to_millis(MIN_BINANCE_SPOT_DATE)
    if start_ms < min_ms:
        raise ValueError(
            "Binance spot API does not have usable training data before 2017-07-01. "
            "Please choose a later start date."
        )
    if start_ms >= end_ms:
        raise ValueError("Start date must be earlier than end date.")
    return start, end


def split_train_val_test(df, train_ratio=0.70, val_ratio=0.15):
    train_end = int(len(df) * train_ratio)
    val_end = int(len(df) * (train_ratio + val_ratio))
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    if min(len(train), len(val), len(test)) < 100:
        raise ValueError(
            "Not enough rows for train/validation/test split. "
            "Choose a wider date range or a smaller timeframe."
        )
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
    if run_id == "best":
        run_id = f"candidate_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(config.artifact_root) / config.symbol.upper() / config.timeframe / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_progress(
    path: str | Path | None,
    stage: str,
    progress: float,
    message: str,
    run_dir: Path | None = None,
    current_steps: int | None = None,
    total_steps: int | None = None,
):
    if not path:
        return
    payload = {
        "stage": stage,
        "progress": max(0.0, min(1.0, float(progress))),
        "message": message,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if run_dir is not None:
        payload["run_dir"] = str(run_dir)
    if current_steps is not None:
        payload["current_steps"] = int(current_steps)
    if total_steps is not None:
        payload["total_steps"] = int(total_steps)
    existing = {}
    progress_path = Path(path)
    if progress_path.exists():
        existing = _safe_read_json(progress_path)
    if existing.get("pid"):
        payload["pid"] = existing["pid"]
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


class FileProgressCallback(BaseCallback):
    def __init__(self, progress_path: str | Path | None, total_timesteps: int, run_dir: Path):
        super().__init__()
        self.progress_path = progress_path
        self.total_timesteps = max(1, int(total_timesteps))
        self.run_dir = run_dir

    def _on_step(self) -> bool:
        if self.n_calls == 1 or self.num_timesteps % 10_000 == 0 or self.num_timesteps >= self.total_timesteps:
            train_progress = min(self.num_timesteps / self.total_timesteps, 1.0)
            shown_steps = min(self.num_timesteps, self.total_timesteps)
            write_progress(
                self.progress_path,
                "training",
                0.20 + train_progress * 0.55,
                f"Training PPO: {shown_steps:,}/{self.total_timesteps:,} timesteps",
                self.run_dir,
                current_steps=shown_steps,
                total_steps=self.total_timesteps,
            )
        return True


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


def run_hpo(config: TrainingConfig, train_df, val_df, env_params, run_dir: Path, progress_path: str | Path | None = None):
    if config.hpo_trials <= 0:
        return {}
    try:
        import optuna
    except ImportError:
        return {"warning": "Optuna is not installed; skipped HPO."}

    trial_steps = max(2_000, min(50_000, config.total_timesteps // 5))

    def objective(trial):
        write_progress(
            progress_path,
            "hpo",
            0.12,
            f"Optuna trial {trial.number + 1}/{config.hpo_trials}: searching PPO hyperparameters",
            run_dir,
        )
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


def run_training(config: TrainingConfig, progress_path: str | Path | None = None) -> dict[str, Any]:
    symbol = config.symbol.upper()
    start, end = resolve_date_range(config)
    run_dir = make_run_dir(config)
    write_progress(
        progress_path,
        "fetching",
        0.03,
        f"Fetching {symbol} {config.timeframe} candles from Binance API ({start} -> {end})",
        run_dir,
    )

    raw = fetch_klines(symbol, config.timeframe, start=start, end=end)
    if raw.empty:
        raise RuntimeError(
            f"No Binance bars returned for {symbol} {config.timeframe}. "
            "Check the selected date range and the symbol listing date."
        )
    write_progress(progress_path, "features", 0.08, "Building technical, volume and order-flow features", run_dir)
    df, feature_cols = add_features(raw)
    write_progress(progress_path, "split", 0.10, "Splitting train/validation/test by time order", run_dir)
    train_df, val_df, test_df = split_train_val_test(df)
    feature_mean, feature_std = fit_stats(train_df, feature_cols)
    settings = default_env_settings(symbol, config.timeframe)
    params = env_kwargs(settings, feature_cols, feature_mean, feature_std, config.reward_mode)

    hpo_report = run_hpo(config, train_df, val_df, params, run_dir, progress_path)
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
    progress_callback = FileProgressCallback(progress_path, config.total_timesteps, run_dir)
    model.learn(total_timesteps=config.total_timesteps, callback=CallbackList([callback, progress_callback]))

    write_progress(
        progress_path,
        "evaluating",
        0.78,
        "Loading best validation checkpoint and evaluating train/test sets",
        run_dir,
    )
    best_path = run_dir / "best_model" / "best_model.zip"
    model_cls = type(model)
    best_model = model_cls.load(str(best_path), env=test_env) if best_path.exists() else model
    best_model.save(str(run_dir / "model"))

    train_eval_env = DummyVecEnv([lambda: make_env(train_df, params, False, None)])
    train_curve, train_metrics = evaluate_model(best_model, train_eval_env, config.timeframe)
    ppo_curve, ppo_metrics = evaluate_model(best_model, test_env, config.timeframe)
    write_progress(progress_path, "baselines", 0.84, "Running Buy & Hold, MA, RSI and random baselines", run_dir)
    base = baseline_curves(test_df, config.timeframe, seed=config.seed)

    write_progress(progress_path, "stress", 0.89, "Running transaction-cost stress test", run_dir)
    stress_params = dict(params)
    stress_params["spread_pips"] = float(stress_params["spread_pips"]) * 2.0
    stress_params["max_slippage_pips"] = float(stress_params["max_slippage_pips"]) * 2.0
    stress_env = DummyVecEnv([lambda: make_env(test_df, stress_params, False, None)])
    stress_curve, stress_metrics = evaluate_model(best_model, stress_env, config.timeframe)

    candidate_metrics = {"ppo": ppo_metrics}
    candidate_metrics.update({name: item["metrics"] for name, item in base.items()})
    selected_strategy = select_strategy(candidate_metrics)

    write_progress(progress_path, "saving", 0.94, "Saving model, metrics, charts and overfit report", run_dir)
    write_json(run_dir / "metrics.json", ppo_metrics)
    write_json(run_dir / "test_metrics.json", ppo_metrics)
    write_json(run_dir / "train_metrics.json", train_metrics)
    write_json(run_dir / "baseline_metrics.json", {name: item["metrics"] for name, item in base.items()})
    write_json(run_dir / "walk_forward_metrics.json", walk_forward_report(ppo_curve, config.timeframe))
    write_json(run_dir / "stress_test_metrics.json", {"cost_x2": stress_metrics})
    write_json(run_dir / "overfit_report.json", pbo_report(candidate_metrics))
    write_json(run_dir / "selected_strategy.json", selected_strategy)

    np.savez(run_dir / "train_stats.npz", feature_mean=feature_mean, feature_std=feature_std)
    full_config = asdict(config)
    full_config.update(
        {
            "source": "Binance API",
            "requested_start": start,
            "requested_end": end,
            "actual_start": str(raw.index.min()),
            "actual_end": str(raw.index.max()),
            "split_rule": "70% train, 15% validation checkpoint selection, 15% OOS test",
            "feature_columns": feature_cols,
            "train_rows": len(train_df),
            "validation_rows": len(val_df),
            "test_rows": len(test_df),
            "env_settings": asdict(settings),
            "hpo_report": hpo_report,
        }
    )
    write_json(run_dir / "train_config.json", full_config)

    save_line_chart(run_dir / "train_equity_curve.png", f"{symbol} PPO Train Equity Curve", {"Train PPO": train_curve})
    save_line_chart(run_dir / "equity_curve.png", f"{symbol} PPO Test Equity Curve", {"Test PPO": ppo_curve})
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
    best_dir, best_report = promote_best_model(run_dir, ppo_metrics)
    write_json(best_dir / "best_selection.json", best_report)
    write_progress(progress_path, "completed", 1.0, "Training completed. Best artifact is ready.", best_dir)

    return {
        "run_dir": str(best_dir),
        "metrics": ppo_metrics,
        "baseline_metrics": {name: item["metrics"] for name, item in base.items()},
        "stress_test_metrics": {"cost_x2": stress_metrics},
        "selected_strategy": selected_strategy,
        "config": full_config,
        "best_selection": best_report,
    }


def model_score(metrics: dict[str, Any]) -> float:
    return_pct = float(metrics.get("return_pct", -1e9))
    drawdown = abs(float(metrics.get("max_drawdown_pct", 1e9)))
    sharpe = float(metrics.get("sharpe_simple", 0.0))
    # Prefer positive OOS return, then penalize fragile drawdown.
    positive_bonus = 100.0 if return_pct > 0 else 0.0
    return positive_bonus + return_pct + (2.0 * sharpe) - (0.25 * drawdown)


def read_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def promote_best_model(run_dir: Path, metrics: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    best_dir = run_dir.parent / "best"
    candidate_score = model_score(metrics)
    current_metrics = read_metrics(best_dir)
    current_score = model_score(current_metrics) if current_metrics else None
    promoted = current_score is None or candidate_score >= current_score

    report = {
        "candidate_run": str(run_dir),
        "best_dir": str(best_dir),
        "candidate_score": candidate_score,
        "previous_best_score": current_score,
        "promoted": promoted,
        "selection_rule": "score = positive_return_bonus + return_pct + 2*sharpe - 0.25*abs(max_drawdown_pct)",
    }
    if promoted:
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(run_dir, best_dir)
        report["message"] = "Candidate promoted to best model."
    else:
        report["message"] = "Candidate did not beat current best model; keeping existing best."

    if run_dir != best_dir and run_dir.exists():
        shutil.rmtree(run_dir)
    cleanup_non_best_runs(best_dir.parent)
    return best_dir, report


def cleanup_non_best_runs(timeframe_dir: Path):
    for child in timeframe_dir.iterdir():
        if child.is_dir() and child.name != "best":
            shutil.rmtree(child)


def select_strategy(candidate_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        candidate_metrics.items(),
        key=lambda item: (
            float(item[1].get("return_pct", -1e9)),
            float(item[1].get("sharpe_simple", -1e9)),
            float(item[1].get("max_drawdown_pct", -1e9)),
        ),
        reverse=True,
    )
    best_name, best_metrics = ranked[0]
    ppo_metrics = candidate_metrics.get("ppo", {})
    return {
        "selected": best_name,
        "selected_return_pct": float(best_metrics.get("return_pct", 0.0)),
        "ppo_return_pct": float(ppo_metrics.get("return_pct", 0.0)),
        "ppo_is_selected": best_name == "ppo",
        "positive_return_available": float(best_metrics.get("return_pct", 0.0)) > 0,
        "note": (
            "PPO selected by OOS return." if best_name == "ppo"
            else "PPO did not beat the best OOS baseline; use this as an anti-overfit warning."
        ),
        "ranked_candidates": [name for name, _metrics in ranked],
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


def _artifact_sort_key(run_dir: Path) -> tuple[float, float, int, float]:
    config_path = run_dir / "train_config.json"
    timesteps = 0
    score = model_score(read_metrics(run_dir))
    if config_path.exists():
        try:
            timesteps = int(json.loads(config_path.read_text(encoding="utf-8")).get("total_timesteps", 0))
        except Exception:
            timesteps = 0
    best_bonus = 1_000_000 if run_dir.name == "best" else 0
    return best_bonus, score, timesteps, run_dir.stat().st_mtime


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
