from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def periods_per_year(timeframe: str) -> int:
    unit = timeframe[-1]
    value = int(timeframe[:-1] or "1")
    if unit == "m":
        return int(365 * 24 * 60 / value)
    if unit == "h":
        return int(365 * 24 / value)
    if unit == "d":
        return int(365 / value)
    return 365


def drawdown_curve(equity_curve: list[float]) -> np.ndarray:
    equity = np.asarray(equity_curve, dtype=float)
    peak = np.maximum.accumulate(equity)
    return equity / np.maximum(peak, 1e-9) - 1.0


def metrics_from_equity(equity_curve: list[float], timeframe: str, extra: dict[str, Any] | None = None):
    equity = np.asarray(equity_curve, dtype=float)
    returns = np.diff(equity) / np.maximum(equity[:-1], 1e-9)
    sharpe = 0.0
    if returns.size > 2 and returns.std() > 0:
        sharpe = np.sqrt(periods_per_year(timeframe)) * returns.mean() / returns.std()
    metrics = {
        "initial_equity": float(equity[0]) if equity.size else 10_000.0,
        "final_equity": float(equity[-1]) if equity.size else 10_000.0,
        "return_pct": float(((equity[-1] / equity[0]) - 1.0) * 100.0) if equity.size else 0.0,
        "max_drawdown_pct": float(drawdown_curve(equity).min() * 100.0) if equity.size else 0.0,
        "sharpe_simple": float(sharpe),
        "num_points": int(equity.size),
    }
    if extra:
        metrics.update(extra)
    return metrics


def evaluate_model(model, env, timeframe: str):
    obs = env.reset()
    equity_curve = [10_000.0]
    total_trades = 0
    winning_trades = 0
    seen_closes = set()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _rewards, dones, infos = env.step(action)
        info = infos[0]
        equity_curve.append(float(info.get("equity_usd", equity_curve[-1])))
        trade = info.get("last_trade_info")
        if trade and trade.get("event") == "CLOSE":
            key = (trade.get("step"), trade.get("reason"), trade.get("exit_price"))
            if key not in seen_closes:
                seen_closes.add(key)
                total_trades += 1
                winning_trades += int(float(trade.get("net_pips", 0.0)) > 0)
        if bool(dones[0]):
            break

    win_rate = 100.0 * winning_trades / total_trades if total_trades else 0.0
    return equity_curve, metrics_from_equity(
        equity_curve,
        timeframe,
        {"total_trades": total_trades, "winning_trades": winning_trades, "win_rate_pct": win_rate},
    )


def baseline_curves(df: pd.DataFrame, timeframe: str, seed: int = 42) -> dict[str, dict[str, Any]]:
    close = df["Close"].astype(float).reset_index(drop=True)
    pct = close.pct_change().fillna(0.0).to_numpy()

    buy_hold = (10_000.0 * close / close.iloc[0]).to_list()

    ma_fast = close.rolling(20).mean()
    ma_slow = close.rolling(50).mean()
    ma_position = (ma_fast > ma_slow).astype(float).shift(1).fillna(0.0).to_numpy()
    ma_curve = (10_000.0 * np.cumprod(1.0 + ma_position * pct)).tolist()

    rsi = df.get("rsi_14", pd.Series(50.0, index=df.index)).reset_index(drop=True)
    rsi_position = np.where(rsi < 35, 1.0, np.where(rsi > 70, 0.0, np.nan))
    rsi_position = pd.Series(rsi_position).ffill().fillna(0.0).shift(1).fillna(0.0).to_numpy()
    rsi_curve = (10_000.0 * np.cumprod(1.0 + rsi_position * pct)).tolist()

    rng = np.random.default_rng(seed)
    random_position = rng.choice([-1.0, 0.0, 1.0], size=len(close), p=[0.15, 0.70, 0.15])
    random_curve = (10_000.0 * np.cumprod(1.0 + np.roll(random_position, 1) * pct)).tolist()

    curves = {
        "buy_hold": buy_hold,
        "ma_crossover": ma_curve,
        "rsi_rule": rsi_curve,
        "random_policy": random_curve,
    }
    return {name: {"equity_curve": curve, "metrics": metrics_from_equity(curve, timeframe)} for name, curve in curves.items()}


def walk_forward_report(equity_curve: list[float], timeframe: str, windows: int = 4):
    chunks = np.array_split(np.asarray(equity_curve, dtype=float), windows)
    report = {}
    for idx, chunk in enumerate(chunks, start=1):
        if len(chunk) < 2:
            continue
        report[f"window_{idx}"] = metrics_from_equity(chunk.tolist(), timeframe)
    return report


def pbo_report(candidate_metrics: dict[str, dict[str, Any]]):
    ranked = sorted(candidate_metrics.items(), key=lambda item: item[1].get("return_pct", -9999), reverse=True)
    best_name, best_metrics = ranked[0]
    spread = ranked[0][1].get("return_pct", 0.0) - ranked[-1][1].get("return_pct", 0.0)
    risk = "low"
    if best_metrics.get("max_drawdown_pct", 0.0) < -30 or spread > 150:
        risk = "medium"
    if best_name != "ppo" and best_metrics.get("return_pct", 0.0) > candidate_metrics.get("ppo", {}).get("return_pct", 0.0):
        risk = "high"
    return {
        "best_candidate": best_name,
        "ranked_candidates": [name for name, _ in ranked],
        "return_spread_pct": float(spread),
        "estimated_overfit_risk": risk,
        "note": "Heuristic PBO-style warning based on candidate ranking and OOS dispersion, not a formal CSCV test.",
    }


def save_line_chart(path: Path, title: str, series: dict[str, list[float]], ylabel: str = "Equity ($)") -> None:
    plt.figure(figsize=(12, 6))
    for name, values in series.items():
        plt.plot(values, label=name, linewidth=1.2)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_drawdown_chart(path: Path, equity_curve: list[float] | dict[str, list[float]]) -> None:
    if isinstance(equity_curve, dict):
        series = {
            name: (drawdown_curve(values) * 100.0).tolist()
            for name, values in equity_curve.items()
        }
    else:
        series = {"Drawdown %": (drawdown_curve(equity_curve) * 100.0).tolist()}
    save_line_chart(path, "Drawdown Curve", series, "Drawdown (%)")
