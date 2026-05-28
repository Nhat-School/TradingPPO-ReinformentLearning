from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "models"

WATCHLIST = [
    "BTCUSDT",
    "ETHUSDT",
    "NEARUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "PAXGUSDT",
]

TIMEFRAME_OPTIONS = ["15m", "1h", "4h", "1d"]
DEFAULT_TIMEFRAME = {
    "BTCUSDT": "1h",
    "PAXGUSDT": "15m",
}


@dataclass
class TrainingConfig:
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    total_timesteps: int = 600_000
    lookback_days: int = 730
    start_date: str | None = None
    end_date: str | None = None
    reward_mode: str = "pnl_drawdown"
    policy_type: str = "mlp"
    hpo_trials: int = 0
    seed: int = 42
    run_name: str | None = None
    use_cache: bool = False
    artifact_root: str = str(ARTIFACT_ROOT)


@dataclass
class EnvSettings:
    window_size: int = 60
    sl_options: tuple[int, ...] = (25, 50, 100)
    tp_options: tuple[int, ...] = (50, 100, 200)
    price_distance_mode: str = "bps"
    pip_value: float = 1.0
    lot_size: float = 1.0
    spread_pips: float = 2.0
    commission_pips: float = 0.0
    max_slippage_pips: float = 5.0
    initial_equity_usd: float = 10_000.0
    risk_fraction_per_trade: float = 0.01
    max_notional_fraction: float = 1.0
    min_equity_fraction: float = 0.2
    reward_scale: float = 100.0
    open_penalty_pips: float = 2.0
    hold_reward_weight: float = 0.10
    time_penalty_pips: float = 0.0
    drawdown_penalty_weight: float = 25.0
    sharpe_window: int = 64


def default_env_settings(symbol: str, timeframe: str) -> EnvSettings:
    if symbol.upper() == "PAXGUSDT":
        return EnvSettings(
            window_size=48 if timeframe.endswith("m") else 60,
            sl_options=(20, 40, 80),
            tp_options=(40, 80, 160),
            spread_pips=1.0,
            max_slippage_pips=1.0,
            open_penalty_pips=1.0,
        )
    return EnvSettings()
