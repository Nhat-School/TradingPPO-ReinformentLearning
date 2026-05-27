from __future__ import annotations

from dataclasses import dataclass

from trading_bot.config import ARTIFACT_ROOT, TrainingConfig as CoreTrainingConfig
from trading_bot.trainer import run_training as run_shared_training


@dataclass
class TrainingConfig:
    """Backward-compatible wrapper for the old `training_runner.py` entrypoint."""

    asset: str = "BTCUSDT"
    symbol: str | None = None
    timeframe: str = "1h"
    total_timesteps: int = 600_000
    start_str: str = "730 days ago UTC"
    end_str: str = "now"
    use_cache: bool = False
    output_root: str = str(ARTIFACT_ROOT)
    run_name: str | None = None
    reward_mode: str = "pnl_drawdown"
    policy_type: str = "mlp"
    hpo_trials: int = 0
    seed: int = 42


def _lookback_days(start_str: str) -> int:
    text = str(start_str).strip()
    if text.endswith("days ago UTC"):
        return int(text.split(" ")[0])
    return 730


def run_training(config: TrainingConfig):
    """Run the shared multi-asset PPO trainer.

    Old scripts can still import `run_training` from this file, but the actual
    implementation now lives in `trading_bot/` so BTC, gold/PAXG and other
    Binance symbols use one common code path.
    """

    symbol = (config.symbol or config.asset).upper()
    return run_shared_training(
        CoreTrainingConfig(
            symbol=symbol,
            timeframe=config.timeframe,
            total_timesteps=config.total_timesteps,
            lookback_days=_lookback_days(config.start_str),
            reward_mode=config.reward_mode,
            policy_type=config.policy_type,
            hpo_trials=config.hpo_trials,
            seed=config.seed,
            run_name=config.run_name,
            use_cache=config.use_cache,
            artifact_root=config.output_root,
        )
    )
