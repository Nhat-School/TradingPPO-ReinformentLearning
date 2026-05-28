from __future__ import annotations

import argparse
import json

from .config import TrainingConfig
from .evaluation import json_default
from .live import latest_signal
from .trainer import run_training


def main():
    parser = argparse.ArgumentParser(description="Multi-asset PPO trading bot CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--symbol", default="BTCUSDT")
    train.add_argument("--timeframe", default="1h")
    train.add_argument("--timesteps", type=int, default=600_000)
    train.add_argument("--lookback-days", type=int, default=730)
    train.add_argument("--start-date", default=None)
    train.add_argument("--end-date", default=None)
    train.add_argument("--reward-mode", default="pnl_drawdown", choices=["pnl", "pnl_drawdown", "sharpe_proxy"])
    train.add_argument("--policy-type", default="mlp", choices=["mlp", "cnn1d", "recurrent_lstm"])
    train.add_argument("--hpo-trials", type=int, default=0)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--run-name", default=None)
    train.add_argument("--progress-file", default=None)

    signal = sub.add_parser("signal")
    signal.add_argument("--symbol", default="BTCUSDT")
    signal.add_argument("--timeframe", default=None)

    args = parser.parse_args()
    if args.command == "train":
        result = run_training(
            TrainingConfig(
                symbol=args.symbol,
                timeframe=args.timeframe,
                total_timesteps=args.timesteps,
                lookback_days=args.lookback_days,
                start_date=args.start_date,
                end_date=args.end_date,
                reward_mode=args.reward_mode,
                policy_type=args.policy_type,
                hpo_trials=args.hpo_trials,
                seed=args.seed,
                run_name=args.run_name,
            ),
            progress_path=args.progress_file,
        )
    else:
        result = latest_signal(args.symbol, args.timeframe)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=json_default))


if __name__ == "__main__":
    main()
