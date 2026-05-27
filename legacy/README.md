# Legacy Project Layout

The previous BTC and Gold scripts are intentionally kept in `ReinforcementTrading_Part_1/`.

The new production-style pipeline lives in `src/trading_bot/` and writes model artifacts to
`artifacts/models/<symbol>/<timeframe>/<run_id>/`.

Keeping the old layout in place makes it easier to compare the original implementation with
the upgraded multi-asset pipeline without risking the existing 10M-step model files.
