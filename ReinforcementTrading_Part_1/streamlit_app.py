from __future__ import annotations

import json

import streamlit as st

from training_runner import TrainingConfig, run_training


st.set_page_config(page_title="Trading PPO Trainer", layout="wide")

st.title("Trading PPO Trainer")
st.caption("Train BTC/PAXG PPO agents from Binance API data and save reproducible artifacts.")

asset = st.selectbox("Asset", ["BTCUSDT", "PAXGUSDT"], index=0)

timeframe_options = {
    "BTCUSDT": ["1h", "4h", "15m"],
    "PAXGUSDT": ["15m", "1h"],
}
default_timeframe = "1h" if asset == "BTCUSDT" else "15m"
timeframe = st.selectbox(
    "Timeframe",
    timeframe_options[asset],
    index=timeframe_options[asset].index(default_timeframe),
)

col_steps, col_days = st.columns(2)
with col_steps:
    timesteps = st.number_input(
        "Training timesteps",
        min_value=1_000,
        max_value=10_000_000,
        value=600_000,
        step=10_000,
    )
with col_days:
    lookback_days = st.number_input(
        "API lookback days",
        min_value=30,
        max_value=3_000,
        value=730,
        step=30,
    )

use_cache = st.checkbox("Use API cache for repeated experiments", value=False)

if st.button("Run", type="primary"):
    config = TrainingConfig(
        asset=asset,
        timeframe=timeframe,
        total_timesteps=int(timesteps),
        start_str=f"{int(lookback_days)} days ago UTC",
        end_str="now",
        use_cache=use_cache,
    )

    with st.spinner("Fetching Binance API data, training PPO, and saving artifacts..."):
        try:
            result = run_training(config)
        except Exception as exc:
            st.error(f"Training failed: {exc}")
            st.stop()

    st.success("Training run finished.")
    st.write("Artifact folder:")
    st.code(result["run_dir"])

    st.subheader("Metrics")
    st.json(result["metrics"])

    metrics_path = f'{result["run_dir"]}/metrics.json'
    st.download_button(
        "Download metrics.json",
        data=json.dumps(result["metrics"], indent=2, ensure_ascii=False),
        file_name="metrics.json",
        mime="application/json",
    )
    st.caption(f"Saved full metrics to {metrics_path}")
