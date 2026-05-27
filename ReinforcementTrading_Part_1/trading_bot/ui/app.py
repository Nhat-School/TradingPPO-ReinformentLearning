from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trading_bot.config import DEFAULT_TIMEFRAME, WATCHLIST, TrainingConfig
from trading_bot.data import get_exchange_symbols
from trading_bot.live import latest_signal
from trading_bot.trainer import latest_artifact, load_run_summary, run_training


st.set_page_config(page_title="Multi-Asset PPO Trading Bot", layout="wide")

if "last_run_dir" not in st.session_state:
    st.session_state.last_run_dir = None
if "last_run_symbol" not in st.session_state:
    st.session_state.last_run_symbol = None

st.markdown(
    """
    <style>
    .asset-card {
        border: 1px solid #2b3138;
        border-radius: 8px;
        padding: 14px 16px;
        background: #111820;
        min-height: 132px;
    }
    .asset-card.ready {
        border: 2px solid #35d07f;
        box-shadow: 0 0 18px rgba(53,208,127,0.25);
        background: linear-gradient(145deg, #102019, #111820);
    }
    .asset-card.dim {
        opacity: 0.62;
    }
    .asset-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .small-muted {
        color: #9ca3af;
        font-size: 0.82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600)
def cached_symbols():
    try:
        return get_exchange_symbols("USDT")
    except Exception:
        return WATCHLIST


def render_asset_card(symbol: str):
    run_dir = latest_artifact(symbol)
    ready = run_dir is not None
    metrics = {}
    timeframe = DEFAULT_TIMEFRAME.get(symbol, "1h")
    if ready:
        summary = load_run_summary(run_dir)
        metrics = summary.get("metrics", {})
        timeframe = summary.get("config", {}).get("timeframe", timeframe)
    css = "asset-card ready" if ready else "asset-card dim"
    status = "MODEL SAVED" if ready else "NO MODEL"
    ret = metrics.get("return_pct")
    dd = metrics.get("max_drawdown_pct")
    detail = f"Return: {ret:.2f}% | DD: {dd:.2f}%" if ret is not None and dd is not None else "Train to create artifact"
    st.markdown(
        f"""
        <div class="{css}">
          <div class="asset-title">{symbol}</div>
          <div class="small-muted">{status} · {timeframe}</div>
          <div style="margin-top:10px;">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_artifact(run_dir: str | Path):
    run_path = Path(run_dir)
    st.write("Artifact folder:")
    st.code(str(run_path))

    metrics_path = run_path / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        cols = st.columns(5)
        cols[0].metric("Final equity", f"${metrics.get('final_equity', 0):,.2f}")
        cols[1].metric("Return", f"{metrics.get('return_pct', 0):.2f}%")
        cols[2].metric("Max drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%")
        cols[3].metric("Sharpe", f"{metrics.get('sharpe_simple', 0):.2f}")
        cols[4].metric("Win rate", f"{metrics.get('win_rate_pct', 0):.1f}%")
        st.json(metrics)

    chart_files = [
        ("Equity curve", "equity_curve.png"),
        ("Drawdown curve", "drawdown_curve.png"),
        ("Baseline comparison", "baseline_comparison.png"),
        ("Stress test", "stress_test_comparison.png"),
    ]
    tabs = st.tabs([label for label, _ in chart_files])
    for tab, (_label, filename) in zip(tabs, chart_files):
        path = run_path / filename
        with tab:
            if path.exists():
                st.image(str(path), width="stretch")
            else:
                st.info(f"{filename} is not available for this run.")

    for label, filename in [
        ("Selected strategy", "selected_strategy.json"),
        ("Baseline metrics", "baseline_metrics.json"),
        ("Walk-forward metrics", "walk_forward_metrics.json"),
        ("Stress-test metrics", "stress_test_metrics.json"),
        ("Overfit report", "overfit_report.json"),
    ]:
        path = run_path / filename
        if path.exists():
            with st.expander(label):
                st.json(json.loads(path.read_text(encoding="utf-8")))


st.title("Multi-Asset PPO Trading Bot")
st.caption("API-first Binance training, per-asset model artifacts, anti-overfit evaluation, and latest-signal preview.")

symbols = cached_symbols()
watchlist = [item for item in WATCHLIST if item in symbols]
st.subheader("Model Dashboard")
cols = st.columns(3)
for idx, symbol in enumerate(watchlist):
    with cols[idx % 3]:
        render_asset_card(symbol)

train_tab, signal_tab, artifacts_tab = st.tabs(["Train", "Latest Signal", "Artifacts"])

with train_tab:
    st.subheader("Run Training")
    if st.session_state.last_run_dir:
        st.success(f"Latest completed run: {st.session_state.last_run_symbol}")
        show_artifact(st.session_state.last_run_dir)

    with st.form("training_form"):
        selected_symbol = st.selectbox(
            "Symbol",
            symbols,
            index=symbols.index("BTCUSDT") if "BTCUSDT" in symbols else 0,
            key="train_symbol",
        )
        timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=2, key="train_timeframe")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            timesteps = st.number_input(
                "Timesteps",
                min_value=1_000,
                max_value=10_000_000,
                value=1_000_000,
                step=10_000,
                key="train_timesteps",
            )
        with col_b:
            lookback_days = st.number_input(
                "API lookback days",
                min_value=30,
                max_value=3000,
                value=730,
                step=30,
                key="train_lookback_days",
            )
        with col_c:
            seed = st.number_input("Seed", min_value=0, max_value=9999, value=42, step=1, key="train_seed")

        reward_mode = st.selectbox(
            "Reward mode",
            ["pnl_drawdown", "pnl", "sharpe_proxy"],
            index=0,
            key="train_reward_mode",
        )
        policy_type = st.selectbox(
            "Policy type",
            ["mlp", "cnn1d", "recurrent_lstm"],
            index=1,
            key="train_policy_type",
        )
        hpo_trials = st.number_input(
            "Optuna trials",
            min_value=0,
            max_value=50,
            value=1,
            step=1,
            key="train_hpo_trials",
        )
        submitted = st.form_submit_button("Run", type="primary")

    if submitted:
        config = TrainingConfig(
            symbol=selected_symbol,
            timeframe=timeframe,
            total_timesteps=int(timesteps),
            lookback_days=int(lookback_days),
            reward_mode=reward_mode,
            policy_type=policy_type,
            hpo_trials=int(hpo_trials),
            seed=int(seed),
        )
        st.info(
            "Received config: "
            f"{config.symbol} | {config.timeframe} | {config.total_timesteps:,} steps | "
            f"{config.policy_type} | Optuna trials={config.hpo_trials}"
        )
        with st.spinner("Fetching Binance API data, training PPO, evaluating baselines, and saving charts..."):
            try:
                result = run_training(config)
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                st.stop()
        st.session_state.last_run_dir = result["run_dir"]
        st.session_state.last_run_symbol = selected_symbol
        st.cache_data.clear()
        st.rerun()

with signal_tab:
    st.subheader("Latest Signal")
    signal_symbol = st.selectbox("Signal symbol", symbols, index=symbols.index("BTCUSDT") if "BTCUSDT" in symbols else 0)
    signal_timeframe = st.selectbox("Signal timeframe", ["latest saved", "15m", "1h", "4h", "1d"], index=0)
    if st.button("Get latest signal"):
        try:
            payload = latest_signal(signal_symbol, None if signal_timeframe == "latest saved" else signal_timeframe)
        except Exception as exc:
            st.error(str(exc))
        else:
            st.json(payload)

with artifacts_tab:
    st.subheader("Saved Artifacts")
    selected = st.selectbox("Artifact symbol", watchlist or symbols, index=0)
    run = latest_artifact(selected)
    if run:
        show_artifact(run)
    else:
        st.info("No artifact found for this symbol yet.")
