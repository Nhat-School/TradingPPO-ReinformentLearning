from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trading_bot.config import DEFAULT_TIMEFRAME, WATCHLIST, TrainingConfig
from trading_bot.live import latest_signal
from trading_bot.trainer import latest_artifact, load_run_summary


st.set_page_config(page_title="Trading Bot Trainer", layout="wide", initial_sidebar_state="collapsed")

if "last_run_dir" not in st.session_state:
    st.session_state.last_run_dir = None
if "last_run_symbol" not in st.session_state:
    st.session_state.last_run_symbol = None
if "active_job" not in st.session_state:
    st.session_state.active_job = None

JOBS_ROOT = SRC_ROOT / "artifacts" / "jobs"

st.markdown(
    """
    <style>
    #MainMenu, header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton {
        display: none !important;
    }
    .block-container {
        padding-top: 1.5rem;
    }
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
    div[data-testid="stMetric"] {
        background: #111820;
        border: 1px solid #26313c;
        border-radius: 10px;
        padding: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def available_symbols():
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
    train_metrics_path = run_path / "train_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        train_metrics = (
            json.loads(train_metrics_path.read_text(encoding="utf-8"))
            if train_metrics_path.exists()
            else {}
        )
        if train_metrics:
            st.markdown("Train/Test performance")
            compare_cols = st.columns(4)
            compare_cols[0].metric("Train return", f"{train_metrics.get('return_pct', 0):.2f}%")
            compare_cols[1].metric("Train max DD", f"{train_metrics.get('max_drawdown_pct', 0):.2f}%")
            compare_cols[2].metric("Test return", f"{metrics.get('return_pct', 0):.2f}%")
            compare_cols[3].metric("Test max DD", f"{metrics.get('max_drawdown_pct', 0):.2f}%")
        cols = st.columns(5)
        cols[0].metric("Final equity", f"${metrics.get('final_equity', 0):,.2f}")
        cols[1].metric("Return", f"{metrics.get('return_pct', 0):.2f}%")
        cols[2].metric("Max drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%")
        cols[3].metric("Sharpe", f"{metrics.get('sharpe_simple', 0):.2f}")
        cols[4].metric("Win rate", f"{metrics.get('win_rate_pct', 0):.1f}%")
        st.json(metrics)

    chart_files = [
        ("Train equity", "train_equity_curve.png"),
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


def _is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def discover_latest_job() -> dict | None:
    if not JOBS_ROOT.exists():
        return None
    candidates: list[dict] = []
    for status_path in JOBS_ROOT.glob("*/status.json"):
        payload = _read_json(status_path)
        if not payload:
            continue
        job_dir = status_path.parent
        parts = job_dir.name.split("_")
        candidates.append(
            {
                "pid": int(payload.get("pid", 0)),
                "job_dir": str(job_dir),
                "progress_file": str(status_path),
                "log_file": str(job_dir / "train.log"),
                "symbol": parts[0] if parts else "UNKNOWN",
                "timeframe": parts[1] if len(parts) > 1 else "",
                "updated_at": payload.get("updated_at", ""),
                "stage": payload.get("stage", ""),
            }
        )
    if not candidates:
        return None
    running = [job for job in candidates if job["pid"] and _is_process_running(job["pid"])]
    pool = running or candidates
    return sorted(pool, key=lambda item: item.get("updated_at", ""), reverse=True)[0]


def start_training_job(config: TrainingConfig) -> dict:
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    job_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    job_dir = JOBS_ROOT / f"{config.symbol}_{config.timeframe}_{job_id}"
    job_dir.mkdir(parents=True, exist_ok=False)
    progress_file = job_dir / "status.json"
    log_file = job_dir / "train.log"
    run_name = config.run_name or f"{config.symbol.lower()}_{config.timeframe}_{job_id}"
    cmd = [
        sys.executable,
        "-m",
        "trading_bot.cli",
        "train",
        "--symbol",
        config.symbol,
        "--timeframe",
        config.timeframe,
        "--timesteps",
        str(config.total_timesteps),
        "--lookback-days",
        str(config.lookback_days),
        "--reward-mode",
        config.reward_mode,
        "--policy-type",
        config.policy_type,
        "--hpo-trials",
        str(config.hpo_trials),
        "--seed",
        str(config.seed),
        "--run-name",
        run_name,
        "--progress-file",
        str(progress_file),
    ]
    if config.start_date:
        cmd.extend(["--start-date", config.start_date])
    if config.end_date:
        cmd.extend(["--end-date", config.end_date])
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT)
    with log_file.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(cmd, cwd=str(SRC_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    progress_file.write_text(
        json.dumps(
            {
                "stage": "queued",
                "progress": 0.0,
                "message": "Training job started in background",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "pid": process.pid,
                "current_steps": 0,
                "total_steps": config.total_timesteps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "pid": process.pid,
        "job_dir": str(job_dir),
        "progress_file": str(progress_file),
        "log_file": str(log_file),
        "symbol": config.symbol,
        "timeframe": config.timeframe,
    }


def show_active_job(job: dict):
    progress_path = Path(job["progress_file"])
    log_path = Path(job["log_file"])
    status = _read_json(progress_path)
    if not status:
        status = {
            "stage": "starting",
            "progress": 0.0,
            "message": "Preparing training job...",
            "pid": job.get("pid", 0),
        }
    pid = int(status.get("pid") or job.get("pid") or 0)
    running = _is_process_running(pid) if pid else False
    stage = status.get("stage", "starting")
    progress = float(status.get("progress", 0.0))
    message = status.get("message", "Waiting for trainer output...")

    st.info(f"{job['symbol']} {job['timeframe']} | {stage}: {message}")
    st.progress(progress, text=f"{progress * 100:.1f}%")
    current_steps = status.get("current_steps")
    total_steps = status.get("total_steps")
    if total_steps:
        cols = st.columns(3)
        cols[0].metric("Current step", f"{int(current_steps or 0):,}")
        cols[1].metric("Target steps", f"{int(total_steps):,}")
        cols[2].metric("Remaining", f"{max(int(total_steps) - int(current_steps or 0), 0):,}")

    run_dir = status.get("run_dir")
    if run_dir:
        st.caption("Artifact đang được ghi vào:")
        st.code(run_dir)

    refresh_col, stop_col = st.columns([1, 5])
    with refresh_col:
        if st.button("Refresh progress", key="refresh_progress"):
            st.rerun()

    if log_path.exists():
        tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:])
        with st.expander("Training log", expanded=False):
            st.code(tail or "Log is being created...")

    if running:
        st.caption("Đang train nền. Bấm Refresh progress để cập nhật số bước mà không bị tự nhảy lên đầu trang.")

    if stage == "completed" and run_dir:
        st.session_state.last_run_dir = run_dir
        st.session_state.last_run_symbol = job["symbol"]
        st.session_state.active_job = None
        st.success("Training completed.")
        if st.button("Back to main screen", key="back_to_main_after_train"):
            job_dir = Path(job.get("job_dir", ""))
            if job_dir.exists():
                shutil.rmtree(job_dir)
            st.session_state.last_run_dir = None
            st.session_state.last_run_symbol = None
            st.session_state.active_job = None
            st.rerun()
        show_artifact(run_dir)
    elif not running:
        st.session_state.active_job = None
        if stage != "completed":
            st.error("Training process stopped before completion. Check the log above.")


symbols = available_symbols()
watchlist = [item for item in WATCHLIST if item in symbols]

st.title("Trading Bot Trainer")
st.caption("Chọn tài sản, timeframe, số bước train rồi bấm Run. Dữ liệu train lấy trực tiếp từ Binance API. Live signal luôn dùng best model.")

active_job = st.session_state.active_job or discover_latest_job()
if active_job:
    st.subheader("Training Progress")
    show_active_job(active_job)
    st.divider()

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

    selected_symbol = st.selectbox(
        "Symbol",
        symbols,
        index=symbols.index("BTCUSDT") if "BTCUSDT" in symbols else 0,
        key="train_symbol",
    )
    timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=2, key="train_timeframe")
    min_binance_date = date(2017, 7, 1)
    today = date.today()
    date_col_a, date_col_b = st.columns(2)
    with date_col_a:
        start_date = st.date_input(
            "Start date",
            value=date(2020, 1, 1),
            min_value=min_binance_date,
            max_value=today,
            key="train_start_date",
        )
    with date_col_b:
        end_date = st.date_input(
            "End date",
            value=date(2023, 2, 2),
            min_value=min_binance_date,
            max_value=today,
            key="train_end_date",
        )
    col_a, col_b = st.columns(2)
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
        policy_type = st.selectbox("Model", ["cnn1d", "mlp", "recurrent_lstm"], index=0, key="train_policy_type")
    lookback_days = 730
    hpo_trials = 0
    st.caption(
        "Dữ liệu lấy theo khoảng ngày đã chọn. Split theo thời gian: 70% train, "
        "30% test OOS. UI sẽ hiện return/drawdown và chart so sánh cho train và test."
    )
    submitted = st.button("Run", type="primary", key="run_training_button")

    if submitted:
        if start_date >= end_date:
            st.error("Start date phải nhỏ hơn End date.")
        elif start_date < min_binance_date:
            st.error("Binance spot API không có dữ liệu training trước 2017-07-01.")
        else:
            config = TrainingConfig(
                symbol=selected_symbol,
                timeframe=timeframe,
                total_timesteps=int(timesteps),
                lookback_days=int(lookback_days),
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                reward_mode="pnl_drawdown",
                policy_type=policy_type,
                hpo_trials=int(hpo_trials),
                seed=42,
            )
            st.info(
                "Received config: "
                f"{config.symbol} | {config.timeframe} | {config.start_date}->{config.end_date} | "
                f"{config.total_timesteps:,} steps | {config.policy_type}"
            )
            st.session_state.active_job = start_training_job(config)
            st.rerun()

with signal_tab:
    st.subheader("Latest Signal From Best Model")
    signal_symbol = st.selectbox("Signal symbol", symbols, index=symbols.index("BTCUSDT") if "BTCUSDT" in symbols else 0)
    signal_timeframe = st.selectbox("Signal timeframe", ["best overall", "15m", "1h", "4h", "1d"], index=0)
    if st.button("Get latest signal"):
        try:
            payload = latest_signal(signal_symbol, None if signal_timeframe == "best overall" else signal_timeframe)
        except Exception as exc:
            st.error(str(exc))
        else:
            st.info(f"Using best model: {payload.get('run_dir')}")
            st.json(payload)

with artifacts_tab:
    st.subheader("Saved Artifacts")
    selected = st.selectbox("Artifact symbol", watchlist or symbols, index=0)
    run = latest_artifact(selected)
    if run:
        show_artifact(run)
    else:
        st.info("No artifact found for this symbol yet.")
