from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
import requests


BINANCE_BASE_URL = "https://api.binance.com"


def to_millis(value: str | datetime) -> int:
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    text = str(value)
    if text.lower() == "now":
        return int(datetime.now(timezone.utc).timestamp() * 1000)
    if text.endswith("days ago UTC"):
        days = int(text.split(" ")[0])
        ts = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
        return int(ts.timestamp() * 1000)

    ts = pd.Timestamp(text)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def get_exchange_symbols(quote_asset: str = "USDT") -> list[str]:
    response = requests.get(f"{BINANCE_BASE_URL}/api/v3/exchangeInfo", timeout=20)
    response.raise_for_status()
    payload = response.json()
    symbols = []
    for item in payload.get("symbols", []):
        if item.get("status") == "TRADING" and item.get("quoteAsset") == quote_asset:
            symbols.append(item["symbol"])
    return sorted(symbols)


def fetch_klines(
    symbol: str,
    interval: str,
    start: str = "730 days ago UTC",
    end: str = "now",
    pause_seconds: float = 0.03,
) -> pd.DataFrame:
    start_ts = to_millis(start)
    end_ts = to_millis(end)
    all_rows: list[list] = []
    cursor = start_ts

    while cursor < end_ts:
        response = requests.get(
            f"{BINANCE_BASE_URL}/api/v3/klines",
            params={
                "symbol": symbol.upper(),
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ts,
                "limit": 1000,
            },
            timeout=20,
        )
        response.raise_for_status()
        rows = response.json()
        if not rows or not isinstance(rows, list):
            break
        all_rows.extend(rows)
        cursor = int(rows[-1][0]) + 1
        time.sleep(pause_seconds)

    if not all_rows:
        raise RuntimeError(f"No Binance bars returned for {symbol} {interval}.")

    df = pd.DataFrame(
        all_rows,
        columns=[
            "Timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Close_time",
            "Quote_volume",
            "Trades",
            "Taker_base",
            "Taker_quote",
            "Ignore",
        ],
    )
    df["Gmt time"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    keep = ["Gmt time", "Open", "High", "Low", "Close", "Volume", "Taker_base", "Quote_volume", "Trades"]
    df = df[keep].set_index("Gmt time")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()
