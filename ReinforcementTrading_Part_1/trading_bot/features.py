from __future__ import annotations

import numpy as np
import pandas as pd


def _sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def _stochrsi(close: pd.Series, length: int = 14, k: int = 3, d: int = 3) -> tuple[pd.Series, pd.Series]:
    rsi = _rsi(close, length)
    lowest = rsi.rolling(length, min_periods=length).min()
    highest = rsi.rolling(length, min_periods=length).max()
    raw = ((rsi - lowest) / (highest - lowest + 1e-10)) * 100.0
    stoch_k = raw.rolling(k, min_periods=k).mean()
    stoch_d = stoch_k.rolling(d, min_periods=d).mean()
    return stoch_k, stoch_d


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 14) -> pd.Series:
    typical = (high + low + close) / 3.0
    money_flow = typical * volume
    direction = typical.diff()
    positive = money_flow.where(direction > 0, 0.0)
    negative = money_flow.where(direction < 0, 0.0)
    positive_sum = positive.rolling(length, min_periods=length).sum()
    negative_sum = negative.rolling(length, min_periods=length).sum()
    ratio = positive_sum / (negative_sum + 1e-10)
    return 100.0 - (100.0 / (1.0 + ratio))


def add_features(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = raw_df.copy()
    df.columns = [str(c).capitalize() for c in df.columns]

    for col in ["Open", "High", "Low", "Close", "Volume", "Taker_base"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["rsi_14"] = _rsi(df["Close"], length=14)
    df["atr_14"] = _atr(df["High"], df["Low"], df["Close"], length=14)
    df["atr_pct"] = (df["atr_14"] / df["Close"]) * 100.0
    df["ma_20"] = _sma(df["Close"], length=20)
    df["ma_50"] = _sma(df["Close"], length=50)
    df["ma_20_slope_pct"] = df["ma_20"].pct_change() * 100.0
    df["ma_50_slope_pct"] = df["ma_50"].pct_change() * 100.0
    df["close_ma20_diff_pct"] = ((df["Close"] - df["ma_20"]) / df["ma_20"]) * 100.0
    df["close_ma50_diff_pct"] = ((df["Close"] - df["ma_50"]) / df["ma_50"]) * 100.0
    df["ma_spread_pct"] = ((df["ma_20"] - df["ma_50"]) / df["ma_50"]) * 100.0
    df["ma_spread_slope_pct"] = df["ma_spread_pct"].diff()

    macd_line = _ema(df["Close"], 12) - _ema(df["Close"], 26)
    macd_signal = _ema(macd_line, 9)
    df["macd_line"] = macd_line
    df["macd_hist"] = macd_line - macd_signal
    bb_mid = _sma(df["Close"], length=20)
    bb_std = df["Close"].rolling(20, min_periods=20).std()
    df["bbands_width"] = ((bb_mid + 2.0 * bb_std) - (bb_mid - 2.0 * bb_std)) / df["Close"] * 100.0
    df["stochrsi_k"], df["stochrsi_d"] = _stochrsi(df["Close"])
    df["mfi_14"] = _mfi(df["High"], df["Low"], df["Close"], df["Volume"], length=14)
    df["vol_sma_20"] = _sma(df["Volume"], length=20)
    df["vol_ratio"] = df["Volume"] / (df["vol_sma_20"] + 1e-10)

    feature_cols = [
        "rsi_14",
        "atr_pct",
        "ma_20_slope_pct",
        "ma_50_slope_pct",
        "close_ma20_diff_pct",
        "close_ma50_diff_pct",
        "ma_spread_pct",
        "ma_spread_slope_pct",
        "macd_line",
        "macd_hist",
        "bbands_width",
        "stochrsi_k",
        "stochrsi_d",
        "mfi_14",
        "vol_ratio",
    ]

    if "Taker_base" in df.columns:
        buy = df["Taker_base"]
        sell = (df["Volume"] - buy).clip(lower=0)
        total = (buy + sell).replace(0, 1e-10)
        df["delta_ratio"] = (buy - sell) / total
        df["buy_sell_ratio"] = buy / total
        df["rel_volume"] = (df["Volume"] / (df["Volume"].rolling(60).mean() + 1e-10)).clip(0, 10)
        raw_delta = buy - sell
        df["cvd_slope"] = raw_delta.cumsum().diff(12) / (
            df["atr_14"] * df["Volume"].rolling(12).mean() + 1e-10
        )
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        date_key = df.index.date
        vwap = (typical * df["Volume"]).groupby(date_key).cumsum() / df["Volume"].groupby(date_key).cumsum()
        df["vwap_diff"] = (df["Close"] - vwap) / (df["atr_14"] + 1e-10)
        feature_cols += ["delta_ratio", "buy_sell_ratio", "rel_volume", "cvd_slope", "vwap_diff"]

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    for col in feature_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std and std > 0:
            df[col] = df[col].clip(mean - 6 * std, mean + 6 * std)
    return df, feature_cols
