from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta_classic as ta


def add_features(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = raw_df.copy()
    df.columns = [str(c).capitalize() for c in df.columns]

    for col in ["Open", "High", "Low", "Close", "Volume", "Taker_base"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["rsi_14"] = ta.rsi(df["Close"], length=14)
    df["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["atr_pct"] = (df["atr_14"] / df["Close"]) * 100.0
    df["ma_20"] = ta.sma(df["Close"], length=20)
    df["ma_50"] = ta.sma(df["Close"], length=50)
    df["ma_20_slope_pct"] = df["ma_20"].pct_change() * 100.0
    df["ma_50_slope_pct"] = df["ma_50"].pct_change() * 100.0
    df["close_ma20_diff_pct"] = ((df["Close"] - df["ma_20"]) / df["ma_20"]) * 100.0
    df["close_ma50_diff_pct"] = ((df["Close"] - df["ma_50"]) / df["ma_50"]) * 100.0
    df["ma_spread_pct"] = ((df["ma_20"] - df["ma_50"]) / df["ma_50"]) * 100.0
    df["ma_spread_slope_pct"] = df["ma_spread_pct"].diff()

    macd = ta.macd(df["Close"])
    df["macd_line"] = macd["MACD_12_26_9"]
    df["macd_hist"] = macd["MACDh_12_26_9"]
    bbands = ta.bbands(df["Close"], length=20)
    df["bbands_width"] = (bbands["BBU_20_2.0"] - bbands["BBL_20_2.0"]) / df["Close"] * 100.0
    stochrsi = ta.stochrsi(df["Close"])
    df["stochrsi_k"] = stochrsi["STOCHRSIk_14_14_3_3"]
    df["stochrsi_d"] = stochrsi["STOCHRSId_14_14_3_3"]
    df["mfi_14"] = ta.mfi(df["High"], df["Low"], df["Close"], df["Volume"], length=14)
    df["vol_sma_20"] = ta.sma(df["Volume"], length=20)
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
