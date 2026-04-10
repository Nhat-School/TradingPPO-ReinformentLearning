import pandas as pd
import pandas_ta_classic as ta
import yfinance as yf
import requests
import time
import os


def preprocess_technical_indicators(df: pd.DataFrame):
    """
    Standardized preprocessing for any OHLCV DataFrame (Yahoo, Binance, or CSV).
    """
    # Ensure columns are properly named for technical analysis
    df.columns = [c.capitalize() for c in df.columns]
    if "Gmt time" in df.columns:
        df = df.set_index("Gmt time")
    
    # Safely ensure the index is always explicitly DateTime
    df.index = pd.to_datetime(df.index)
    
    # Ensure numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- Advanced Technicals & Scaling ----
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
    df["vol_ratio"] = df["Volume"] / df["vol_sma_20"]

    df.dropna(inplace=True)

    feature_cols = [
        "rsi_14", "atr_pct", "ma_20_slope_pct", "ma_50_slope_pct",
        "close_ma20_diff_pct", "close_ma50_diff_pct", "ma_spread_pct",
        "ma_spread_slope_pct", "macd_line", "macd_hist", "bbands_width",
        "stochrsi_k", "stochrsi_d", "mfi_14", "vol_ratio"
    ]
    return df, feature_cols

def load_yfinance_data(symbol: str = "BTC-USD", period: str = "120d", interval: str = "1h"):
    print(f"Downloading live {symbol} data from Yahoo Finance...")
    df = yf.download(tickers=symbol, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return preprocess_technical_indicators(df)

def load_binance_data(symbol="BTCUSDT", start_str="2019-01-01", end_str="2026-12-31"):
    """
    Fetches comprehensive historical data from Binance API.
    """
    cache_file = f"data/binance_{symbol}_{start_str}_{end_str}.csv"
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        df = pd.read_csv(cache_file, index_col=0)
        df.index = pd.to_datetime(df.index)
        return preprocess_technical_indicators(df)

    print(f"Fetching {symbol} from Binance ({start_str} to {end_str})...")
    os.makedirs("data", exist_ok=True)
    
    url = "https://api.binance.com/api/v3/klines"
    start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_str).timestamp() * 1000)
    
    all_klines = []
    curr_start = start_ts
    
    while curr_start < end_ts:
        params = {"symbol": symbol, "interval": "1h", "startTime": curr_start, "endTime": end_ts, "limit": 1000}
        res = requests.get(url, params=params)
        data = res.json()
        if not data or not isinstance(data, list): break
        all_klines.extend(data)
        curr_start = data[-1][0] + 1
        print(f"  Received {len(all_klines)} bars...", end="\r")
        time.sleep(0.1)

    df = pd.DataFrame(all_klines, columns=[
        'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close_time', 'Quote_vol', 'Trades', 'Taker_base', 'Taker_quote', 'Ignore'
    ])
    df['Gmt time'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df = df[['Gmt time', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    df.to_csv(cache_file)
    return preprocess_technical_indicators(df)
