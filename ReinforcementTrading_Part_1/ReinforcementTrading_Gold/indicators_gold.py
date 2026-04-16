import pandas as pd
import numpy as np
import requests
import time
import os

def calculate_vwap(df: pd.DataFrame):
    """
    Calculates Daily VWAP (Volume Weighted Average Price).
    Resets at the start of each daily session.
    """
    q = df["Volume"]
    p = (df["High"] + df["Low"] + df["Close"]) / 3
    # Use pandas groupby by date to reset cumsum each day
    df['Date'] = df.index.date
    df["VWAP"] = (p * q).groupby(df['Date'], group_keys=False).cumsum() / q.groupby(df['Date'], group_keys=False).cumsum()
    df.drop(columns=['Date'], inplace=True)
    return df

def preprocess_order_flow(df: pd.DataFrame):
    """
    Calculates BOTH Order Flow AND Price Action metrics.
    This gives the RL agent a complete market picture.
    """
    # Fix columns
    df.columns = [c.capitalize() for c in df.columns]
    
    # Safely ensure the index is always explicitly DateTime
    if "Gmt time" in df.columns:
        df = df.set_index("Gmt time")
        
    df.index = pd.to_datetime(df.index)
    
    # Ensure numeric
    for col in ["Open", "High", "Low", "Close", "Volume", "Taker_base"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ============================================================
    # SECTION A: PRICE ACTION FEATURES (Trend, Momentum, Volatility)
    # ============================================================
    
    # 1. Price Returns (percentage change) - captures trend direction
    df["Return_1"] = df["Close"].pct_change(1)     # 1-bar return (5 min)
    df["Return_6"] = df["Close"].pct_change(6)     # 30 min momentum
    df["Return_12"] = df["Close"].pct_change(12)   # 1 hour momentum
    df["Return_48"] = df["Close"].pct_change(48)   # 4 hour trend
    
    # 2. ATR (Average True Range) - volatility measure
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift(1)).abs()
    tr3 = (df["Low"] - df["Close"].shift(1)).abs()
    df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR_12"] = df["TR"].rolling(window=12).mean()   # 1-hour ATR
    df["ATR_48"] = df["TR"].rolling(window=48).mean()   # 4-hour ATR
    
    # 3. Moving Average crossover signals
    df["EMA_12"] = df["Close"].ewm(span=12).mean()    # 1-hour EMA
    df["EMA_48"] = df["Close"].ewm(span=48).mean()    # 4-hour EMA
    # Normalized distance: positive = bullish, negative = bearish
    df["EMA_cross"] = (df["EMA_12"] - df["EMA_48"]) / df["ATR_48"]
    
    # 4. RSI-like momentum (bounded 0-1, easier for RL to learn)
    delta_price = df["Close"].diff()
    gain = delta_price.clip(lower=0).rolling(14).mean()
    loss = (-delta_price.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df["RSI"] = 1.0 - (1.0 / (1.0 + rs))

    # ============================================================
    # SECTION B: ORDER FLOW FEATURES (Normalized for stability)
    # ============================================================
    
    # 1. Order Flow Delta (Buy - Sell) — NORMALIZED as ratio [-1, 1]
    df["Buy_vol"] = df["Taker_base"]
    df["Sell_vol"] = df["Volume"] - df["Buy_vol"]
    total_vol = df["Buy_vol"] + df["Sell_vol"]
    total_vol = total_vol.replace(0, 1e-10)  # avoid division by zero
    df["Delta_ratio"] = (df["Buy_vol"] - df["Sell_vol"]) / total_vol
    
    # 2. Buy/Sell volume ratio (0 = all sell, 1 = all buy)
    df["BuySell_ratio"] = df["Buy_vol"] / total_vol
    
    # 3. CVD (Cumulative Volume Delta) — as SLOPE over 12 bars instead of raw value
    df['_date'] = df.index.date
    raw_delta = df["Buy_vol"] - df["Sell_vol"]
    df["Cvd_raw"] = raw_delta.groupby(df['_date']).cumsum()
    df["Cvd_slope"] = df["Cvd_raw"].diff(12) / (df["ATR_12"] * df["Volume"].rolling(12).mean() + 1e-10)
    df.drop(columns=['_date', 'Cvd_raw'], inplace=True)
    
    # 4. Daily VWAP - distance normalized by ATR
    df = calculate_vwap(df)
    df["Vwap_diff"] = (df["Close"] - df["VWAP"]) / (df["ATR_12"] + 1e-10)
    
    # 5. Big Trades Indicator (Volume Spikes)
    df["Vol_sma_60"] = df["Volume"].rolling(window=60).mean()
    df["Big_trade_flag"] = np.where(df["Volume"] > (df["Vol_sma_60"] * 3), 1.0, 0.0)
    
    # 6. Relative Volume (current volume vs average — normalized)
    df["Rel_volume"] = df["Volume"] / (df["Vol_sma_60"] + 1e-10)
    df["Rel_volume"] = df["Rel_volume"].clip(0, 10)  # cap at 10x average
    
    # ============================================================
    # CLEANUP
    # ============================================================
    df.dropna(inplace=True)
    
    # Clip extreme outliers (±5 std) on all feature columns
    feature_cols = [
        # Price Action
        "Return_1", "Return_6", "Return_12", "Return_48",
        "EMA_cross", "RSI",
        # Order Flow
        "Delta_ratio", "BuySell_ratio", "Cvd_slope",
        "Vwap_diff", "Big_trade_flag", "Rel_volume"
    ]
    
    for col in feature_cols:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col] = df[col].clip(mean - 5*std, mean + 5*std)
    
    return df, feature_cols

def load_binance_data(symbol="PAXGUSDT", interval="15m", start_str="1700 days ago UTC", end_str="now"):
    """
    Optimized for RAM. 
    Fetches data using float32 to prevent overflow. Target ~500k candles.
    """
    print(f"Fetching {symbol} {interval} data (MAX HISTORY) directly to RAM...")
    
    url = "https://api.binance.com/api/v3/klines"
    
    if start_str.endswith("ago UTC"):
        days = int(start_str.split(" ")[0])
        start_ts = int((pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days)).timestamp() * 1000)
    else:
        start_ts = int(pd.Timestamp(start_str, tz='UTC').timestamp() * 1000)
        
    end_ts = int(pd.Timestamp.now(tz='UTC').timestamp() * 1000) if end_str == "now" else int(pd.Timestamp(end_str, tz='UTC').timestamp() * 1000)
    
    all_klines = []
    curr_start = start_ts
    total_received = 0
    
    # Use a temporary file to store chunks
    temp_file = f"data/gold_{interval}_temp.csv"
    if os.path.exists(temp_file): os.remove(temp_file)
    os.makedirs("data", exist_ok=True)

    header = True
    while curr_start < end_ts:
        params = {"symbol": symbol, "interval": interval, "startTime": curr_start, "endTime": end_ts, "limit": 1000}
        try:
            res = requests.get(url, params=params, timeout=10)
            data = res.json()
            if not data or not isinstance(data, list) or "code" in data: break
            
            all_klines.extend(data)
            curr_start = data[-1][0] + 1
            total_received += len(data)
            
            # Every 50,000 bars, save to disk and clear RAM
            if len(all_klines) >= 50000:
                df_chunk = pd.DataFrame(all_klines, columns=[
                    'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'Close_time', 'Quote_vol', 'Trades', 'Taker_base', 'Taker_quote', 'Ignore'
                ])
                # Keep only what we need and use float32
                df_chunk = df_chunk[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Taker_base']].astype('float32')
                df_chunk.to_csv(temp_file, mode='a', header=header, index=False)
                header = False
                all_klines = [] # CLEAR RAM
                print(f"  Saved to disk. Total: {total_received}...", end="\r")
                
            time.sleep(0.01)
        except Exception as e:
            time.sleep(1)
            continue
            
        if total_received >= 500000: # Limit to ~500k bars as requested
            break
            
    # Save the remaining bars
    if all_klines:
        df_chunk = pd.DataFrame(all_klines, columns=[
            'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close_time', 'Quote_vol', 'Trades', 'Taker_base', 'Taker_quote', 'Ignore'
        ]).astype('float32')
        df_chunk = df_chunk[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Taker_base']]
        df_chunk.to_csv(temp_file, mode='a', header=header, index=False)

    print(f"\nFinished fetching {total_received} bars. Loading final dataset from disk...")
    
    # Load the whole processed thing back for indicators
    final_df = pd.read_csv(temp_file, dtype='float32')
    final_df['Gmt time'] = pd.to_datetime(final_df['Timestamp'], unit='ms')
    final_df = final_df[['Gmt time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Taker_base']]
    
    # Clean up temp file
    os.remove(temp_file)
    
    return preprocess_order_flow(final_df)

if __name__ == "__main__":
    # Test fetch
    df, features = load_binance_data("PAXGUSDT", interval="15m", start_str="1 days ago UTC", end_str="now")
    print(f"Features: {features}")
    print(df[features].tail())
