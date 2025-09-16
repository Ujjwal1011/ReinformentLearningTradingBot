# data_processor.py

import pandas as pd
import numpy as np
import ta
import warnings

# Import custom functions from other modules

from DataProcessing.external_data import get_google_trends
from DataProcessing.indicators import calculate_dema, calculate_vwap
# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def process_ohlcvt_data(input_path, output_path):
    """
    Loads raw, sparse OHLCVT data, fills gaps to a MINUTE level, calculates
    technical indicators, and saves the final features.
    """
    # --- Configuration for Indicators ---
    DEMA_WINDOW = 20
    BBAND_WINDOW = 20
    RSI_WINDOW = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    STOCH_WINDOW = 14
    ADX_WINDOW = 14
    AROON_WINDOW = 25
    CCI_WINDOW = 20
    DEMA_CLOSE_WINDOW = 10
    
    print(f"Loading data from {input_path}...")
    
    # 1. Load and Clean Initial Data
    column_names = ['time', 'open', 'high', 'low', 'close', 'volume', 'timestamp']
    df = pd.read_csv(input_path, header=None, names=column_names, on_bad_lines='skip')

    # Convert columns to numeric type with appropriate dtypes
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce', downcast='float')
    df['time'] = pd.to_numeric(df['time'], errors='coerce')

    numeric_columns = ['open', 'high', 'low', 'close']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print("Data types after conversion:")
    print(df.dtypes)
    print("\nSample data:")
    print(df.head())
    df['datetime'] = pd.to_datetime(df['time'], unit='ms', errors='coerce')
    df.dropna(subset=['datetime'], inplace=True)
    print(df.head())
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)

    df.drop(columns=['time'], inplace=True, errors='ignore')


    print("Resampling data to minute-level and filling gaps...")
    # agg_dict = {'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}


    # df_minute = df.resample('T').agg(agg_dict)
    df_minute = df
    df_minute['volume'] = df_minute['volume'].fillna(0)
    df_minute[['open', 'high', 'low', 'close']] = df_minute[['open', 'high', 'low', 'close']].ffill()
    df_minute.dropna(inplace=True)

    print("Calculating technical indicators and features...")
    
    # 2. Basic & Technical Features
    print("Calculating log returns...")
    for col in ['open', 'high', 'low', 'close', 'volume']:
        safe_current = df_minute[col].replace(0, 1e-10)
        safe_prev = df_minute[col].shift(1).replace(0, 1e-10)
        df_minute[f'{col}_log_returns'] = np.log(safe_current / safe_prev)
    df_minute['hour'] = df_minute.index.hour
    
    print("Calculating DEMA, VWAP, Bollinger Bands, ADL, and OBV...")
    df_minute['dema'] = calculate_dema(df_minute['close'], window=DEMA_WINDOW)
    df_minute['vwap'] = calculate_vwap(df_minute)
    bb_indicator = ta.volatility.BollingerBands(close=df_minute['close'], window=BBAND_WINDOW, window_dev=2, fillna=True)
    df_minute['bband_up'] = bb_indicator.bollinger_hband()
    df_minute['bband_down'] = bb_indicator.bollinger_lband()
    df_minute['adl'] = ta.volume.acc_dist_index(high=df_minute['high'], low=df_minute['low'], close=df_minute['close'], volume=df_minute['volume'], fillna=True)
    df_minute['obv'] = ta.volume.on_balance_volume(close=df_minute['close'], volume=df_minute['volume'], fillna=True)

    print("Calculating MACD, Stochastic, Aroon, RSI, ADX, and CCI...")
    macd_indicator = ta.trend.MACD(close=df_minute['close'], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL, fillna=True)
    df_minute['macd'] = macd_indicator.macd()
    df_minute['macd_signal'] = macd_indicator.macd_signal()
    df_minute['macd_diff'] = macd_indicator.macd_diff()
    stoch_indicator = ta.momentum.StochasticOscillator(high=df_minute['high'], low=df_minute['low'], close=df_minute['close'], window=STOCH_WINDOW, smooth_window=3, fillna=True)
    df_minute['stoch_k'] = stoch_indicator.stoch()
    df_minute['stoch_d'] = stoch_indicator.stoch_signal()
    aroon_indicator = ta.trend.AroonIndicator(close=df_minute['close'], window=AROON_WINDOW, fillna=True)
    df_minute['aroon_up'] = aroon_indicator.aroon_up()
    df_minute['aroon_down'] = aroon_indicator.aroon_down()
    df_minute['rsi'] = ta.momentum.rsi(close=df_minute['close'], window=RSI_WINDOW, fillna=True)
    df_minute['adx'] = ta.trend.adx(high=df_minute['high'], low=df_minute['low'], close=df_minute['close'], window=ADX_WINDOW, fillna=True)
    df_minute['cci'] = ta.trend.cci(high=df_minute['high'], low=df_minute['low'], close=df_minute['close'], window=CCI_WINDOW, constant=0.015, fillna=True)

    # 3. Fetch and Merge Google Trends Data
    print("Fetching Google Trends data...")
    start_date = df_minute.index.min().date()
    end_date = df_minute.index.max().date()
    df_trends = get_google_trends(['Ethereum'], start_date, end_date)
    
    if not df_trends.empty:
        df_minute = df_minute.merge(df_trends, left_index=True, right_index=True, how='left')
        df_minute['trends'].ffill(inplace=True)
        df_minute['trends'].bfill(inplace=True)
    df_minute['trends'] = df_minute['trends'].fillna(0)

    # 4. Calculate Derivative/Combination Features
    print("Calculating derivative features...")
    df_minute['macd_signal_diffs'] = df_minute['macd_diff']
    df_minute['stoch'] = df_minute['stoch_k']
    df_minute['close_dema'] = calculate_dema(df_minute['close'], window=DEMA_CLOSE_WINDOW)
    df_minute['close_vwap'] = calculate_vwap(df_minute)
    df_minute['bband_up_close'] = df_minute['bband_up'] - df_minute['close']
    df_minute['close_bband_down'] = df_minute['close'] - df_minute['bband_down']
    df_minute['adl_diffs2'] = df_minute['adl'].diff(2)
    df_minute['obv_diffs2'] = df_minute['obv'].diff(2)
    
    df_minute.fillna(0, inplace=True)

    # 5. Final Column Selection and Ordering
    final_columns = [
        'open', 'high', 'low', 'close', 'volume', 'open_log_returns', 
        'high_log_returns', 'low_log_returns', 'close_log_returns', 'volume_log_returns', 
         'hour', 'dema', 'vwap', 'bband_up', 'bband_down', 'adl', 'obv', 
        'macd', 'macd_signal', 'macd_diff', 'stoch_k', 'stoch_d', 'aroon_up', 'aroon_down', 
        'rsi', 'adx', 'cci', 'trends', 'macd_signal_diffs', 'stoch', 'close_dema', 
        'close_vwap', 'bband_up_close', 'close_bband_down', 'adl_diffs2', 'obv_diffs2'
    ]
    
    for col in final_columns:
        if col not in df_minute.columns:
            df_minute[col] = 0
            
    df_final = df_minute[final_columns]
    
    df_final.to_csv(output_path, index=True)
    print(f"\n✅ Successfully processed data and saved to {output_path}")
    print(f"Final dataset shape: {df_final.shape}")