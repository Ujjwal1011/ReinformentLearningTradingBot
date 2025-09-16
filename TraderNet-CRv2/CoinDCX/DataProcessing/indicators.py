# indicators.py
import pandas as pd

def calculate_dema(series: pd.Series, window: int) -> pd.Series:
    """Calculates the Double Exponential Moving Average (DEMA)."""
    ema = series.ewm(span=window, adjust=False).mean()
    dema = 2 * ema - ema.ewm(span=window, adjust=False).mean()
    return dema

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the Volume Weighted Average Price (VWAP) using a typical price.
    This is a cumulative calculation over the entire DataFrame.
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    # Use a cumulative sum to calculate VWAP over the entire period
    # Replace potential zero-division with 1 to avoid errors on the first row
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum().replace(0, 1)
    return vwap