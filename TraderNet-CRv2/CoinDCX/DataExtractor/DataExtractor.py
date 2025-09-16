import requests
import time
import shutil

import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# --- Helper Functions (from your original script) ---

def to_millis(dt_str):
    """Converts a datetime string to milliseconds."""
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    return int(dt.timestamp() * 1000)

def interval_to_millis(interval_str):
    """Converts an interval string (e.g., '5m', '1h') to milliseconds."""
    unit = interval_str[-1]
    value = int(interval_str[:-1])
    if unit == 'm':
        return value * 60 * 1000
    elif unit == 'h':
        return value * 60 * 60 * 1000
    elif unit == 'd':
        return value * 24 * 60 * 60 * 1000
    else:
        raise ValueError("Unknown interval unit. Use 'm', 'h', or 'd'.")

def fetch_chunk(pair, interval, start_time, end_time, limit):
    """Fetches a single chunk of candle data from the API."""
    url = "https://public.coindcx.com/market_data/candles"
    params = {
        "pair": pair,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": limit
    }
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        print(f"Fetched {len(data)} candles from {datetime.fromtimestamp(start_time // 1000)}")
        return data
    except Exception as e:
        print(f"Error fetching chunk starting at {datetime.fromtimestamp(start_time // 1000)}: {e}")
        return []

# --- Main Logic Function ---

def fetch_and_save_data(pair, interval, start_date, end_date, output_file, limit, workers):
    """Main function to orchestrate fetching and saving the data."""
    print(f"Starting data fetch for {pair} at {interval} interval.")
    print(f"Date Range: {start_date} to {end_date}")

    # 1. Prepare time chunks
    start_time_ms = to_millis(f"{start_date} 00:00:00")
    end_time_ms = to_millis(f"{end_date} 00:00:00")
    interval_ms = interval_to_millis(interval)
    chunk_duration = interval_ms * limit

    time_chunks = []
    current_start = start_time_ms
    while current_start < end_time_ms:
        chunk_end = min(current_start + chunk_duration, end_time_ms)
        time_chunks.append((current_start, chunk_end))
        current_start = chunk_end + 1 # Add 1ms to avoid overlap

    print(f"Split date range into {len(time_chunks)} chunks to fetch.")

    # 2. Fetch data in parallel
    all_data = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_chunk = {executor.submit(fetch_chunk, pair, interval, s, e, limit): (s, e) for s, e in time_chunks}
        for future in as_completed(future_to_chunk):
            result = future.result()
            if result:
                all_data.extend(result)

    if not all_data:
        print("\n❌ No data was fetched. Please check your parameters and the API status.")
        return

    # 3. Convert to DataFrame and process
    df = pd.DataFrame(all_data, columns=["time", "open", "high", "low", "close", "volume"])
    df.drop_duplicates(subset=["time"], inplace=True)
    df["timestamp"] = pd.to_datetime(df["time"], unit='ms')
    df.sort_values("timestamp", inplace=True)
    
    # 4. Save to CSV
    df.to_csv(output_file, index=False)
    print(f"\n✅ Success! Saved {len(df)} unique data points to {output_file}")

    source = output_file
    destination = rf'C:\Users\ujjwa_n18433z\Desktop\ujjwal\DAIICT\Minor Project\TraderNet-CRv2\CoinDCX\Database\raw\{output_file}'

    shutil.move(source, destination)


# --- Command-Line Interface ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical candle data from CoinDCX API in parallel.")
    
    parser.add_argument("--pair", type=str, default="B-ETH_USDT", help="The trading pair (e.g., B-BTC_USDT).")
    parser.add_argument("--interval", type=str, default="5m", help="Candle interval (e.g., '1m', '5m', '1h').")
    parser.add_argument("--start", type=str, default="2023-01-01", help="Start date in YYYY-MM-DD format.")
    parser.add_argument("--end", type=str, default="2023-11-03", help="End date in YYYY-MM-DD format.")
    parser.add_argument("--output", type=str, help="Output CSV file name. Defaults to a generated name like 'b-eth_usdt_5m.csv'.")
    parser.add_argument("--limit", type=int, default=500, help="Number of candles per API request (max 500).")
    parser.add_argument("--workers", type=int, default=20, help="Number of parallel download threads.")
    
    args = parser.parse_args()

    # If no output file is specified, generate one automatically
    if not args.output:
        args.output = f"{args.pair.lower()}_{args.interval}.csv"

    fetch_and_save_data(
        pair=args.pair,
        interval=args.interval,
        start_date=args.start,
        end_date=args.end,
        output_file=args.output,
        limit=args.limit,
        workers=args.workers
    )