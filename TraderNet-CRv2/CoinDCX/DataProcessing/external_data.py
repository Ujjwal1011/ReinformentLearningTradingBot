# external_data.py
import pandas as pd
from pytrends.request import TrendReq
import time
import random

def get_google_trends(keywords, start_date, end_date):
    """
    Fetches daily Google Trends data for a list of keywords within a date range.
    Handles potential rate limiting from the API.
    """
    print(f"Fetching Google Trends for keywords: {keywords}")
    pytrends = TrendReq(hl='en-US', tz=330) # tz=330 is for India Standard Time
    timeframe = f'{start_date.strftime("%Y-%m-%d")} {end_date.strftime("%Y-%m-%d")}'
    
    try:
        pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo='', gprop='')
        trends_df = pytrends.interest_over_time()
        time.sleep(0.5)
        if not trends_df.empty:
            trends_df.rename(columns={keywords[0]: 'trends', 'isPartial': 'is_partial'}, inplace=True)
            # Resample to daily frequency to ensure no gaps if trends data is sparse
            return trends_df[['trends']].resample('D').mean() 
        
    except Exception as e:
        print(f"Could not fetch Google Trends data due to an error: {e}")
        # To avoid breaking the main script, return an empty DataFrame
        
    return pd.DataFrame()