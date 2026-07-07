import yfinance as yf
import urllib.request
import pandas as pd
from datetime import datetime, timedelta

print("Testing yfinance for TATAPOWER.NS...")
try:
    ticker = yf.Ticker("TATAPOWER.NS")
    df = ticker.history(period="3mo")
    print(f"TATAPOWER.NS success, shape: {df.shape}")
except Exception as e:
    print(f"TATAPOWER.NS error: {e}")

print("Testing yfinance for ^NSEI...")
try:
    ticker = yf.Ticker("^NSEI")
    df = ticker.history(period="3mo")
    print(f"NSEI success, shape: {df.shape}")
except Exception as e:
    print(f"NSEI error: {e}")

print("Testing yfinance for INRUSD=X...")
try:
    ticker = yf.Ticker("INRUSD=X")
    df = ticker.history(period="3mo")
    print(f"INRUSD=X success, shape: {df.shape}")
except Exception as e:
    print(f"INRUSD=X error: {e}")

print("Testing World Bank API...")
try:
    url = "https://api.worldbank.org/v2/country/IN/indicator/FP.CPI.TOTL.ZG?format=json&mrv=3"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = pd.read_json(resp.read().decode())
        print(f"World Bank success, data: {data}")
except Exception as e:
    print(f"World Bank error: {e}")
