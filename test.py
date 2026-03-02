import yfinance as yf
import pandas as pd

df = yf.download("2330.TW", start="2026-02-24", end="2026-03-02")
df.columns = df.columns.get_level_values(0)
df = df.rename(columns={
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume"
})
df = df.reset_index()
df["date"] = pd.to_datetime(df["Date"])
print(df)