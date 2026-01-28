# backtest.py
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from train_model.train_model第五版 import compute_rsi

# ========== 參數 ==========
SYMBOL = "2330.TW"
START = "2018-01-01"
END   = "2026-01-01"
THRESHOLD = 0.6
HOLD_DAYS = 60

# ========== 載入模型 ==========
model = joblib.load("rf_model_2330.pkl")
print("Model loaded")

# ========== 載資料 ==========
df = yf.download(SYMBOL, start=START, end=END)
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
df = df[["date","open","high","low","close","volume"]]
df = df.sort_values("date").reset_index(drop=True)

# ========== 技術指標 ==========
df["ma5"] = df["close"].rolling(5).mean()
df["ma20"] = df["close"].rolling(20).mean()
df["ma60"] = df["close"].rolling(60).mean()
df["ma120"] = df["close"].rolling(120).mean()
df["ma240"] = df["close"].rolling(240).mean()

df["return_1"] = df["close"].pct_change(1)
df["return_5"] = df["close"].pct_change(5)

df["rsi_120"] = compute_rsi(df["close"], 120)
df["rsi_240"] = compute_rsi(df["close"], 240)
df["rsi_420"] = compute_rsi(df["close"], 420)

df["ema120"] = df["close"].ewm(span=120).mean()
df["ema240"] = df["close"].ewm(span=240).mean()
df["ema420"] = df["close"].ewm(span=420).mean()
df["ema200"] = df["close"].ewm(span=200).mean()

df = df.dropna().reset_index(drop=True)

# ========== 特徵 ==========
FEATURES = [
    "open","high","low","close","volume",
    "ma5","ma20","ma60","ma120","ma240",
    "return_1","return_5",
    "rsi_120","rsi_240","rsi_420",
    "ema120","ema240","ema420","ema200"
]

X = df[FEATURES]
proba = model.predict_proba(X)[:,1]
df["signal"] = (proba > THRESHOLD).astype(int)

# ========== 回測引擎 ==========
df["position"] = 0
holding = 0

for i in range(len(df)):
    if holding > 0:
        df.loc[i, "position"] = 1
        holding -= 1
    elif df.loc[i, "signal"] == 1:
        df.loc[i, "position"] = 1
        holding = HOLD_DAYS

df["strategy_return"] = df["position"].shift(1) * df["return_1"]
df["cum_return"] = (1 + df["strategy_return"]).cumprod()
df["drawdown"] = df["cum_return"] / df["cum_return"].cummax() - 1

# ========== 統計 ==========
print("年化報酬:", df["strategy_return"].mean()*252)
print("年化波動:", df["strategy_return"].std()*np.sqrt(252))
print("最大回撤:", df["drawdown"].min())
print("交易次數:", df["signal"].sum())

# ========== 繪圖 ==========
plt.figure(figsize=(12,4))
plt.plot(df["date"], df["cum_return"])
plt.title("Backtest Equity Curve")
plt.show()
