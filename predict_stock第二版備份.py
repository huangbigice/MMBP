import joblib
import yfinance as yf
import pandas as pd
from train_model.train_model第五版 import compute_rsi
from system_rating import system_rating, get_latest_fund_score

# ----------------------
# 輸入股票代碼
# ----------------------
SYMBOL = input("輸入股票代碼 (例: 2330.TW): ")

# ----------------------
# 載入模型
# ----------------------
model = joblib.load("rf_model_2330.pkl")
print("模型類別:", model.classes_)

# ----------------------
# 下載股票資料
# ----------------------
df = yf.download(SYMBOL, period="10y")
if isinstance(df.columns, pd.MultiIndex):
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
df = df.sort_values("date").reset_index(drop=True)

# ----------------------
# 技術指標
# ----------------------
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

# ----------------------
# 基本面分數
# ----------------------
fund_score_dict = get_latest_fund_score(SYMBOL)
df["fund_score"] = fund_score_dict["total_score"]

# ----------------------
# 欄位順序
# ----------------------
FEATURES = [
    "open","high","low","close","volume",
    "ma5","ma20","ma60","ma120","ma240",
    "return_1","return_5",
    "rsi_120","rsi_240","rsi_420",
    "ema120","ema240","ema420","ema200",
    "fund_score"
]

# ----------------------
# 只取最新一天進行預測
# ----------------------
df_latest = df.iloc[-1:].copy()

# 檢查是否有 NaN
if df_latest[FEATURES].isna().any().any():
    raise ValueError("⚠️ 最新一天特徵有缺失值，資料不足以進行預測")

X = df_latest[FEATURES]

# ----------------------
# 預測
# ----------------------
proba = model.predict_proba(X)[0]
classes = model.classes_

# 對應類別名稱
label_map = {0: "不建議持有", 1: "長期持有", 2: "觀望"}

print("\n==== 模型判斷 ====")
for i, c in enumerate(classes):
    print(f"{label_map[c]}: {round(proba[i],3)}")

# ----------------------
# 系統評分
# ----------------------
df_latest["proba_buy"] = proba[list(classes).index(1)]  # label=1 長期持有
rated_df = system_rating(df_latest)  # 這裡會產生 tech_score, system_score, recommendation

# ----------------------
# 顯示結果
# ----------------------
row = rated_df.iloc[0]

print("\n==== 系統決策 ====")
print(f"System score: {row['system_score']}")
print(f"技術面分數: {row['tech_score']}")
print(f"基本面分數: {row['fund_score']}")
print(f"模型信心: {row['proba_buy']}")
print("\n👉 最終建議:", row["recommendation"])
