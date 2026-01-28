import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, log_loss
from sklearn.metrics import classification_report
import joblib  # 用來存模型

# 1. 下載過去三年的歷史資料
df = yf.download("2330.TW", start="2023-01-01", end="2026-01-01")
# df = yf.Ticker("2330.TW").history(period="6y")
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
df = df[["date", "open", "high", "low", "close", "volume"]]
df = df.sort_values("date").reset_index(drop=True)

df = df.dropna()

# 去極值
features = ["open", "high", "low", "close", "volume"]
for col in features:
    low = df[col].quantile(0.01)
    high = df[col].quantile(0.99)
    df[col] = df[col].clip(low, high)

def compute_rsi(series, window):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# 計算技術指標
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


df = df.dropna().reset_index(drop=True)

# 標籤：未來 60 天報酬
horizon = 60
df["future_return_60"] = df["close"].shift(-horizon) / df["close"] - 1


df["label"] = 2  # 預設觀望
# df.loc[df["future_return_60"] > 0.15, "label"] = 1   # 長期持有
df.loc[df["future_return_60"] > 0.12, "label"] = 1   # 長期持有
df.loc[df["future_return_60"] < -0.05, "label"] = 0  # 不適合

df = df.dropna().reset_index(drop=True)
print(df["label"].value_counts(normalize=True))

# df = df.dropna().reset_index(drop=True)


# 存檔
df.to_csv("model_input_2330.csv", index=False)
print("已產生 model_input_2330.csv")
print(df["label"].value_counts(normalize=True))

# 切訓練集與測試集
train = df[df["date"] < "2025-01-01"]
test  = df[df["date"] >= "2025-01-01"]

features = df.columns.drop(["date", "label", "future_return_60"])

X_train = train[features]
y_train = train["label"]

X_test = test[features]
y_test = test["label"]

# 建立模型
model = RandomForestClassifier(n_estimators=400, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 存模型
joblib.dump(model, "rf_model_2330.pkl")
print("模型已存檔：rf_model_2330.pkl")

# 預測
proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# 評估指標
# auc = roc_auc_score(y_test, proba)
# acc = accuracy_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# loss = log_loss(y_test, proba)

# print(f"AUC: {auc:.4f}")
# print(f"Accuracy: {acc:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"Log Loss: {loss:.4f}")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# 預測機率
# proba = model.predict_proba(X_test)[:, 1]
proba = model.predict_proba(X_test)
test["proba_buy"] = proba[:, 1]   # 類別1的機率（長期持有）

# 篩選信號
# test["signal"] = (test["proba_buy"] > 0.6).astype(int)

test["proba_avoid"] = proba[:, 0]


# 訊號策略
test = test.copy()
# test["proba"] = proba
# threshold = test["proba"].quantile(0.85)
# test["signal"] = (proba >= threshold).astype(int)
test["signal"] = (test["proba_buy"] > 0.6).astype(int)
test["position"] = test["signal"].replace(0, np.nan).ffill().fillna(0)

test["position"] = 0
holding = 0

for i in range(len(test)):
    if holding > 0:
        test.loc[test.index[i], "position"] = 1
        holding -= 1
    elif test.loc[test.index[i], "signal"] == 1:
        test.loc[test.index[i], "position"] = 1
        holding = 60


test["strategy_return"] = test["position"].shift(1) * test["return_1"]
test["cum_return"] = (1 + test["strategy_return"]).cumprod()

test["drawdown"] = test["cum_return"] / test["cum_return"].cummax() - 1
print("Max Drawdown:", test["drawdown"].min())

# for t in [0.6, 0.65, 0.7, 0.75]:
#     y_pred = (proba >= t).astype(int)
#     print(t,
#           recall_score(y_test, y_pred),
#           accuracy_score(y_test, y_pred))

print("年化報酬:", test["strategy_return"].mean() * 252)   # 年化報酬
print("年化波動:", test["strategy_return"].std() * np.sqrt(252))  # 年化波動
print("最大回撤:", test["drawdown"].min())  # 最大回撤

# 只在模型給高信心時才允許進場
# trade_days = test[test["proba"] > 0.8]
trade_days = test[test["signal"] == 1]


print("一年幾次:", len(trade_days))  # 一年幾次
print("平均報酬:", trade_days["future_return_60"].mean())

# 繪製策略曲線
plt.figure(figsize=(12,4))
plt.plot(test["date"], test["cum_return"])
plt.title("Strategy Equity Curve")
plt.show()
