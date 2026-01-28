import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 下載歷史數據 (以 2020 全年為例，作為訓練集)
df = yf.download("2330.TW", start="2020-01-01", end="2020-12-31")
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

print("\n前幾筆數據預覽：")
print(df[['Open', 'High', 'Low', 'Close', 'Volume']].head(10))

# 統一欄位名稱
df = df.rename(columns={
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume"
})

# 轉成普通欄位
df = df.reset_index()
df["date"] = pd.to_datetime(df["Date"])
df = df[["date", "open", "high", "low", "close", "volume"]]
df = df.sort_values("date").reset_index(drop=True)


print("缺值統計：")
print(df.isna().sum())

df = df.dropna()


features = ["open", "high", "low", "close", "volume"]

for col in features:
    low = df[col].quantile(0.01)
    high = df[col].quantile(0.99)
    df[col] = df[col].clip(low, high)
    
    
# MA
df["ma5"] = df["close"].rolling(5).mean()
df["ma20"] = df["close"].rolling(20).mean()

# 報酬率
df["return_1"] = df["close"].pct_change(1)
df["return_5"] = df["close"].pct_change(5)


df = df.dropna().reset_index(drop=True)


df["future_return"] = df["close"].shift(-5) / df["close"] - 1
df["label"] = (df["future_return"] > 0.02).astype(int)
df = df.dropna().reset_index(drop=True)


df.to_csv("model_input_2330.csv", index=False)
print("已產生 model_input_2330.csv")

print(df["label"].value_counts())
print(df["label"].value_counts(normalize=True))


# df["label_cum"] = df["label"].cumsum()

# plt.figure(figsize=(12,4))
# plt.plot(df["date"], df["label_cum"])
# plt.title("Cumulative Label")
# plt.show()


train = df[df["date"] < "2020-10-01"]
test  = df[df["date"] >= "2020-10-01"]

from sklearn.ensemble import RandomForestClassifier

features = df.columns.drop(["date", "label", "future_return"])

X_train = train[features]
y_train = train["label"]

X_test = test[features]
y_test = test["label"]

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)


from sklearn.metrics import roc_auc_score

proba = model.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, proba)
print("AUC:", auc)


test = test.copy()
test["proba"] = proba

# test["signal"] = 0
threshold = test["proba"].quantile(0.7)  # 取前 30% 機率作買入
test["signal"] = (test["proba"] >= threshold).astype(int)
test.loc[test["proba"] > 0.5, "signal"] = 1
# test.loc[test["proba"] < 0.4, "signal"] = -1

test["position"] = test["signal"].replace(0, np.nan).ffill().fillna(0)
test["strategy_return"] = test["position"].shift(1) * test["return_1"]

test["cum_return"] = (1 + test["strategy_return"]).cumprod()
test["drawdown"] = test["cum_return"] / test["cum_return"].cummax() - 1
test.loc[test["drawdown"] < -0.1, "position"] = 0

print(test["strategy_return"].describe())
print((1 + test["strategy_return"]).cumprod().tail(10))
print(test["position"].value_counts())



plt.plot(test["date"], test["cum_return"])
plt.title("Strategy Equity Curve")
plt.show()
