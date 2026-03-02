import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.data_service import DataService

# =====================
# CONFIG
# =====================
START_DATE = "2020-01-01"
TEST_START = "2023-01-01"
END_DATE = "2026-02-24"

HORIZON = 1
TOP_Q = 0.9
BOTTOM_Q = 0.1
COST = 0.001   # 0.1%
N_ESTIMATORS = 200
ROLLING_YEARS = 3  # 訓練資料滾動窗口
MIN_STOCKS = 5     # 多空最少股票數量

# =====================
# Load stock list
# =====================
json_path = ROOT / "stock_list.json"
with open(json_path, "r") as f:
    data = json.load(f)

stock_list = list(data["stocks"].keys())
print(f"Total stocks: {len(stock_list)}")

data_service = DataService()
all_df = []

# =====================
# Load all data once
# =====================
for symbol in stock_list:
    try:
        df = data_service.fetch_stock_data(symbol, start=START_DATE, end=END_DATE)
        df = data_service.add_indicators(df)
        df = data_service.add_market_regime(df)

        df["future_return"] = df["close"].shift(-1) / df["close"] - 1
        df["symbol"] = symbol
        df = df.dropna()
        all_df.append(df)
    except Exception as e:
        print(f"Skip {symbol}: {e}")

df_all = pd.concat(all_df)
df_all["date"] = pd.to_datetime(df_all["date"])
df_all = df_all.sort_values("date").reset_index(drop=True)
print("Total samples:", len(df_all))

# =====================
# Feature columns
# =====================
exclude = {"date", "symbol", "future_return", "close", "Date"}
feature_cols = [col for col in df_all.columns if col not in exclude and pd.api.types.is_numeric_dtype(df_all[col])]

# =====================
# Walk-forward monthly retraining
# =====================
portfolio_returns = []

monthly_points = pd.date_range(TEST_START, END_DATE, freq="ME")

for month_start in monthly_points:
    train_start = month_start - pd.DateOffset(years=ROLLING_YEARS)
    train_end = month_start - pd.Timedelta(days=1)

    train_df = df_all[(df_all["date"] >= train_start) & (df_all["date"] <= train_end)]
    test_df = df_all[(df_all["date"] >= month_start) & (df_all["date"] <= month_start + pd.offsets.MonthEnd(1))]

    if len(train_df) < 1000 or len(test_df) == 0:
        continue

    y_train = (train_df["future_return"] > 0).astype(int)
    X_train = train_df[feature_cols]

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=8,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    X_test = test_df[feature_cols]
    test_df = test_df.copy()
    test_df["proba"] = model.predict_proba(X_test)[:,1]

    # Daily long-short
    for date, g in test_df.groupby("date"):
        long = g[g["proba"] >= g["proba"].quantile(TOP_Q)]
        short = g[g["proba"] <= g["proba"].quantile(BOTTOM_Q)]

        # 動態調整最少股票數量
        min_stocks = max(MIN_STOCKS, int(len(g) * 0.1))
        if len(long) < min_stocks or len(short) < min_stocks:
            continue

        ret = long["future_return"].mean() - short["future_return"].mean()
        ret -= 2 * COST  # 交易成本
        portfolio_returns.append((date, ret))

# =====================
# Performance
# =====================
port_df = pd.DataFrame(portfolio_returns, columns=["date","ret"])
# 同一日期可能出現多次（多個月份 walk-forward），先聚合成一筆再補齊日期
port_df = port_df.groupby("date", as_index=False)["ret"].mean()
port_df = port_df.sort_values("date").reset_index(drop=True)

# 補齊日期，缺失日期收益為 0（reindex 要求 index 無重複）
all_dates = pd.date_range(port_df["date"].min(), port_df["date"].max(), freq="B")
port_df = port_df.set_index("date").reindex(all_dates, fill_value=0).rename_axis("date").reset_index()

port_df["cum"] = (1 + port_df["ret"]).cumprod()
annual_return = port_df["cum"].iloc[-1] ** (252/len(port_df)) - 1
sharpe = np.sqrt(252) * port_df["ret"].mean() / port_df["ret"].std()
drawdown = port_df["cum"] / port_df["cum"].cummax() - 1
mdd = drawdown.min()

# =====================
# Advanced Performance Diagnostics
# =====================

# ---- 月度報酬 ----
port_df["year"] = port_df["date"].dt.year
port_df["month"] = port_df["date"].dt.month

monthly_ret = (
    port_df
    .set_index("date")["ret"]
    .resample("M")
    .apply(lambda x: (1+x).prod()-1)
)

monthly_df = monthly_ret.to_frame("monthly_return")
monthly_df["year"] = monthly_df.index.year
monthly_df["month"] = monthly_df.index.month

# ---- 勝率 ----
win_rate = (port_df["ret"] > 0).mean()

# ---- 盈虧比 ----
avg_win = port_df.loc[port_df["ret"] > 0, "ret"].mean()
avg_loss = port_df.loc[port_df["ret"] < 0, "ret"].mean()
profit_factor = abs(avg_win / avg_loss)

# ---- Rolling Sharpe ----
rolling_6m = (
    port_df["ret"]
    .rolling(126)
    .apply(lambda x: np.sqrt(252)*x.mean()/x.std() if x.std()!=0 else 0)
)

rolling_12m = (
    port_df["ret"]
    .rolling(252)
    .apply(lambda x: np.sqrt(252)*x.mean()/x.std() if x.std()!=0 else 0)
)

# ---- Max Drawdown Period ----
drawdown = port_df["cum"] / port_df["cum"].cummax() - 1
mdd = drawdown.min()
mdd_end = drawdown.idxmin()
mdd_start = port_df["cum"][:mdd_end].idxmax()

# =====================
# Print Diagnostics
# =====================
print("========== Advanced Diagnostics ==========")
print(f"Win Rate: {win_rate:.2%}")
print(f"Profit Factor: {profit_factor:.2f}")
print(f"Max Drawdown Period: {port_df.loc[mdd_start,'date']} "
      f"to {port_df.loc[mdd_end,'date']}")
print("===========================================")

# =====================
# Plot Section
# =====================

plt.figure(figsize=(14,6))
plt.plot(port_df["date"], port_df["cum"])
plt.axvspan(port_df.loc[mdd_start,'date'],
            port_df.loc[mdd_end,'date'],
            color='red', alpha=0.2)
plt.title("Equity Curve with Max Drawdown Highlighted")
plt.grid()
plt.show()

plt.figure(figsize=(14,4))
plt.plot(port_df["date"], rolling_6m, label="6M Sharpe")
plt.plot(port_df["date"], rolling_12m, label="12M Sharpe")
plt.legend()
plt.title("Rolling Sharpe Ratio")
plt.grid()
plt.show()

plt.figure(figsize=(14,4))
plt.plot(port_df["date"], drawdown)
plt.title("Drawdown Curve")
plt.grid()
plt.show()

plt.figure(figsize=(14,4))
plt.bar(monthly_df.index, monthly_df["monthly_return"])
plt.title("Monthly Returns")
plt.grid()
plt.show()


print("================================")
print(f"Annual Return: {annual_return:.2%}")
print(f"Sharpe: {sharpe:.2f}")
print(f"Max Drawdown: {mdd:.2%}")
print("================================")

# =====================
# Plot
# =====================
plt.figure(figsize=(12,6))
plt.plot(port_df["date"], port_df["cum"])
plt.axhline(1, linestyle="--", color="gray")
plt.title("Optimized Walk-Forward Long-Short Equity Curve")
plt.grid()
plt.show()
