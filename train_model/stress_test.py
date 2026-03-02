import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib

# 設定支援中文的字型（Linux / Mac 通常可用）
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "SimHei", "Microsoft JhengHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

_SCRIPT_DIR = Path(__file__).resolve().parent
_MODEL_PATH = str(_SCRIPT_DIR / "rf_model_multi_No7.pkl")

TOP_PCT = 0.1        # 前 10%
REB_FREQ = "M"       # 每月調倉
MARKET_SYMBOL = "^TWII"  # 用於 Beta / Alpha 計算

# ---------- 讀股票清單 ----------
def load_stock_list(path="stock_list.json"):
    with open(path, "r") as f:
        data = json.load(f)
    return list(data["stocks"].keys())

# ---------- 績效指標 ----------
def performance_metrics(returns):
    if returns is None or len(returns) == 0:
        return {
            "Total Return": np.nan,
            "Annual Return": np.nan,
            "Annual Volatility": np.nan,
            "Sharpe Ratio": np.nan,
            "Max Drawdown": np.nan,
            "Win Rate": np.nan,
        }
    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    annual_return = (1 + total_return) ** (12 / len(returns)) - 1
    annual_vol = returns.std() * np.sqrt(12)
    sharpe = annual_return / annual_vol if annual_vol != 0 else 0
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1
    max_dd = drawdown.min()
    win_rate = (returns > 0).mean()
    return {
        "Total Return": total_return,
        "Annual Return": annual_return,
        "Annual Volatility": annual_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Win Rate": win_rate
    }

# ---------- 回測核心 ----------
def backtest(df, start=None, end=None, market_returns=None):
    model_bundle = joblib.load(_MODEL_PATH)
    model = model_bundle["model"]
    FEATURES = model_bundle["features"]

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    if start:
        df = df[df["date"] >= start]
    if end:
        df = df[df["date"] <= end]

    rebalance_dates = df.groupby(df["date"].dt.to_period(REB_FREQ))["date"].max().values
    portfolio_returns = []
    top_returns_all = []

    for date in rebalance_dates:
        monthly_data = df[df["date"] == date].copy()
        if len(monthly_data) == 0:
            continue

        X = monthly_data[FEATURES].fillna(0)
        monthly_data["pred"] = model.predict(X)
        monthly_data["rank"] = monthly_data["pred"].rank(pct=True)
        top = monthly_data[monthly_data["rank"] >= 1 - TOP_PCT]

        ret = top["future_return_1y"].mean()
        top_returns_all.append(ret)
        if not np.isnan(ret):
            portfolio_returns.append(ret / 12)  # 月報酬

    returns = pd.Series(portfolio_returns)
    metrics = performance_metrics(returns)

    # 計算 Beta / Alpha / 信息比率
    if market_returns is not None and len(returns) > 0:
        mkt_monthly = (1 + market_returns).resample("ME").prod() - 1
        if start is not None and end is not None:
            mkt_monthly = mkt_monthly.loc[start:end]
        mkt_monthly = mkt_monthly.dropna()
        n = min(len(returns), len(mkt_monthly))
        if n >= 2:
            port_slice = returns.values[:n]
            mkt_slice = mkt_monthly.values[:n]
            cov = np.cov(port_slice, mkt_slice)[0, 1]
            var = np.var(mkt_slice)
            beta = cov / var if var != 0 else np.nan
            mkt_ann = float(mkt_monthly.mean() * 12)
            alpha = metrics["Annual Return"] - beta * mkt_ann if pd.notna(beta) else np.nan
            vol = metrics["Annual Volatility"]
            ir = (metrics["Annual Return"] - mkt_ann) / vol if vol and vol > 0 else np.nan
            metrics["Beta"] = beta
            metrics["Annual Alpha"] = alpha
            metrics["Information Ratio"] = ir
        else:
            metrics["Beta"] = np.nan
            metrics["Annual Alpha"] = np.nan
            metrics["Information Ratio"] = np.nan

    return returns, metrics, top_returns_all

# ---------- 壓力測試 ----------
def stress_test(df):
    print("\n===== 壓力測試開始 =====\n")

    import yfinance as yf
    mkt = yf.download(MARKET_SYMBOL, start=df["date"].min(), end=df["date"].max())
    mkt["mkt_return"] = mkt["Close"].pct_change()
    mkt_returns = mkt["mkt_return"].fillna(0)

    results = {}

    extreme_periods = {
        "COVID 崩盤": ("2020-02-01", "2020-04-30"),
        "2008 金融危機": ("2008-09-01", "2008-12-31"),
        "2022 通膨沖擊": ("2022-01-01", "2022-06-30"),
    }

    # 壓力測試
    for name, (start, end) in extreme_periods.items():
        returns, metrics, top_returns = backtest(df, start=start, end=end, market_returns=mkt_returns)
        results[name] = {
            "metrics": metrics,
            "top_returns": top_returns,
            "cumulative": (1 + returns).cumprod()
        }
        print(f"===== {name} 壓力測試結果 =====")
        if len(returns) == 0:
            print("  該區間無資料或無有效報酬")
            for k in metrics.keys():
                print(f"  {k}: N/A")
        else:
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}" if pd.notna(v) else f"  {k}: N/A")
        print()

    # 蒙地卡羅隨機化
    print("===== 蒙地卡羅隨機化報酬測試 =====")
    for i in range(5):
        shuffled_returns = df["future_return_1y"].dropna().sample(frac=1).reset_index(drop=True)
        returns, metrics, top_returns = backtest(df.assign(future_return_1y=shuffled_returns), market_returns=mkt_returns)
        results[f"蒙地卡羅模擬{i+1}"] = {
            "metrics": metrics,
            "top_returns": top_returns,
            "cumulative": (1 + returns).cumprod()
        }
        tr = metrics["Total Return"]
        print(f"模擬 {i+1}: Total Return = {tr:.4f}" if pd.notna(tr) else f"模擬 {i+1}: Total Return = N/A")
    print()

    # 缺失資料 / 異常值測試
    print("===== 資料缺失/異常值測試 =====")
    df_missing = df.copy()
    df_missing.loc[df_missing.sample(frac=0.01).index, "return_1"] = np.nan
    returns, metrics, top_returns = backtest(df_missing, market_returns=mkt_returns)
    results["缺失資料測試"] = {"metrics": metrics, "top_returns": top_returns, "cumulative": (1 + returns).cumprod()}
    print("缺失資料測試結果:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if pd.notna(v) else f"  {k}: N/A")
    print()

    df_outliers = df.copy()
    df_outliers["return_1"] = df_outliers["return_1"].clip(-0.05, 0.05)
    returns, metrics, top_returns = backtest(df_outliers, market_returns=mkt_returns)
    results["特徵異常值測試"] = {"metrics": metrics, "top_returns": top_returns, "cumulative": (1 + returns).cumprod()}
    print("特徵異常值測試結果:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if pd.notna(v) else f"  {k}: N/A")
    print("\n===== 壓力測試完成 =====")

    # ---------- 統一繪圖 ----------
    for name, data in results.items():
        if len(data["cumulative"]) > 0:
            plt.figure(figsize=(12,5))
            plt.plot(data["cumulative"], label=f"{name} 累積報酬")
            plt.title(f"{name} 累積報酬曲線")
            plt.legend()
            plt.show()

            plt.figure(figsize=(12,5))
            plt.plot(data["top_returns"], label=f"{name} Top10% 月報酬")
            plt.title(f"{name} Top10% 月報酬")
            plt.axhline(0, color="red", linestyle="--")
            plt.legend()
            plt.show()

    # ==============================
    # 統一比較圖表
    # ==============================

    # print("\n===== 繪製統一比較圖 =====")

    # # ---- 累積報酬統一圖 ----
    # plt.figure(figsize=(14,6))

    # for name, data in results.items():
    #     cum = data["cumulative"]
    #     if len(cum) > 0:
    #         # 重設 index 讓不同期間可以一起畫
    #         cum_reset = cum.reset_index(drop=True)
    #         plt.plot(cum_reset, label=name)

    # plt.title("壓力測試－累積報酬比較")
    # plt.xlabel("月份")
    # plt.ylabel("累積報酬")
    # plt.legend()
    # plt.grid(True)
    # plt.show()


    # # ---- Top10% 月報酬統一圖 ----
    # plt.figure(figsize=(14,6))

    # for name, data in results.items():
    #     top_ret = data["top_returns"]
    #     if len(top_ret) > 0:
    #         top_series = pd.Series(top_ret)
    #         plt.plot(top_series.reset_index(drop=True), label=name)

    # plt.title("壓力測試－Top10% 月報酬比較")
    # plt.axhline(0, linestyle="--")
    # plt.xlabel("月份")
    # plt.ylabel("月報酬")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

# ---------- 主程式 ----------
def main():
    import sys
    sys.path.insert(0, str(_SCRIPT_DIR.parent.parent))
    import train_model_copy

    stock_list_path = str(_SCRIPT_DIR.parent.parent / "stock_list.json")
    print("載入資料中...")
    df = train_model_copy.download_and_prepare_data(stock_list_path=stock_list_path)
    if len(df) == 0:
        print("無資料可回測")
        return
    stress_test(df)

if __name__ == "__main__":
    main()