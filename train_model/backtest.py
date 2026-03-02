import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent.parent
MODEL_PATH = str(_SCRIPT_DIR / "rf_model_multi_No7.pkl")

TOP_PCT = 0.1        # 前 10%
REB_FREQ = "M"       # 每月調倉

# ========= 讀股票清單 =========
def load_stock_list(path="stock_list.json"):
    with open(path, "r") as f:
        data = json.load(f)
    return list(data["stocks"].keys())

# ========= 績效指標 =========
def performance_metrics(returns):

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
        "Win Rate": win_rate,
    }


def backtest(df, transaction_cost=0.006):

    model_bundle = joblib.load(MODEL_PATH)
    model = model_bundle["model"]
    FEATURES = model_bundle["features"]

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])

    df = df[df["date"] >= "2019-01-01"]

    # ==============================
    # 1️⃣ 轉成「月資料」避免日資料錯位
    # ==============================
    df["year_month"] = df["date"].dt.to_period("M")

    monthly = (
        df.groupby(["ticker", "year_month"])
        .last()
        .reset_index()
    )

    monthly["date"] = monthly["year_month"].dt.to_timestamp("M")

    # ==============================
    # 2️⃣ 計算真實「下一個月報酬」
    # ==============================
    monthly["next_close"] = (
        monthly.groupby("ticker")["close"].shift(-1)
    )

    monthly["next_month_return"] = (
        monthly["next_close"] / monthly["close"] - 1
    )

    # ==============================
    # 3️⃣ Benchmark (0050)
    # ==============================
    benchmark = monthly[monthly["ticker"] == "0050.TW"].copy()
    benchmark["bench_ret"] = benchmark["next_month_return"]
    benchmark = benchmark[["date", "bench_ret"]]

    # ==============================
    # 4️⃣ 月調倉回測
    # ==============================
    rebalance_dates = monthly["date"].sort_values().unique()

    portfolio_returns = []
    benchmark_returns = []
    prev_holdings = set()

    for i in range(len(rebalance_dates) - 1):

        decision_date = rebalance_dates[i]

        month_data = monthly[
            monthly["date"] == decision_date
        ].copy()

        if len(month_data) == 0:
            continue

        # ===== 模型預測（只用當下特徵）=====
        X = month_data[FEATURES].fillna(0)
        month_data["pred"] = model.predict(X)
        month_data["rank"] = month_data["pred"].rank(pct=True)

        top = month_data[
            month_data["rank"] >= 1 - TOP_PCT
        ]

        if len(top) == 0:
            continue

        current_holdings = set(top["ticker"])

        # ===== 換手率成本 =====
        turnover = len(
            current_holdings.symmetric_difference(prev_holdings)
        ) / max(len(current_holdings), 1)

        cost = turnover * transaction_cost
        prev_holdings = current_holdings

        # ===== 使用「真正下一月報酬」=====
        next_returns = top["next_month_return"].dropna()

        if len(next_returns) == 0:
            continue

        portfolio_returns.append(
            next_returns.mean() - cost
        )

        # ===== Benchmark 同期報酬 =====
        bench_row = benchmark[
            benchmark["date"] == decision_date
        ]

        if len(bench_row) > 0:
            benchmark_returns.append(
                bench_row["bench_ret"].values[0]
            )
        else:
            benchmark_returns.append(0)

    returns = pd.Series(portfolio_returns)
    bench = pd.Series(benchmark_returns[:len(returns)])

    # ==============================
    # 5️⃣ 績效指標
    # ==============================
    metrics = performance_metrics(returns)

    # ==============================
    # 6️⃣ CAPM 分析（穩定版）
    # ==============================
    aligned = pd.concat([returns, bench], axis=1).dropna()
    aligned.columns = ["ret", "bench"]

    if len(aligned) > 5:
        cov = np.cov(aligned["ret"], aligned["bench"])
        beta = cov[0][1] / np.var(aligned["bench"])
        alpha = aligned["ret"].mean() - beta * aligned["bench"].mean()
        annual_alpha = alpha * 12

        tracking_error = (
            (aligned["ret"] - aligned["bench"]).std()
            * np.sqrt(12)
        )

        information_ratio = (
            (aligned["ret"].mean() - aligned["bench"].mean()) * 12
            / tracking_error
            if tracking_error != 0 else 0
        )
    else:
        beta = alpha = annual_alpha = information_ratio = 0

    print("\n===== 嚴格無穿越回測結果 =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n===== Alpha 分析 =====")
    print(f"Beta: {beta:.4f}")
    print(f"Annual Alpha: {annual_alpha:.4f}")
    print(f"Information Ratio: {information_ratio:.4f}")

    return returns, bench, metrics
def main():
    import sys
    sys.path.insert(0, str(_ROOT))
    sys.path.insert(0, str(_SCRIPT_DIR))
    import train_model_copy

    stock_list_path = str(_ROOT / "stock_list.json")
    print("載入資料中...")
    df = train_model_copy.download_and_prepare_data(stock_list_path=stock_list_path)
    if len(df) == 0:
        print("無資料可回測")
        return
    backtest(df)


if __name__ == "__main__":
    main()