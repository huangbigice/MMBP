import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, log_loss
from sklearn.metrics import classification_report
import joblib  # 用來存模型
import json

def compute_rsi(series, window):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

#===================基本面資料======================
# 財報營收 CAGR
def get_revenue_cagr(ticker, years=6):
    t = yf.Ticker(ticker)
    fin = t.financials.T  # index = 年份

    if "Total Revenue" not in fin.columns:
        return None

    rev = fin["Total Revenue"].dropna().sort_index()
    if len(rev) < years:
        return None

    start = rev.iloc[-years]
    end = rev.iloc[-1]
    cagr = (end / start) ** (1/years) - 1
    return cagr

# 股息殖利率
def get_dividend_yield(ticker):
    t = yf.Ticker(ticker)
    div = t.dividends

    if len(div) == 0:
        return 0.0

    last_year_div = div[div.index > (div.index.max() - pd.Timedelta("365D"))].sum()
    price = t.history(period="1d")["Close"].iloc[-1]

    return last_year_div / price


# 超額報酬 alpha（vs 大盤）
def get_alpha(stock_df, market="^TWII"):
    mkt = yf.download(market, start=stock_df["date"].min(),
                      end=stock_df["date"].max())

    mkt_ret = mkt["Close"].pct_change().add(1).cumprod().iloc[-1] - 1
    stk_ret = stock_df["close"].pct_change().add(1).cumprod().iloc[-1] - 1

    return float(stk_ret - mkt_ret)




def main():
    # 1. 下載過去三年的歷史資料
    # with open("stock_list.json") as f:
    #     STOCKS = json.load(f)

    # all_df = []
    # for ticker, industry in STOCKS.items():
    #     df = yf.download(ticker, start="2020-01-01", end="2026-01-01")
    # if isinstance(df.columns, pd.MultiIndex):
    #     df.columns = df.columns.get_level_values(0)

    #     df = df.rename(columns={
    #         "Open": "open",
    #         "High": "high",
    #         "Low": "low",
    #         "Close": "close",
    #         "Volume": "volume"
    #     })

    #     df = df.reset_index()
    #     df["date"] = pd.to_datetime(df["Date"])
    #     df = df[["date", "open", "high", "low", "close", "volume"]]
    #     df = df.sort_values("date").reset_index(drop=True)

    #     df = df.dropna()

    #     # 去極值
    #     features = ["open", "high", "low", "close", "volume"]
    #     for col in features:
    #         low = df[col].quantile(0.01)
    #         high = df[col].quantile(0.99)
    #         df[col] = df[col].clip(low, high)

    # df = pd.concat(all_df)

    with open("stock_list.json") as f:
        config = json.load(f)

    STOCKS = config["stocks"]

    all_df = []

    for ticker, industry in STOCKS.items():
        print("下載:", ticker)

        df = yf.download(ticker, start="2020-01-01", end="2026-01-01")

        if df is None or len(df) == 0:
            print("跳過空資料:", ticker)
            continue

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
        for col in ["open", "high", "low", "close", "volume"]:
            low = df[col].quantile(0.01)
            high = df[col].quantile(0.99)
            df[col] = df[col].clip(low, high)

        df["ticker"] = ticker
        df["industry"] = industry

        all_df.append(df)

    print("成功股票數:", len(all_df))
    df = pd.concat(all_df).reset_index(drop=True)
    print("總樣本數:", len(df))


    from dataclasses import dataclass

    # ---------- 工具函數 ----------
    def clamp(x, min_val=0.0, max_val=1.0):
        return max(min(x, max_val), min_val)


    # ---------- 各指標轉分數（可依產業調整） ----------
    def score_cagr(cagr, industry="other"):
        """
        CAGR 評分
        - industry: "tech", "finance", "other"
        - 不同產業有不同的低/高門檻
        """
        if cagr is None:
            return 0.0

        thresholds = {
            "tech": (0.05, 0.25),     # 科技股成長門檻高
            "finance": (0.02, 0.12),  # 金融股成長門檻低
            "other": (0.03, 0.15)     # 其他產業中間
        }
        low, high = thresholds.get(industry, thresholds["other"])
        return clamp((cagr - low) / (high - low))


    def score_relative_performance(alpha, industry="other"):
        """
        相對市場表現（超額報酬）
        - alpha = 股票報酬 - 大盤報酬
        - 不同產業可調 alpha 範圍，但這裡保持原始 -20~20% 為主
        """
        if alpha is None:
            return 0.5
        # 可選擇依產業調整上限，例如金融股波動小，科技股波動大
        ranges = {
            "tech": (-0.25, 0.25),
            "finance": (-0.15, 0.15),
            "other": (-0.2, 0.2)
        }
        low, high = ranges.get(industry, ranges["other"])
        return clamp((alpha - low) / (high - low))


    def score_dividend_yield(div_yield, industry="other"):
        """
        股息殖利率評分
        - 不鼓勵過高殖利率（封頂）
        - 不同行業期望不同
        """
        if div_yield is None:
            return 0.0

        limits = {
            "tech": (0.0, 0.03),      # 科技股通常低股息
            "finance": (0.0, 0.06),   # 金融股較高股息
            "other": (0.0, 0.05)      # 其他產業中間
        }
        low, high = limits.get(industry, limits["other"])
        return clamp((div_yield - low) / (high - low))


    # ---------- 權重 ----------
    @dataclass
    class FundamentalWeights:
        growth: float = 0.45
        relative: float = 0.30
        dividend: float = 0.25


    # ---------- 總評分 ----------
    def fundamental_score(
        revenue_cagr,
        relative_alpha,
        dividend_yield,
        weights=FundamentalWeights(),
        industry="other"
    ):
        """
        回傳：
        - total_score (0~1)
        - breakdown dict
        """
        growth_score = score_cagr(revenue_cagr, industry)
        relative_score = score_relative_performance(relative_alpha, industry)
        dividend_score = score_dividend_yield(dividend_yield, industry)

        total = (
            growth_score * weights.growth +
            relative_score * weights.relative +
            dividend_score * weights.dividend
        )

        return {
            "total_score": round(total, 3),
            "growth_score": round(growth_score, 3),
            "relative_score": round(relative_score, 3),
            "dividend_score": round(dividend_score, 3)
        }



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

    df["ema120"] = df["close"].ewm(span=120, adjust=False).mean()
    df["ema240"] = df["close"].ewm(span=240, adjust=False).mean()
    df["ema420"] = df["close"].ewm(span=420, adjust=False).mean()
    # df["ema200"] = float(df["close"].rolling(window=200).mean()).iloc[-1]
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()


    df = df.dropna().reset_index(drop=True)

    # ====== 基本面資料（手動版，之後可接 API） ======
    # fundamental_df = pd.DataFrame([
    #     {"date": "2023-08-15", "revenue_cagr": 0.12, "alpha": 0.08, "div_yield": 0.025},
    #     {"date": "2024-03-15", "revenue_cagr": 0.15, "alpha": 0.10, "div_yield": 0.022},
    #     {"date": "2024-11-15", "revenue_cagr": 0.18, "alpha": 0.14, "div_yield": 0.020},
    # ])
    # fundamental_df["date"] = pd.to_datetime(fundamental_df["date"])

    # # 算基本面分數
    # fundamental_df["fund_score"] = fundamental_df.apply(
    #     lambda row: fundamental_score(
    #         row["revenue_cagr"],
    #         row["alpha"],
    #         row["div_yield"],
    #         industry="tech"
    #     )["total_score"],
    #     axis=1
    # )

    # # 對齊到每日股價
    # df = df.merge(fundamental_df[["date", "fund_score"]],
    #             on="date",
    #             how="left")
    # df["fund_score"] = df["fund_score"].ffill()
    
    
    # ====== 基本面資料（真實 yfinance 版） ======
    revenue_cagr = get_revenue_cagr(ticker)
    div_yield = get_dividend_yield(ticker)
    alpha = get_alpha(df)

    fund_score_dict = fundamental_score(
        revenue_cagr,
        alpha,
        div_yield,
        industry=industry
    )

    fundamental_df = pd.DataFrame([{
        "date": df["date"].max(),   # 最新一筆財報視為目前有效
        "fund_score": fund_score_dict["total_score"]
    }])

    # 對齊到每日股價
    df = df.merge(fundamental_df, on="date", how="left")
    df["fund_score"] = df["fund_score"].ffill()



    # 標籤：未來 60 天報酬
    horizon = 30
    df["future_return_60"] = df["close"].shift(-horizon) / df["close"] - 1


    # df["label"] = 2  # 預設觀望
    df.loc[(df["future_return_60"] > 0.05) & (df["future_return_60"] <= 0.10), "label"] = 2 # 建議觀望
    df.loc[df["future_return_60"] > 0.10, "label"] = 1   # 長期持有
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

    # features = df.columns.drop(["date", "label", "future_return_60"])
    features = [
        "open","high","low","close","volume",
        "ma5","ma20","ma60","ma120","ma240",
        "return_1","return_5",
        "rsi_120","rsi_240","rsi_420",
        "ema120","ema240","ema420","ema200",
        "fund_score"   # 只用當下可取得的資訊
    ]

    

    X_train = train[features]
    y_train = train["label"]

    X_test = test[features]
    y_test = test["label"]

    # 建立模型
    model = RandomForestClassifier(
        n_estimators=400, 
        max_depth=10, 
        random_state=42,
        class_weight="balanced"
    )
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
    # test["proba_buy"] = proba[:, 1]   # 類別1的機率（長期持有）
    # test["proba_avoid"] = proba[:, 0]

    test.loc[:, "proba_buy"] = proba[:, 1]
    test.loc[:, "proba_avoid"] = proba[:, 0]

    # 篩選信號
    # test["signal"] = (test["proba_buy"] > 0.6).astype(int)

    # 訊號策略
    test = test.copy()
    # test["proba"] = proba
    # threshold = test["proba"].quantile(0.85)
    # test["signal"] = (proba >= threshold).astype(int)
    # test["signal"] = (test["proba_buy"] > 0.7).astype(int)
    test["signal"] = ((test["proba_buy"] > 0.6) & (test["fund_score"] > 0.6)).astype(int)
    # test["position"] = test["signal"].replace(0, np.nan).ffill().fillna(0)
    test["allow_trade"] = (test["fund_score"] > 0.6).astype(int)
    # test["position"] = test["signal"] * test["allow_trade"]



    # test["position"] = 0
    # holding = 0
    # for i in range(len(test)):
    #     if holding > 0:
    #         test.loc[test.index[i], "position"] = 1
    #         holding -= 1
    #     # elif test.loc[test.index[i], "signal"] == 1:
    #     #     test.loc[test.index[i], "position"] = 1
    #     #     holding = 60
    #     elif (test.loc[test.index[i], "signal"] == 1) and (test.loc[test.index[i], "fund_score"] > 0.6):
    #         test.loc[test.index[i], "position"] = 1
    #         # holding = 20
    #         holding = int(20 * test.loc[test.index[i], "proba_buy"])

    # test["position"] = 0.0
    # holding = 0
    # current_weight = 0

    # for i in range(len(test)):
    #     if holding > 0:
    #         test.loc[test.index[i], "position"] = current_weight
    #         holding -= 1

    #     elif (test.loc[test.index[i], "signal"] == 1):
    #         current_weight = test.loc[test.index[i], "proba_buy"]
    #         test.loc[test.index[i], "position"] = current_weight
    #         holding = horizon

    MAX_POSITION = 0.5  # 最多只用 50% 資金
    test["position"] = test["proba_buy"] * test["allow_trade"]
    test["position"] = test["position"].clip(upper=MAX_POSITION)

    # 單日虧損超過 threshold 時，position 直接歸零
    LOSS_THRESHOLD = -0.05  # 單日跌超過 5%
    test["strategy_return"] = test["position"].shift(1) * test["return_1"]
    test.loc[test["return_1"] < LOSS_THRESHOLD, "strategy_return"] = 0
    test["cum_return"] = (1 + test["strategy_return"]).cumprod()

    test["drawdown"] = test["cum_return"] / test["cum_return"].cummax() - 1
    
    test["position"] = (test["position"].ewm(span=3, adjust=False).mean())
    # test["position"] = test["position"].ewm(span=5, adjust=False).mean()


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


if __name__ == "__main__":
    # 這裡才放：
    # 訓練模型
    # 存模型
    # 印報表
    # 回測
    main()