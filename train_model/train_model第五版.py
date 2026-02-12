import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, log_loss
from sklearn.metrics import classification_report
import joblib  # 用來存模型
import json

# from labeling import build_relative_labels
from train_model.labeling import build_relative_labels
from train_model.market_regime import (
# from market_regime import (
    MKT_REGIME_FEATURES,
    compute_market_regime_features,
    merge_regime_into_panel,
)

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
def get_dividend_yield(ticker, price_fallback=None):
    """
    price_fallback: 若 yfinance 當日無報價（休市、下市等），用此價格計算，避免 IndexError。
    若 yfinance API 回傳 None（限流、暫停或該標的無資料），改回傳 0.0 避免程式崩潰。
    """
    t = yf.Ticker(ticker)
    try:
        div = t.dividends
    except (TypeError, KeyError):
        return 0.0
    if div is None or len(div) == 0:
        return 0.0

    last_year_div = div[div.index > (div.index.max() - pd.Timedelta("365D"))].sum()
    try:
        hist = t.history(period="1d")
    except (TypeError, KeyError):
        hist = None
    if hist is None or hist.empty:
        if price_fallback is not None and price_fallback > 0:
            price = float(price_fallback)
        else:
            return 0.0
    else:
        price = float(hist["Close"].iloc[-1])

    return last_year_div / price


# 超額報酬 alpha（vs 大盤）
def get_alpha(stock_df, market="^TWII"):
    mkt = yf.download(market, start=stock_df["date"].min(),
                      end=stock_df["date"].max())

    mkt_ret = mkt["Close"].pct_change().add(1).cumprod().iloc[-1] - 1
    stk_ret = stock_df["close"].pct_change().add(1).cumprod().iloc[-1] - 1
    diff = stk_ret - mkt_ret
    return float(diff.iloc[0]) if hasattr(diff, "iloc") else float(diff)




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

    from dataclasses import dataclass

    def clamp(x, min_val=0.0, max_val=1.0):
        return max(min(x, max_val), min_val)

    def score_cagr(cagr, industry="other"):
        if cagr is None:
            return 0.0
        thresholds = {
            "tech": (0.05, 0.25),
            "finance": (0.02, 0.12),
            "other": (0.03, 0.15)
        }
        low, high = thresholds.get(industry, thresholds["other"])
        return clamp((cagr - low) / (high - low))

    def score_relative_performance(alpha, industry="other"):
        if alpha is None:
            return 0.5
        ranges = {
            "tech": (-0.25, 0.25),
            "finance": (-0.15, 0.15),
            "other": (-0.2, 0.2)
        }
        low, high = ranges.get(industry, ranges["other"])
        return clamp((alpha - low) / (high - low))

    def score_dividend_yield(div_yield, industry="other"):
        if div_yield is None:
            return 0.0
        limits = {
            "tech": (0.0, 0.03),
            "finance": (0.0, 0.06),
            "other": (0.0, 0.05)
        }
        low, high = limits.get(industry, limits["other"])
        return clamp((div_yield - low) / (high - low))

    @dataclass
    class FundamentalWeights:
        growth: float = 0.50
        relative: float = 0.25
        dividend: float = 0.25

    def fundamental_score(
        revenue_cagr,
        relative_alpha,
        dividend_yield,
        weights=FundamentalWeights(),
        industry="other"
    ):
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

    horizon = 30

    for ticker, industry in STOCKS.items():
        print("下載:", ticker)

        single = yf.download(ticker, start="2020-01-01", end="2026-01-01")

        if single is None or len(single) == 0:
            print("跳過空資料:", ticker)
            continue

        if isinstance(single.columns, pd.MultiIndex):
            single.columns = single.columns.get_level_values(0)

        single = single.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        }).reset_index()
        single["date"] = pd.to_datetime(single["Date"])
        single = single[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True).dropna()

        for col in ["open", "high", "low", "close", "volume"]:
            lo, hi = single[col].quantile(0.01), single[col].quantile(0.99)
            single[col] = single[col].clip(lo, hi).astype(float)

        single["ticker"] = ticker
        single["industry"] = industry

        # 技術指標：僅用「此檔」的 close，不跨股票
        single["ma5"] = single["close"].rolling(5).mean()
        single["ma20"] = single["close"].rolling(20).mean()
        single["ma60"] = single["close"].rolling(60).mean()
        single["ma120"] = single["close"].rolling(120).mean()
        single["ma240"] = single["close"].rolling(240).mean()
        single["return_1"] = single["close"].pct_change(1)
        single["return_5"] = single["close"].pct_change(5)
        single["rsi_120"] = compute_rsi(single["close"], 120)
        single["rsi_240"] = compute_rsi(single["close"], 240)
        single["rsi_420"] = compute_rsi(single["close"], 420)
        single["ema120"] = single["close"].ewm(span=120, adjust=False).mean()
        single["ema240"] = single["close"].ewm(span=240, adjust=False).mean()
        single["ema420"] = single["close"].ewm(span=420, adjust=False).mean()
        single["ema200"] = single["close"].ewm(span=200, adjust=False).mean()

        # 基本面：此檔專用（若當日無報價則用該檔最近收盤價當 fallback）
        revenue_cagr = get_revenue_cagr(ticker)
        div_yield = get_dividend_yield(ticker, price_fallback=single["close"].iloc[-1])
        alpha = get_alpha(single)
        fund_dict = fundamental_score(revenue_cagr, alpha, div_yield, industry=industry)
        single["fund_score"] = fund_dict["total_score"]

        # 未來報酬（同一檔內計算）；標籤改在合併後以「相對排名」建構（見 build_relative_labels）
        single["future_return_30"] = single["close"].shift(-horizon) / single["close"] - 1

        single = single.dropna(subset=["ma240", "rsi_420", "ema200", "future_return_30"]).reset_index(drop=True)
        if len(single) == 0:
            continue
        all_df.append(single)

    print("成功股票數:", len(all_df))
    df = pd.concat(all_df, ignore_index=True)
    print("合併後樣本數:", len(df))

    # 相對標籤：依當日橫斷面排名，前 20% → 1、後 20% → 0、中間 → 2
    df = build_relative_labels(
        df,
        horizon_col="future_return_30",
        top_pct=0.2,
        bottom_pct=0.2,
        min_stocks_per_date=5,
    )
    print("相對標籤後樣本數（剔除當日不足 5 檔的日期）:", len(df))
    print(df["label"].value_counts(normalize=True))

    # 大盤 regime 特徵：^TWII 趨勢、波動、回撤，以 date 合併進 panel
    market_symbol = "^TWII"
    date_min, date_max = df["date"].min(), df["date"].max()
    mkt = yf.download(
        market_symbol,
        start=(date_min - pd.Timedelta(days=100)).strftime("%Y-%m-%d"),
        end=date_max.strftime("%Y-%m-%d"),
    )
    if mkt is None or mkt.empty:
        raise RuntimeError("無法下載大盤資料，請檢查網路或代碼 " + market_symbol)
    if isinstance(mkt.columns, pd.MultiIndex):
        mkt.columns = mkt.columns.get_level_values(0)
    mkt = mkt.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}).reset_index()
    mkt["date"] = pd.to_datetime(mkt["Date"])
    regime_df = compute_market_regime_features(mkt)
    df = merge_regime_into_panel(df, regime_df)
    df = df.dropna(subset=MKT_REGIME_FEATURES).reset_index(drop=True)
    print("合併大盤 regime 後樣本數:", len(df))

    # 存檔（多檔合併後）
    df.to_csv("model_input_multi.csv", index=False)
    print("已產生 model_input_multi.csv")

    # 切訓練集與測試集
    train = df[df["date"] < "2025-01-01"]
    test  = df[df["date"] >= "2025-01-01"]

    # 特徵：個股技術 + 基本面 + 大盤 regime
    features = [
        "open", "high", "low", "close", "volume",
        "ma5", "ma20", "ma60", "ma120", "ma240",
        "return_1", "return_5",
        "rsi_120", "rsi_240", "rsi_420",
        "ema120", "ema240", "ema420", "ema200",
        "fund_score",
        *MKT_REGIME_FEATURES,
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

    # 存模型（多檔訓練，適用全部股票；請將 MODEL_PATH 指向此檔）
    joblib.dump(model, "rf_model_multi.pkl")
    print("模型已存檔：rf_model_multi.pkl（多檔通用，請設定 MODEL_PATH=./rf_model_multi.pkl）")

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
    print("平均報酬:", trade_days["future_return_30"].mean())

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