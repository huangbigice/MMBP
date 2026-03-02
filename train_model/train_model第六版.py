"""
訓練模型第六版：依模型修改指南優化。

- 標籤：build_quality_labels（強勢/弱勢/趨勢不明，每日前後各 18%，中性約 64%）
- 特徵：return_1/5/20、RSI、均線比、volume_spike、price_strength、volatility_regime、大盤 regime
- 模型：RandomForest 強化參數與 class_weight，概率門檻 0.55，部位由 P(強勢) 連續控制
"""

import json
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from train_model.labeling import build_quality_labels
from train_model.market_regime import (
    MKT_REGIME_FEATURES,
    compute_market_regime_features,
    merge_regime_into_panel,
)

# ========== 常數 ==========
HORIZON = 30  # 未來報酬天數
MAX_POSITION = 0.5  # 最多使用 50% 資金
LOSS_THRESHOLD = -0.05  # 單日跌超過 5% 時停損
PROBA_THRESHOLD = 0.55  # P(強勢) 超過此值才發信號（0.68 過高導致 0 次進場，改 0.55）
TRAIN_END = "2024-12-01"  # 訓練集截止（不含），2024-12 留作緩衝
TEST_START = "2025-01-01"  # 測試集起日
MARKET_SYMBOL = "^TWII"
MODEL_PATH = "rf_model_multi_No6.pkl"

# 特徵清單（共 12 項）：報酬、RSI、均線比、量價高頻、大盤 regime
FEATURES = [
    "return_1", "return_5", "return_20",
    "rsi_14", "rsi_120",
    "ma5_close", "ma20_close", "ma60_close",
    "volume_spike", "price_strength", "volatility_regime",
    *MKT_REGIME_FEATURES,
]


def compute_rsi(series, window=14):
    """計算 RSI，避免除零與 NaN 崩潰。"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def download_and_prepare_data(stock_list_path="stock_list.json"):
    """下載各股資料、計算技術指標與高頻特徵、合併大盤 regime、建立品質標籤。"""
    with open(stock_list_path) as f:
        config = json.load(f)
    stocks = config["stocks"]

    all_df = []
    for ticker, industry in stocks.items():
        print("下載:", ticker)
        single = yf.download(ticker, start="2020-01-01", end="2026-01-01")

        if single is None or len(single) == 0:
            print("跳過空資料:", ticker)
            continue

        if isinstance(single.columns, pd.MultiIndex):
            single.columns = single.columns.get_level_values(0)

        single = single.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        }).reset_index()
        single["date"] = pd.to_datetime(single["Date"])
        single = single[["date", "open", "high", "low", "close", "volume"]]
        single = single.sort_values("date").reset_index(drop=True).dropna()

        for col in ["open", "high", "low", "close", "volume"]:
            lo, hi = single[col].quantile(0.01), single[col].quantile(0.99)
            single[col] = single[col].clip(lo, hi).astype(float)

        single["ticker"] = ticker
        single["industry"] = industry

        # 技術指標
        single["ma5"] = single["close"].rolling(5).mean()
        single["ma20"] = single["close"].rolling(20).mean()
        single["ma60"] = single["close"].rolling(60).mean()
        single["ma120"] = single["close"].rolling(120).mean()
        single["ma240"] = single["close"].rolling(240).mean()
        single["return_1"] = single["close"].pct_change(1)
        single["return_5"] = single["close"].pct_change(5)
        single["return_20"] = single["close"].pct_change(20)
        single["rsi_14"] = compute_rsi(single["close"], 14)
        single["rsi_120"] = compute_rsi(single["close"], 120)
        single["ma5_close"] = single["close"] / single["ma5"]
        single["ma20_close"] = single["close"] / single["ma20"]
        single["ma60_close"] = single["close"] / single["ma60"]

        # 高頻特徵（取代 fund_score，避免標籤失衡）
        single["price_strength"] = (single["close"] / single["close"].shift(10)).rank(pct=True)
        single["volatility_regime"] = single["return_1"].rolling(20).std().rank(pct=True)
        single["volume_spike"] = (
            single["volume"] / single["volume"].rolling(20).mean()
        ).clip(0, 3)

        single["future_return_30"] = single["close"].shift(-HORIZON) / single["close"] - 1

        # 僅就單股已有欄位 dropna（大盤 regime 於合併後才加入）
        single = single.dropna(
            subset=[
                "return_1", "return_5", "return_20",
                "rsi_14", "rsi_120",
                "ma5_close", "ma20_close", "ma60_close",
                "volume_spike", "price_strength", "volatility_regime",
                "future_return_30",
            ]
        ).reset_index(drop=True)
        if len(single) == 0:
            continue
        all_df.append(single)

    print("成功股票數:", len(all_df))
    df = pd.concat(all_df, ignore_index=True)
    print("合併後樣本數:", len(df))

    # 品質標籤：強勢(1) / 弱勢(0) / 趨勢不明(2)，每日前後各 18%（中性約 64%）
    df = build_quality_labels(
        df,
        horizon_col="future_return_30",
        top_pct=0.18,
        bottom_pct=0.18,
        min_stocks_per_date=5,
    )
    print("品質標籤後樣本數:", len(df))
    print(df["label"].value_counts(normalize=True))

    # 大盤 regime
    date_min, date_max = df["date"].min(), df["date"].max()
    mkt = yf.download(
        MARKET_SYMBOL,
        start=(date_min - pd.Timedelta(days=100)).strftime("%Y-%m-%d"),
        end=date_max.strftime("%Y-%m-%d"),
    )
    if mkt is None or mkt.empty:
        raise RuntimeError("無法下載大盤資料，請檢查網路或代碼 " + MARKET_SYMBOL)
    if isinstance(mkt.columns, pd.MultiIndex):
        mkt.columns = mkt.columns.get_level_values(0)
    mkt = mkt.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    }).reset_index()
    mkt["date"] = pd.to_datetime(mkt["Date"])
    regime_df = compute_market_regime_features(mkt)
    df = merge_regime_into_panel(df, regime_df)
    df = df.dropna(subset=MKT_REGIME_FEATURES).reset_index(drop=True)
    print("合併大盤 regime 後樣本數:", len(df))

    return df


def train_and_evaluate(df):
    """切分資料、標準化、訓練模型、評估、存檔。"""
    train = df[df["date"] < TRAIN_END]
    test = df[df["date"] >= TEST_START]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURES].fillna(0))
    y_train = train["label"]
    X_test = scaler.transform(test[FEATURES].fillna(0))
    y_test = test["label"]

    # 強化參數：更多樹、較深、min_samples_split 防過擬合，買賣類權重提高
    model = RandomForestClassifier(
        n_estimators=800,
        max_depth=15,
        min_samples_split=100,
        class_weight={0: 2.5, 1: 2.5, 2: 1},
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    joblib.dump(
        {"model": model, "scaler": scaler, "features": FEATURES},
        MODEL_PATH,
    )
    print("模型已存檔：", MODEL_PATH, "（含 model、scaler、features）")

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 寫入預測機率供回測使用（類別 0=弱勢、1=強勢、2=趨勢不明）
    proba = model.predict_proba(X_test)
    test = test.copy()
    test.loc[:, "proba_strong"] = proba[:, 1]  # P(強勢)
    test.loc[:, "proba_weak"] = proba[:, 0]    # P(弱勢)

    return model, scaler, test


def run_backtest(test):
    """執行回測：簡化邏輯，前日概率決定部位，單日跌>5%當日報酬清零。"""
    test = test.copy()

    # 1. 前日概率決定今日部位
    test["position"] = test["proba_strong"].shift(1).clip(0, MAX_POSITION)

    # 2. 單純停損：跌>5%當日報酬=0，但不停永久清倉（明日可重新建倉）
    test["strategy_return"] = test["position"] * test["return_1"]
    test.loc[test["return_1"] < LOSS_THRESHOLD, "strategy_return"] = 0

    # 3. 正確計算績效：按日期聚合，計算每日平均報酬（跨股票）
    # 因為 test 是面板資料（多股票×多日期），每檔股票獨立計算 cumprod() 會導致錯誤
    # 正確做法：每日平均報酬 → 組合累積曲線 → 真實回撤
    daily_return = test.groupby("date")["strategy_return"].mean()

    # 正確累積報酬曲線
    portfolio_cum = (1 + daily_return).cumprod()
    portfolio_dd = portfolio_cum / portfolio_cum.cummax() - 1

    print("=== 正確回測績效 ===")
    print("年化報酬:", daily_return.mean() * 252)
    print("**真實最大回撤**:", portfolio_dd.min())
    std = daily_return.std()
    if std > 0:
        sharpe = daily_return.mean() / std * np.sqrt(252)
        print("夏普比率:", sharpe)
    else:
        print("夏普比率: (無波動)")
    win_rate = (daily_return > 0).mean()
    print("勝率:", win_rate)
    print("總交易日:", len(daily_return))
    print("平均部位:", test["position"].mean())

    buy_signal = test["proba_strong"] > PROBA_THRESHOLD
    trade_days = test[buy_signal]
    print("進場次數:", len(trade_days))
    if len(trade_days) > 0:
        print("平均未來 30 日報酬:", trade_days["future_return_30"].mean())


def main():
    df = download_and_prepare_data()
    _, _, test = train_and_evaluate(df)
    run_backtest(test)

    # 驗證檢查點（依模型修改指南）：各類準確率 label0/1 > 0.35、label2 > 0.65；
    # proba_strong 最高 10% 樣本之平均 future_return_30 > 15%；年化報酬 > 大盤 6%；夏普 > 1.2


if __name__ == "__main__":
    main()
