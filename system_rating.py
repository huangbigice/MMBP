import pandas as pd
import numpy as np
import yfinance as yf

# ===============================
# 系統權重（之後優化只調這裡）
# ===============================
W_PROBA = 0.50 # 模型機率
W_FUND  = 0.25 # 基本面
W_TECH  = 0.25 # 技術面

# BUY_THRESHOLD = 0.65
# HOLD_THRESHOLD = 0.50 

THRESH_BUY  = 0.60
THRESH_HOLD = 0.50

def compute_tech_score(row):
    score = 0

    # 均線多頭排列
    if row["close"] > row["ma20"] > row["ma60"] > row["ma120"]:
        score += 2
    elif row["close"] > row["ma20"]:
        score += 1
    else:
        score -= 1

    # RSI
    if row["rsi_120"] < 35:
        score += 1
    elif row["rsi_120"] > 75:
        score -= 1

    # EMA 趨勢
    if row["ema120"] > row["ema240"]:
        score += 1
    else:
        score -= 1

    # 正規化到 0~1
    # return (score + 4) / 8
    return (score + 3) / 7   # 嚴格 0~1



# def clamp(x, min_val=0.0, max_val=1.0):
#     return max(min(x, max_val), min_val)

def clamp(x, min_val=0.0, max_val=1.0):
    if x is None or pd.isna(x):
        return 0.5   # 給「中性分數」
    return max(min(x, max_val), min_val)



def fundamental_score(
    revenue_growth,     # 年營收成長率（yfinance）
    earnings_growth,    # EPS 成長率（yfinance）
    dividend_yield,     # 已修正為比例 (0.02 = 2%)
    industry="tech"
):
    """
    回傳：
    {
        total_score,
        growth_score,
        earnings_score,
        dividend_score
    }
    """

    # ========= 產業合理區間 =========
    growth_range = {
        "tech":     (0.05, 0.25),
        "finance":  (0.02, 0.15),
        "other":    (0.03, 0.20)
    }
    low_g, high_g = growth_range.get(industry, growth_range["other"])

    # ========= 各分項（yfinance 有時缺欄位，None 以中性分數處理）=========
    if revenue_growth is None:
        growth_score = 0.5
    else:
        growth_score = clamp((revenue_growth - low_g) / (high_g - low_g))

    if earnings_growth is None:
        earnings_score = 0.5
    else:
        earnings_score = clamp((earnings_growth - low_g) / (high_g - low_g))

    # 股息不鼓勵過高；缺資料時視為 0
    if dividend_yield is None:
        dividend_score = 0.0
    else:
        dividend_score = clamp(dividend_yield / 0.06)

    # ========= 權重（可調） =========
    total = (
        0.50 * growth_score +
        0.35 * earnings_score +
        0.20 * dividend_score
    )

    return {
        "total_score": round(total, 3),
        "growth_score": round(growth_score, 3),
        "earnings_score": round(earnings_score, 3),
        "dividend_score": round(dividend_score, 3),
    }

# def get_latest_fund_score(symbol: str, industry="tech"):
#     """
#     回傳 0~1 的基本面分數
#     使用 yfinance info（低頻、穩定）
#     """
#     try:
#         ticker = yf.Ticker(symbol)
#         info = ticker.info

#         revenue_growth = info.get("revenueGrowth")      # e.g. 0.15
#         earnings_growth = info.get("earningsGrowth")    # e.g. 0.12
#         dividend_yield = info.get("dividendYield")      # e.g. 0.025

#         # ---- 成長分數（對應 revenue_cagr）----
#         if revenue_growth is None:
#             growth_score = 0.5
#         else:
#             growth_score = clamp((revenue_growth - 0.03) / (0.20 - 0.03))

#         # ---- 相對表現（簡化版 alpha）----
#         if earnings_growth is None:
#             relative_score = 0.5
#         else:
#             relative_score = clamp((earnings_growth - 0.02) / (0.15 - 0.02))

#         # ---- 股息 ----
#         if dividend_yield is None:
#             dividend_score = 0.0
#         else:
#             dividend_score = clamp(dividend_yield / 0.05)

#         # ---- 加權 ----
#         fund_score = (
#             0.50 * growth_score +
#             0.25 * relative_score +
#             0.25 * dividend_score
#         )
#         print(f"""
#         [{symbol} 基本面拆解]
#         revenueGrowth:   {revenue_growth}
#         earningsGrowth: {earnings_growth}
#         dividendYield:  {dividend_yield}

#         scores:
#             growth_score   = {growth_score:.3f}
#             relative_score = {relative_score:.3f}
#             dividend_score = {dividend_score:.3f}
#         """)

#         return round(fund_score, 3)

#     except Exception as e:
#         print(f"[WARN] 基本面資料取得失敗 ({symbol}): {e}")
#         return 0.5

def get_latest_fund_score(symbol, industry="tech"):
    t = yf.Ticker(symbol)
    info = t.info or {}

    revenue_growth = info.get("revenueGrowth")
    earnings_growth = info.get("earningsGrowth")

    dividend = info.get("trailingAnnualDividendRate") or 0
    price = info.get("regularMarketPrice") or np.nan

    if price is not None and not (isinstance(price, float) and np.isnan(price)) and price > 0:
        dividend_yield = (dividend or 0) / price
    else:
        dividend_yield = 0.0

    dividend_yield = min(dividend_yield, 0.08)

    scores = fundamental_score(
        revenue_growth,
        earnings_growth,
        dividend_yield,
        industry=industry
    )

    # 金融直覺封頂
    scores["total_score"] = min(scores["total_score"], 0.85)

    return scores


# def system_rating(row, proba):
#     tech = compute_tech_score(row)

#     system_score = (
#         W_PROBA * max(proba) +
#         W_FUND  * row["fund_score"] +
#         W_TECH  * tech
#     )

#     if system_score >= THRESH_BUY:
#         rec = "長期持有"
#     elif system_score >= THRESH_HOLD:
#         rec = "觀望"
#     else:
#         rec = "不建議持有"

#     return {
#         "proba": round(max(proba),3),
#         "fund": round(row["fund_score"],3),
#         "tech": round(tech,3),
#         "system_score": round(system_score,3),
#         "recommendation": rec
#     }

def system_rating(df: pd.DataFrame):
    df = df.copy()

    # 技術分數
    df["tech_score"] = df.apply(compute_tech_score, axis=1)

    # 系統總分
    df["system_score"] = (
        W_PROBA * df["proba_buy"] +
        W_FUND  * df["fund_score"] +
        W_TECH  * df["tech_score"]
    )

# ==================== 最終決策 ==================== 
    
    #第一版決策
    
    # df["recommendation"] = "不建議持有"
    # df.loc[df["system_score"] >= HOLD_THRESHOLD, "recommendation"] = "觀望"
    # df.loc[df["system_score"] >= BUY_THRESHOLD,  "recommendation"] = "建議持有"

    #第二版決策
    df["recommendation"] = "不建議持有"
    df.loc[df["system_score"] >= THRESH_HOLD, "recommendation"] = "觀望"
    df.loc[df["system_score"] >= THRESH_BUY,  "recommendation"] = "長期持有"


    return df
