"""
② validate_model.py（依模型修改指南）

負責：load model、load scaler、只用未來資料、回測、統計 Sharpe / MDD / Spread；
      產出權益曲線圖、回撤曲線圖與驗證結果 TXT。
"""

import sys
from pathlib import Path

# 確保可從同目錄載入 train
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train import (
    FEATURES,
    LOSS_THRESHOLD,
    MAX_POSITION,
    MODEL_PATH,
    PROBA_THRESHOLD,
    TEST_START,
    download_and_prepare_data,
)

ROLL_WINDOW = 60        # 滾動觀察 60 日
SHARPE_CUTOFF = 0.5     # 低於此值 → 降低槓桿
SHARPE_STOP = 0         # 低於 0 → 暫停
RISK_REDUCTION = 0.5    # 降低為原本 50% 部位


_SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_EQUITY_PNG = _SCRIPT_DIR / "validation_equity_curve.png"
OUTPUT_DRAWDOWN_PNG = _SCRIPT_DIR / "validation_drawdown_curve.png"
OUTPUT_EQUITY_TOP10_PNG = _SCRIPT_DIR / "validation_equity_curve_top10.png"
OUTPUT_DRAWDOWN_TOP10_PNG = _SCRIPT_DIR / "validation_drawdown_curve_top10.png"
OUTPUT_RESULT_TXT = _SCRIPT_DIR / "validation_result.txt"
OUTPUT_MONTHLY_CSV = _SCRIPT_DIR / "validation_monthly_table.csv"
OUTPUT_QUARTERLY_CSV = _SCRIPT_DIR / "validation_quarterly_table.csv"

# Long-Short 與選股驗證用
TOP_PCT = 0.10   # 強勢前 10%
BOTTOM_PCT = 0.10  # 弱勢後 10%


def load_model_artifacts():
    """載入模型、scaler、特徵清單。"""
    data = joblib.load(MODEL_PATH)
    return data["model"], data["scaler"], data["features"]

def apply_risk_control(daily_spread):
    rolling_mean = daily_spread.rolling(ROLL_WINDOW).mean()
    rolling_std = daily_spread.rolling(ROLL_WINDOW).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)

    exposure = pd.Series(1.0, index=daily_spread.index)

    # Sharpe < 0 → 全停
    exposure[rolling_sharpe < SHARPE_STOP] = 0

    # 0 < Sharpe < cutoff → 降低部位
    mask = (rolling_sharpe >= SHARPE_STOP) & (rolling_sharpe < SHARPE_CUTOFF)
    exposure[mask] = RISK_REDUCTION

    adjusted_spread = daily_spread * exposure
    return adjusted_spread, exposure


def run_backtest(test):
    """
    只用未來資料回測：前日概率決定部位，單日跌>5% 當日報酬清零。
    回傳 daily_return、portfolio_cum、portfolio_dd、以及帶有 position 的 test 供統計。
    """
    test = test.copy()
    test["position"] = test["proba_strong"].shift(1).clip(0, MAX_POSITION)
    test["strategy_return"] = test["position"] * test["return_1"]
    test.loc[test["return_1"] < LOSS_THRESHOLD, "strategy_return"] = 0

    daily_return = test.groupby("date")["strategy_return"].mean()
    portfolio_cum = (1 + daily_return).cumprod()
    portfolio_dd = portfolio_cum / portfolio_cum.cummax() - 1
    return daily_return, portfolio_cum, portfolio_dd, test


def run_long_short_backtest(test):
    """
    每日以「前一日」proba_strong 排名，做多 top 10%、做空 bottom 10%。
    Spread = 當日 long 平均報酬 - short 平均報酬（真實 long-short alpha）。
    回傳 daily_spread、權益與回撤序列。
    """
    t = test.copy()
    t["proba_prev"] = t.groupby("ticker")["proba_strong"].shift(1)
    t = t.dropna(subset=["proba_prev"])
    if len(t) == 0:
        return None, None, None

    t["rank_pct"] = t.groupby("date")["proba_prev"].rank(pct=True)
    long_ret = t[t["rank_pct"] >= (1 - TOP_PCT)].groupby("date")["return_1"].mean()
    short_ret = t[t["rank_pct"] <= BOTTOM_PCT].groupby("date")["return_1"].mean()
    common_idx = long_ret.index.intersection(short_ret.index).sort_values()
    if len(common_idx) == 0:
        return None, None, None
    daily_spread = long_ret.loc[common_idx] - short_ret.loc[common_idx]
    
    # 加入風控
    daily_spread_adj, __ = apply_risk_control(daily_spread)
    
    cum = (1 + daily_spread_adj).cumprod()
    dd = cum / cum.cummax() - 1
    return daily_spread, cum, dd


def run_long_only_top_backtest(test):
    """
    每日僅前 10% 強勢股持倉（等權），其餘 0。檢視選股能力，避免等權多頭稀釋。
    """
    t = test.copy()
    t["proba_prev"] = t.groupby("ticker")["proba_strong"].shift(1)
    t = t.dropna(subset=["proba_prev"])
    if len(t) == 0:
        return None, None, None
    t["rank_pct"] = t.groupby("date")["proba_prev"].rank(pct=True)
    daily_return = t[t["rank_pct"] >= (1 - TOP_PCT)].groupby("date")["return_1"].mean()
    if len(daily_return) == 0:
        return None, None, None
    cum = (1 + daily_return).cumprod()
    dd = cum / cum.cummax() - 1
    return daily_return, cum, dd


def run_long_short_backtest_by_period(test, freq="M"):
    """
    按 月(freq='M') 或 季(freq='Q') 聚合，每期內依前一日 proba_strong 分成 Top 10% / Bottom 10%，
    計算該期 Long-Short 日報酬序列，再算 True Spread 年化、Sharpe、MDD、勝率。
    回傳 DataFrame，方便存 CSV 或畫表。
    """
    t = test.copy()
    t["date"] = pd.to_datetime(t["date"])
    t["proba_prev"] = t.groupby("ticker")["proba_strong"].shift(1)
    t = t.dropna(subset=["proba_prev"])
    if len(t) == 0:
        return pd.DataFrame()

    if freq.upper() == "M":
        t["period"] = t["date"].dt.to_period("M")
    else:
        t["period"] = t["date"].dt.to_period("Q")

    rows = []
    for period, g in t.groupby("period", sort=True):
        g = g.copy()
        g["rank_pct"] = g.groupby("date")["proba_prev"].rank(pct=True)
        long_ret = g[g["rank_pct"] >= (1 - TOP_PCT)].groupby("date")["return_1"].mean()
        short_ret = g[g["rank_pct"] <= BOTTOM_PCT].groupby("date")["return_1"].mean()
        common_idx = long_ret.index.intersection(short_ret.index).sort_values()
        if len(common_idx) == 0:
            continue
        daily_spread = long_ret.loc[common_idx] - short_ret.loc[common_idx]
        stats = compute_stats_from_series(daily_spread, is_spread=True)
        if stats is None:
            continue
        rows.append({
            "period": str(period),
            "true_spread_ann": round(stats["true_spread_ann"], 4),
            "sharpe": round(stats["sharpe"], 4) if pd.notna(stats["sharpe"]) else None,
            "mdd": round(stats["mdd"], 4),
            "win_rate": round(stats["win_rate"], 4),
            "n_days": int(stats["n_days"]),
        })
    return pd.DataFrame(rows)


def compute_stats_from_series(daily_series, is_spread=False):
    """由每日報酬（或 spread）序列計算統計。is_spread=True 時標示為年化 spread。"""
    n = len(daily_series)
    if n == 0:
        return None
    mean_d = float(daily_series.mean())
    std_d = float(daily_series.std())
    ann = mean_d * 252
    sharpe = float((mean_d / std_d * np.sqrt(252))) if std_d > 0 else np.nan
    cum = (1 + daily_series).cumprod()
    dd = cum / cum.cummax() - 1
    mdd = float(dd.min())
    win_rate = float((daily_series > 0).mean())
    key_ann = "true_spread_ann" if is_spread else "ann_return"
    return {
        key_ann: ann,
        "sharpe": sharpe,
        "mdd": mdd,
        "win_rate": win_rate,
        "n_days": n,
    }


def compute_stats_equal_weight(daily_return, portfolio_dd, test):
    """等權多頭統計（易有指數化效應，僅供參考）。"""
    n = len(daily_return)
    if n == 0:
        return None
    ann_return = float(daily_return.mean() * 252)
    mdd = float(portfolio_dd.min())
    std = float(daily_return.std())
    sharpe = float((daily_return.mean() / std * np.sqrt(252))) if std > 0 else np.nan
    win_rate = float((daily_return > 0).mean())
    avg_position = float(test["position"].mean())
    buy_signal = test["proba_strong"] > PROBA_THRESHOLD
    n_trades = len(test[buy_signal])
    return {
        "ann_return": ann_return,
        "sharpe": sharpe,
        "mdd": mdd,
        "win_rate": win_rate,
        "n_days": n,
        "avg_position": avg_position,
        "n_trades": n_trades,
    }


def print_stats(stats_ew, stats_ls, stats_lo):
    """先輸出 Long-Short 與 Long-Only Top10%（主指標），等權多頭僅供參考放最後。"""
    print("=== 回測統計（僅未來資料）===")
    print("※ 主指標：Long-Short 與 Long-Only Top 10%；等權多頭僅供參考（低波動指數化效應）。\n")

    if stats_ls is not None:
        print("【Long-Short】Top 10% vs Bottom 10% → 衡量模型是否真正區分強弱（真實 alpha）")
        print("  True Spread 年化:", stats_ls["true_spread_ann"], "  夏普:", stats_ls["sharpe"], "  MDD:", stats_ls["mdd"])
        print("  勝率:", stats_ls["win_rate"], "  交易日:", stats_ls["n_days"])
        print()

    if stats_lo is not None:
        print("【Long-Only Top 10%】僅強勢股持倉 → 測選股能力，風險真實")
        print("  年化報酬:", stats_lo["ann_return"], "  夏普:", stats_lo["sharpe"], "  MDD:", stats_lo["mdd"])
        print("  勝率:", stats_lo["win_rate"], "  交易日:", stats_lo["n_days"])
        print()

    if stats_ew is not None:
        print("【等權多頭】僅供參考（低波動指數化效應，非選股能力指標）")
        print("  年化報酬:", stats_ew["ann_return"], "  夏普:", stats_ew["sharpe"], "  MDD:", stats_ew["mdd"])
        print("  平均部位:", stats_ew["avg_position"], "  進場次數:", stats_ew["n_trades"], "  交易日:", stats_ew["n_days"])


def _write_one_equity_drawdown(portfolio_cum, portfolio_dd, equity_path, dd_path, title_suffix):
    """寫入一組權益曲線 + 回撤曲線到指定路徑。"""
    if portfolio_cum is None or len(portfolio_cum) == 0:
        return
    dates = portfolio_cum.index
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(dates, portfolio_cum.values, color="tab:blue", linewidth=1)
    ax1.set_ylabel("累積報酬")
    ax1.set_title("驗證回測：權益曲線（累積報酬）" + title_suffix)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1, color="gray", linestyle="--", alpha=0.7)
    fig.tight_layout()
    fig.savefig(equity_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.fill_between(dates, portfolio_dd.values, 0, color="tab:red", alpha=0.5)
    ax2.plot(dates, portfolio_dd.values, color="tab:red", linewidth=0.8)
    ax2.set_ylabel("回撤")
    ax2.set_title("驗證回測：回撤曲線" + title_suffix)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
    fig2.tight_layout()
    fig2.savefig(dd_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)


def save_plots(cum_ls, dd_ls, cum_lo, dd_lo):
    """產出 Long-Short 與 Long-Only Top 10% 兩組圖表（主指標，不做等權）。"""
    if cum_ls is not None and len(cum_ls) > 0:
        _write_one_equity_drawdown(
            cum_ls, dd_ls,
            OUTPUT_EQUITY_PNG, OUTPUT_DRAWDOWN_PNG,
            " (Long-Short Top10% vs Bottom10%)",
        )
        print("圖表已儲存 (Long-Short):", OUTPUT_EQUITY_PNG, OUTPUT_DRAWDOWN_PNG)
    if cum_lo is not None and len(cum_lo) > 0:
        _write_one_equity_drawdown(
            cum_lo, dd_lo,
            OUTPUT_EQUITY_TOP10_PNG, OUTPUT_DRAWDOWN_TOP10_PNG,
            " (Long-Only Top10%)",
        )
        print("圖表已儲存 (Long-Only Top10%):", OUTPUT_EQUITY_TOP10_PNG, OUTPUT_DRAWDOWN_TOP10_PNG)


def save_result_txt(stats_ew, stats_ls, stats_lo):
    """TXT 以 Long-Short 與 Long-Only Top 10% 為主，附正確解讀方式；等權多頭僅供參考。"""
    lines = [
        "=== 驗證結果（僅未來資料回測） ===",
        "模型路徑: " + str(MODEL_PATH),
        "測試區間起始: " + str(TEST_START),
        "主指標: Long-Short Top " + str(int(TOP_PCT * 100)) + "% vs Bottom " + str(int(BOTTOM_PCT * 100)) + "%，Long-Only Top 10%",
        "",
        "--- 正確解讀方式 ---",
        "• Long-Short Spread 的 Sharpe ≈ 1+、MDD < 10~15% → 模型確實有 alpha",
        "• Long-Only Top 10% 年化報酬 > 大盤且 Sharpe > 1 → 選股有效",
        "• 等權多頭之夏普/MDD 易受低波動指數化效應影響，僅供 beta 參考，不可當作選股能力指標",
        "",
    ]
    if stats_ls is not None:
        lines.extend([
            "【Long-Short】Top 10% vs Bottom 10% → 衡量模型是否真正區分強弱（真實 alpha）",
            "  True Spread 年化: " + str(stats_ls["true_spread_ann"]),
            "  夏普比率 (Sharpe): " + str(stats_ls["sharpe"]),
            "  最大回撤 (MDD): " + str(stats_ls["mdd"]),
            "  勝率: " + str(stats_ls["win_rate"]),
            "  總交易日: " + str(stats_ls["n_days"]),
            "",
        ])
    if stats_lo is not None:
        lines.extend([
            "【Long-Only Top 10%】僅強勢股持倉 → 測選股能力，風險真實",
            "  年化報酬: " + str(stats_lo["ann_return"]),
            "  夏普比率 (Sharpe): " + str(stats_lo["sharpe"]),
            "  最大回撤 (MDD): " + str(stats_lo["mdd"]),
            "  勝率: " + str(stats_lo["win_rate"]),
            "  總交易日: " + str(stats_lo["n_days"]),
            "",
        ])
    if stats_ew is not None:
        lines.extend([
            "【等權多頭】僅供參考（低波動指數化效應，非選股能力指標）",
            "  年化報酬: " + str(stats_ew["ann_return"]),
            "  夏普比率 (Sharpe): " + str(stats_ew["sharpe"]),
            "  最大回撤 (MDD): " + str(stats_ew["mdd"]),
            "  平均部位: " + str(stats_ew["avg_position"]),
            "  進場次數: " + str(stats_ew["n_trades"]),
            "  總交易日: " + str(stats_ew["n_days"]),
        ])
    OUTPUT_RESULT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print("驗證結果已儲存:", OUTPUT_RESULT_TXT)


def main(stock_list_path="stock_list.json"):
    model, scaler, features = load_model_artifacts()
    if features != FEATURES:
        print("警告：磁碟中的 feature 順序與目前 FEATURES 不一致，以磁碟為準")
        features = list(features)

    df = download_and_prepare_data(stock_list_path=stock_list_path)
    test = df[df["date"] >= TEST_START].copy()
    if len(test) == 0:
        print("無未來區間資料可驗證，請確認 TEST_START 與資料區間")
        return

    X_test = scaler.transform(test[FEATURES].fillna(0))
    proba = model.predict_proba(X_test)
    test.loc[:, "proba_strong"] = proba[:, 1]
    test.loc[:, "proba_weak"] = proba[:, 0]

    daily_return_ew, _portfolio_cum_ew, portfolio_dd_ew, test_with_position = run_backtest(test)
    stats_ew = compute_stats_equal_weight(daily_return_ew, portfolio_dd_ew, test_with_position)

    daily_spread, cum_ls, dd_ls = run_long_short_backtest(test)
    stats_ls = compute_stats_from_series(daily_spread, is_spread=True) if daily_spread is not None else None

    daily_lo, cum_lo, dd_lo = run_long_only_top_backtest(test)
    stats_lo = compute_stats_from_series(daily_lo, is_spread=False) if daily_lo is not None else None

    print_stats(stats_ew, stats_ls, stats_lo)

    # 圖表與 TXT 僅產出 Long-Short + Long-Only Top 10%；等權多頭僅在 TXT 參考
    save_plots(cum_ls, dd_ls, cum_lo, dd_lo)
    save_result_txt(stats_ew, stats_ls, stats_lo)

    # 每月 / 每季 Top vs Bottom 分組回測表格 → DataFrame + CSV
    df_monthly = run_long_short_backtest_by_period(test, freq="M")
    df_quarterly = run_long_short_backtest_by_period(test, freq="Q")
    if len(df_monthly) > 0:
        df_monthly.to_csv(OUTPUT_MONTHLY_CSV, index=False, encoding="utf-8-sig")
        print("每月 Long-Short 表已儲存:", OUTPUT_MONTHLY_CSV)
    if len(df_quarterly) > 0:
        df_quarterly.to_csv(OUTPUT_QUARTERLY_CSV, index=False, encoding="utf-8-sig")
        print("每季 Long-Short 表已儲存:", OUTPUT_QUARTERLY_CSV)


if __name__ == "__main__":
    main()
