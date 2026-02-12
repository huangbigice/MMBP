"""
五層 Alpha 驗證測試。

- Layer 1: Null Model 殺戮測試（真實 vs 打亂 vs 永遠觀望）
- Layer 2: Walk-forward 穩定性（每段 Sharpe 分佈）
- Layer 3: Regime 生存測試（多頭/空頭/高波動/低波動）
- Layer 4: Benchmark 智商測試（B&H、Momentum、MA cross、RSI(14)>50）
- Layer 5: Economic Plausibility 自評表
"""

from __future__ import annotations

import sys
from pathlib import Path

# 專案根目錄
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config.config_loader import ConfigLoader
from config.versioning import get_model_version_info
from models.model_loader import ModelLoadConfig, ModelLoader
from services.backtest_service import (
    BacktestService,
    FEE_RATE,
    RISK_FREE_RATE,
)
from services.data_service import DataService
from services.backtest.walk_forward import (
    generate_walk_forward_windows,
    run_walk_forward,
)
from train_model.train_model第五版 import compute_rsi

DEFAULT_SYMBOL = "2330.TW"


# ---------------------------------------------------------------------------
# Layer 1: Null Model 殺戮測試
# ---------------------------------------------------------------------------

def run_layer1_null_test(
    backtest_service: BacktestService,
    symbol: str = DEFAULT_SYMBOL,
    start: str = "2018-01-01",
    end: str = "2025-12-31",
    *,
    sharpe_min_advantage: float = 0.2,
    ann_return_min_advantage: float = 0.02,
) -> dict:
    """
    第一層：Null Model 殺戮測試。
    同一回測、同一成本、同一風控下，比較：
    1. 真實模型
    2. Label 打亂（隨機訊號）
    3. 永遠觀望

    若真實模型在風險調整後報酬上未顯著優於 2 和 3，則判定為統計幻覺。
    """
    print("\n" + "=" * 60)
    print("【第一層】Null Model 殺戮測試")
    print("=" * 60)

    results = {}
    for label, null_mode, seed in [
        ("真實模型", None, None),
        ("打亂訊號", "shuffled", 42),
        ("永遠觀望", "hold", None),
    ]:
        try:
            r = backtest_service.run_backtest(
                symbol=symbol,
                start=start,
                end=end,
                null_mode=null_mode,
                shuffle_seed=seed,
            )
            results[label] = {
                "annualized_return": r.annualized_return,
                "volatility": r.volatility,
                "max_drawdown": r.max_drawdown,
                "sharpe_ratio": r.sharpe_ratio if r.sharpe_ratio is not None else 0.0,
                "trade_count": r.trade_count,
            }
            print(
                f"  {label}: "
                f"年化報酬={r.annualized_return:.4f} "
                f"波動={r.volatility:.4f} "
                f"Sharpe={r.sharpe_ratio or 0:.4f} "
                f"最大回撤={r.max_drawdown:.4f} "
                f"交易次數={r.trade_count}"
            )
        except Exception as e:
            print(f"  {label}: 執行失敗 - {e}")
            results[label] = None

    real = results.get("真實模型")
    shuffled = results.get("打亂訊號")
    hold = results.get("永遠觀望")

    passed = True
    if real and shuffled and hold:
        sr_real = real["sharpe_ratio"]
        sr_shuffled = shuffled["sharpe_ratio"] if shuffled else 0.0
        sr_hold = hold["sharpe_ratio"] if hold else 0.0
        ret_real = real["annualized_return"]
        ret_shuffled = shuffled["annualized_return"] if shuffled else 0.0

        if sr_real <= sr_shuffled + sharpe_min_advantage:
            print("\n  ⚠ 失敗：真實模型 Sharpe 未顯著高於「打亂訊號」")
            passed = False
        if sr_real <= sr_hold + sharpe_min_advantage:
            print("\n  ⚠ 失敗：真實模型 Sharpe 未顯著高於「永遠觀望」")
            passed = False
        if ret_real <= ret_shuffled + ann_return_min_advantage:
            print("\n  ⚠ 失敗：真實模型年化報酬未顯著高於「打亂訊號」")
            passed = False
        if passed:
            print("\n  ✓ 通過：真實模型在風險調整後顯著優於隨機與觀望")
    else:
        passed = False
        print("\n  ⚠ 無法比較：部分回測失敗")

    return {"results": results, "passed": passed}


# ---------------------------------------------------------------------------
# Layer 2: Walk-forward 穩定性
# ---------------------------------------------------------------------------

def run_layer2_walk_forward_stability(
    backtest_service: BacktestService,
    symbol: str = DEFAULT_SYMBOL,
    start: str = "2010-01-01",
    end: str = "2025-12-31",
    train_years: int = 5,
    test_years: int = 1,
    *,
    plot: bool = True,
) -> dict:
    """
    第二層：Walk-forward 穩定性。
    時間滾動切：2010–2015 → 測 2016，2011–2016 → 測 2017 ...
    輸出每段 Sharpe 分佈、變異數、負 Sharpe 比例。
    """
    print("\n" + "=" * 60)
    print("【第二層】Walk-forward 穩定性測試")
    print("=" * 60)

    def run_bt(sym: str, s: str | None, e: str | None):
        return backtest_service.run_backtest(symbol=sym, start=s, end=e)

    windows = generate_walk_forward_windows(
        start, end, train_years=train_years, test_years=test_years
    )
    print(f"  視窗數: {len(windows)}")

    wf_results = run_walk_forward(
        symbol=symbol,
        start=start,
        end=end,
        run_backtest_fn=run_bt,
        train_years=train_years,
        test_years=test_years,
    )

    sharpes = [r.sharpe_ratio for r in wf_results if r.sharpe_ratio is not None]
    if not sharpes:
        print("  無有效 Sharpe 資料")
        return {"sharpes": [], "passed": False}

    sharpes_arr = np.array(sharpes)
    n_neg = (sharpes_arr < 0).sum()
    pct_neg = n_neg / len(sharpes_arr) * 100
    print(f"  各段 Sharpe: 平均={np.mean(sharpes_arr):.4f} 標準差={np.std(sharpes_arr):.4f}")
    print(f"  負 Sharpe 比例: {pct_neg:.1f}% ({n_neg}/{len(sharpes_arr)})")

    if plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 4))
            plt.hist(sharpes_arr, bins=min(15, len(sharpes_arr)), edgecolor="black", alpha=0.7)
            plt.axvline(0, color="red", linestyle="--", label="Sharpe=0")
            plt.xlabel("Sharpe Ratio (per test window)")
            plt.ylabel("Count")
            plt.title("Layer 2: Walk-forward Sharpe 分佈")
            plt.legend()
            out = ROOT / "validation" / "layer2_sharpe_dist.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=120)
            plt.close()
            print(f"  圖表已存: {out}")
        except Exception as e:
            print(f"  繪圖略過: {e}")

    # 通過條件：Sharpe 變異不要過大，且負 Sharpe 比例 < 50%
    passed = np.std(sharpes_arr) < 1.5 and pct_neg < 50
    if pct_neg >= 50:
        print("\n  ⚠ 失敗：超過一半時間 Sharpe 為負，典型 overfit")
    elif np.std(sharpes_arr) >= 1.5:
        print("\n  ⚠ 注意：Sharpe 波動大，穩定性存疑")
    else:
        print("\n  ✓ 通過：Sharpe 分佈尚可接受")

    return {"sharpes": sharpes, "pct_negative": pct_neg, "std_sharpe": float(np.std(sharpes_arr)), "passed": passed}


# ---------------------------------------------------------------------------
# Layer 3: Regime 生存測試
# ---------------------------------------------------------------------------

def run_layer3_regime_survival(
    backtest_service: BacktestService,
    symbol: str = "2330.TW",
    start: str = "2018-01-01",
    end: str = "2025-12-31",
    *,
    vol_percentile_high: float = 0.75,
) -> dict:
    """
    第三層：Regime 生存測試。
    依多頭/空頭、高波動/低波動切分，看最爛 regime 是否爆炸。
    若多頭賺、空頭爆 → 可能只是加槓桿的 beta。
    """
    print("\n" + "=" * 60)
    print("【第三層】Regime 生存測試")
    print("=" * 60)

    try:
        df = backtest_service.get_backtest_df(symbol=symbol, start=start, end=end)
    except Exception as e:
        print(f"  取得回測 df 失敗: {e}")
        return {"passed": False}

    if "regime" not in df.columns or "mkt_vol_60d" not in df.columns:
        print("  缺少 regime 或 mkt_vol_60d，改用 close 波動代理")
        df["ret"] = df["return_1"]
        vol_60 = df["ret"].rolling(60).std()
        if "mkt_vol_60d" not in df.columns:
            df["mkt_vol_60d"] = vol_60

    # 波動高低：以 mkt_vol_60d 分位數切
    vol_th = df["mkt_vol_60d"].quantile(vol_percentile_high)
    df["vol_regime"] = np.where(df["mkt_vol_60d"] >= vol_th, "high_vol", "low_vol")

    regime_stats = []
    for regime_name, mask in [
        ("多頭 (regime=1)", df["regime"] == 1),
        ("空頭 (regime=-1)", df["regime"] == -1),
        ("高波動", df["vol_regime"] == "high_vol"),
        ("低波動", df["vol_regime"] == "low_vol"),
    ]:
        sub = df.loc[mask, "strategy_return"].dropna()
        if len(sub) < 10:
            regime_stats.append({"regime": regime_name, "n": len(sub), "ann_ret": None, "max_dd": None})
            print(f"  {regime_name}: 樣本不足 (n={len(sub)})")
            continue
        n_years = len(sub) / 252
        ann_ret = (1 + sub).prod() ** (1 / n_years) - 1 if n_years > 0 else 0.0
        cum = (1 + sub).cumprod()
        dd = cum / cum.cummax() - 1
        max_dd = float(dd.min())
        regime_stats.append({"regime": regime_name, "n": len(sub), "ann_ret": ann_ret, "max_dd": max_dd})
        print(f"  {regime_name}: 年化報酬={ann_ret:.4f} 最大回撤={max_dd:.4f} (n={len(sub)})")

    # 檢查最爛 regime 是否「爆炸」（例如最大回撤 < -0.25 或年化報酬 < -0.15）
    worst_dd = min((s["max_dd"] for s in regime_stats if s["max_dd"] is not None), default=0)
    worst_ret = min((s["ann_ret"] for s in regime_stats if s["ann_ret"] is not None), default=0)
    passed = worst_dd > -0.35 and worst_ret > -0.20
    if worst_dd <= -0.35:
        print("\n  ⚠ 失敗：某 regime 最大回撤過大（可能只是 beta）")
    elif worst_ret <= -0.20:
        print("\n  ⚠ 失敗：某 regime 年化報酬過差")
    else:
        print("\n  ✓ 通過：各 regime 未見爆炸")

    return {"regime_stats": regime_stats, "worst_drawdown": worst_dd, "worst_return": worst_ret, "passed": passed}


# ---------------------------------------------------------------------------
# Layer 4: Benchmark 智商測試
# ---------------------------------------------------------------------------

def _benchmark_returns(
    df: pd.DataFrame,
    fee_rate: float = FEE_RATE,
) -> dict[str, pd.Series]:
    """同一段資料、同一成本，計算各基準策略的日報酬。回傳之 Series 以 date 為 index。"""
    df = df.copy()
    if "date" not in df.columns and df.index.dtype == "datetime64[ns]":
        df = df.reset_index().rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    ret = df["return_1"].copy()
    out = {}

    # Buy & Hold
    pos_bh = pd.Series(1.0, index=df.index)
    turnover_bh = pos_bh.diff().abs().fillna(0.0)
    sr = pos_bh.shift(1) * ret - turnover_bh * fee_rate
    out["Buy & Hold"] = sr.set_axis(df["date"].values).fillna(0.0)

    # Momentum: 過去 252 日報酬 > 0 則持多
    if len(df) >= 260:
        ret_252 = df["close"].pct_change(252)
        pos_mom = (ret_252 > 0).astype(float)
        turnover_mom = pos_mom.diff().abs().fillna(0.0)
        sr = pos_mom.shift(1) * ret - turnover_mom * fee_rate
        out["Momentum"] = sr.set_axis(df["date"].values).fillna(0.0)
    else:
        out["Momentum"] = pd.Series(0.0, index=df["date"].values)

    # MA crossover: ma20 > ma60 持多
    if "ma20" in df.columns and "ma60" in df.columns:
        pos_ma = (df["ma20"] > df["ma60"]).astype(float)
        turnover_ma = pos_ma.diff().abs().fillna(0.0)
        sr = pos_ma.shift(1) * ret - turnover_ma * fee_rate
        out["MA crossover"] = sr.set_axis(df["date"].values).fillna(0.0)
    else:
        out["MA crossover"] = pd.Series(0.0, index=df["date"].values)

    # RSI(14) > 50 持多
    rsi14 = compute_rsi(df["close"], 14)
    pos_rsi = (rsi14 > 50).astype(float)
    pos_rsi = pos_rsi.fillna(0.5)
    turnover_rsi = pos_rsi.diff().abs().fillna(0.0)
    sr = pos_rsi.shift(1) * ret - turnover_rsi * fee_rate
    out["RSI(14)>50"] = sr.set_axis(df["date"].values).fillna(0.0)

    return out


def run_layer4_benchmark_iq(
    backtest_service: BacktestService,
    data_service: DataService,
    symbol: str = DEFAULT_SYMBOL,
    start: str = "2018-01-01",
    end: str = "2025-12-31",
) -> dict:
    """
    第四層：Benchmark 智商測試。
    真實策略必須在「風險調整後報酬」上打贏：
    Buy & Hold、Momentum、MA crossover、RSI(14)>50。
    """
    print("\n" + "=" * 60)
    print("【第四層】Benchmark 智商測試")
    print("=" * 60)

    try:
        df_bt = backtest_service.get_backtest_df(symbol=symbol, start=start, end=end)
    except Exception as e:
        print(f"  取得回測 df 失敗: {e}")
        return {"passed": False}

    # 策略日報酬（已有手續費）
    strategy_ret = df_bt.set_index("date")["strategy_return"]

    # 基準策略需要同一段資料 + 同一成本
    df_data = data_service.fetch_stock_data(symbol=symbol, start=start, end=end)
    df_data = data_service.add_indicators(df_data)
    df_data = data_service.add_market_regime(df_data)
    df_data["ma20"] = df_data["close"].rolling(20).mean()
    df_data["ma60"] = df_data["close"].rolling(60).mean()
    df_data = df_data.dropna(subset=["ma60"]).reset_index(drop=True)
    df_data["return_1"] = df_data["close"].pct_change(1)

    bench_rets = _benchmark_returns(df_data, fee_rate=FEE_RATE)

    # 對齊日期（策略與 df_data 的 date）
    common_dates = strategy_ret.index.intersection(
        pd.DatetimeIndex(pd.to_datetime(df_data["date"]).dt.normalize().unique())
    ).sort_values()
    if len(common_dates) < 20:
        print("  共同區間不足")
        return {"passed": False}

    def sharpe(s: pd.Series) -> float:
        s = s.dropna()
        if len(s) < 2:
            return 0.0
        ann_ret = (1 + s).prod() ** (252 / len(s)) - 1
        ann_vol = s.std() * np.sqrt(252)
        if ann_vol <= 0:
            return 0.0
        return (ann_ret - RISK_FREE_RATE) / ann_vol

    # 策略 Sharpe（對齊後）
    strat_aligned = strategy_ret.reindex(common_dates).fillna(0.0)
    sharpe_strat = sharpe(strat_aligned)

    print(f"  真實策略 Sharpe (對齊區間): {sharpe_strat:.4f}")

    results = {"策略": sharpe_strat}
    for name, sr in bench_rets.items():
        aligned = sr.reindex(common_dates).fillna(0.0)
        sh = sharpe(aligned)
        results[name] = sh
        print(f"  {name}: Sharpe={sh:.4f}")

    # 通過：策略 Sharpe 不低於所有 benchmark
    passed = all(sharpe_strat >= results[k] - 1e-6 for k in results if k != "策略")
    if not passed:
        print("\n  ⚠ 失敗：風險調整後報酬未全面打贏基準（若輸給 MA crossover，經濟上無存在意義）")
    else:
        print("\n  ✓ 通過：風險調整後報酬優於或持平所有基準")

    return {"sharpe_results": results, "passed": passed}


# ---------------------------------------------------------------------------
# Layer 5: Economic Plausibility
# ---------------------------------------------------------------------------

def run_layer5_economic_plausibility() -> dict:
    """
    第五層：Economic Plausibility 自評。
    輸出極殘酷問題清單，若只能答「模型學出來的」= 在專業世界裡不存在。
    """
    print("\n" + "=" * 60)
    print("【第五層】Economic Plausibility 自評")
    print("=" * 60)
    print("""
  請明確寫出：這個 alpha 的經濟來源是什麼？

  常見來源舉例：
    - 行為偏誤（例如散戶過度反應、處置效應）
    - 流動性溢價（承擔流動性風險的補償）
    - 風險補償（承擔特定風險因子的補償）
    - 資訊延遲（公開資訊未被即時反映）
    - 結構性因素（例如指數再平衡、基金調倉）

  若你只能說「模型學出來的」→ 在專業投資世界裡等於不存在。

  請將你的答案寫入：docs/economic_plausibility_填寫.md
""")
    out_path = ROOT / "docs" / "economic_plausibility_填寫.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        template = """# Economic Plausibility 自評

## 本策略 Alpha 的經濟來源（請填寫）

1. **主要來源**：（例如：行為偏誤 / 流動性溢價 / 風險補償 / 資訊延遲 / 其他）

2. **具體說明**：（一至兩段話）

3. **可驗證方式**：（如何用資料或文獻支持）

---
*若僅能回答「模型學出來的」，請視為未通過第五層。*
"""
        out_path.write_text(template, encoding="utf-8")
        print(f"  已建立範本: {out_path}")
    else:
        print(f"  請編輯: {out_path}")
    return {"template_path": str(out_path), "passed": None}  # 需人工判定


# ---------------------------------------------------------------------------
# 全五層一次執行
# ---------------------------------------------------------------------------

def run_all_five_layers(
    symbol: str = DEFAULT_SYMBOL,
    start: str = "2018-01-01",
    end: str = "2025-12-31",
    wf_start: str = "2010-01-01",
    *,
    plot_layer2: bool = True,
) -> dict:
    """依序執行五層測試。需在專案根目錄執行，或確保 config/model 可載入。"""
    config = ConfigLoader().config
    model_loader = ModelLoader(ModelLoadConfig(model_path=config.model_path))
    data_service = DataService()
    version_info = get_model_version_info()
    backtest_service = BacktestService(
        data_service=data_service,
        model_loader=model_loader,
        model_version_info=version_info,
    )

    summary = {}
    summary["layer1"] = run_layer1_null_test(backtest_service, symbol=symbol, start=start, end=end)
    summary["layer2"] = run_layer2_walk_forward_stability(
        backtest_service, symbol=symbol, start=wf_start, end=end, plot=plot_layer2
    )
    summary["layer3"] = run_layer3_regime_survival(backtest_service, symbol=symbol, start=start, end=end)
    summary["layer4"] = run_layer4_benchmark_iq(backtest_service, data_service, symbol=symbol, start=start, end=end)
    summary["layer5"] = run_layer5_economic_plausibility()

    passed_count = sum(
        1 for k in ["layer1", "layer2", "layer3", "layer4"]
        if summary.get(k) and summary[k].get("passed") is True
    )
    print("\n" + "=" * 60)
    print(f"五層測試摘要：{passed_count}/4 層通過（Layer 5 需人工填寫）")
    print("=" * 60)
    return summary


if __name__ == "__main__":
    run_all_five_layers(
        symbol=DEFAULT_SYMBOL,
        start="2018-01-01",
        end="2025-12-31",
        wf_start="2010-01-01",
        plot_layer2=True,
    )
