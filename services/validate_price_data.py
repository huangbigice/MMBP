"""
股價數據驗證工具：檢查從 yfinance 下載的股價數據是否準確。

使用方法：
    from services.validate_price_data import validate_price_data
    from services.data_service import DataService
    
    data_service = DataService()
    df = data_service.fetch_stock_data("2330.TW", start="2024-01-01", end="2025-01-01")
    
    issues = validate_price_data(df, symbol="2330.TW", verbose=True)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any


def validate_price_data(
    df: pd.DataFrame,
    symbol: str = "",
    verbose: bool = True,
) -> dict[str, Any]:
    """
    驗證股價數據的準確性和完整性。
    
    Parameters
    ----------
    df : pd.DataFrame
        包含股價數據的 DataFrame，應有 date, open, high, low, close, volume 欄位。
    symbol : str
        股票代碼（用於顯示）。
    verbose : bool
        是否輸出詳細驗證資訊。
    
    Returns
    -------
    dict
        驗證結果，包含：
        - issues: 發現的問題列表
        - stats: 數據統計資訊
        - recommendations: 建議修正措施
    """
    issues = []
    stats = {}
    recommendations = []
    
    if verbose:
        print("=" * 60)
        print(f"股價數據驗證：{symbol}")
        print("=" * 60)
    
    # 1. 檢查必要欄位
    required_cols = ["date", "open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"缺少必要欄位：{missing_cols}")
        if verbose:
            print(f"❌ 缺少必要欄位：{missing_cols}")
        return {"issues": issues, "stats": stats, "recommendations": recommendations}
    
    # 2. 檢查數據完整性
    total_rows = len(df)
    stats["total_rows"] = total_rows
    
    if verbose:
        print(f"\n數據總筆數：{total_rows}")
    
    # 檢查缺失值
    for col in required_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            issues.append(f"{col} 欄位有 {nan_count} 筆缺失值")
            if verbose:
                print(f"⚠️ {col} 欄位：{nan_count} 筆缺失值（{nan_count/total_rows:.1%}）")
    
    # 3. 檢查價格邏輯（high >= low, high >= close, low <= close 等）
    price_issues = []
    if "high" in df.columns and "low" in df.columns:
        invalid_hl = (df["high"] < df["low"]).sum()
        if invalid_hl > 0:
            price_issues.append(f"high < low：{invalid_hl} 筆")
    
    if "high" in df.columns and "close" in df.columns:
        invalid_hc = (df["high"] < df["close"]).sum()
        if invalid_hc > 0:
            price_issues.append(f"high < close：{invalid_hc} 筆")
    
    if "low" in df.columns and "close" in df.columns:
        invalid_lc = (df["low"] > df["close"]).sum()
        if invalid_lc > 0:
            price_issues.append(f"low > close：{invalid_lc} 筆")
    
    if price_issues:
        issues.extend(price_issues)
        if verbose:
            print(f"\n❌ 價格邏輯錯誤：")
            for issue in price_issues:
                print(f"   - {issue}")
        recommendations.append("檢查數據源，可能是除權除息或數據錯誤")
    elif verbose:
        print("\n✅ 價格邏輯正確")
    
    # 4. 檢查異常值
    if "close" in df.columns:
        # 檢查是否有負值或零值
        non_positive = (df["close"] <= 0).sum()
        if non_positive > 0:
            issues.append(f"close 欄位有 {non_positive} 筆非正值")
            if verbose:
                print(f"❌ close 欄位：{non_positive} 筆非正值")
        
        # 檢查異常大的單日變化（可能是數據錯誤）
        pct_change = df["close"].pct_change().abs()
        extreme_changes_50 = (pct_change > 0.5).sum()
        extreme_changes_20 = (pct_change > 0.2).sum()
        
        stats["extreme_changes_50pct"] = extreme_changes_50
        stats["extreme_changes_20pct"] = extreme_changes_20
        
        if verbose:
            print(f"\n單日變化統計：")
            print(f"  >50%：{extreme_changes_50} 筆")
            print(f"  >20%：{extreme_changes_20} 筆")
        
        if extreme_changes_50 > 0:
            issues.append(f"有 {extreme_changes_50} 筆單日變化超過50%")
            recommendations.append("檢查是否為除權除息或數據錯誤")
        
        # 檢查價格範圍是否合理
        if len(df) > 0:
            price_min = df["close"].min()
            price_max = df["close"].max()
            price_mean = df["close"].mean()
            price_std = df["close"].std()
            
            stats["price_range"] = {
                "min": float(price_min),
                "max": float(price_max),
                "mean": float(price_mean),
                "std": float(price_std),
            }
            
            if verbose:
                print(f"\n價格統計：")
                print(f"  範圍：[{price_min:.2f}, {price_max:.2f}]")
                print(f"  平均：{price_mean:.2f}")
                print(f"  標準差：{price_std:.2f}")
            
            # 如果價格範圍異常（比如台股價格不應該<10或>10000），發出警告
            if price_min < 1 or price_max > 10000:
                issues.append(f"價格範圍異常：[{price_min:.2f}, {price_max:.2f}]")
                recommendations.append("檢查股票代碼是否正確（台股請使用 .TW 後綴）")
    
    # 5. 檢查日期連續性
    if "date" in df.columns:
        df_sorted = df.sort_values("date")
        date_diff = df_sorted["date"].diff().dt.days
        # 檢查是否有異常大的日期間隔（可能是數據缺失）
        large_gaps = (date_diff > 10).sum()
        if large_gaps > 0:
            issues.append(f"日期間隔超過10天的有 {large_gaps} 處")
            if verbose:
                print(f"\n⚠️ 日期間隔：{large_gaps} 處超過10天（可能是數據缺失）")
        elif verbose:
            print("\n✅ 日期連續性正常")
    
    # 6. 檢查成交量
    if "volume" in df.columns:
        zero_volume = (df["volume"] == 0).sum()
        if zero_volume > total_rows * 0.1:  # 如果超過10%是零成交量
            issues.append(f"零成交量比例過高：{zero_volume/total_rows:.1%}")
            if verbose:
                print(f"\n⚠️ 零成交量：{zero_volume} 筆（{zero_volume/total_rows:.1%}）")
    
    # 總結
    if verbose:
        print("\n" + "=" * 60)
        if issues:
            print(f"發現 {len(issues)} 個問題：")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
            
            if recommendations:
                print(f"\n建議修正措施：")
                for i, rec in enumerate(recommendations, 1):
                    print(f"  {i}. {rec}")
        else:
            print("✅ 未發現問題，數據質量良好！")
        print("=" * 60)
    
    return {
        "issues": issues,
        "stats": stats,
        "recommendations": recommendations,
    }
