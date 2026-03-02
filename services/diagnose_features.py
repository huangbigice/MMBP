"""
特徵工程終極診斷工具：用於診斷 MKT_REGIME_FEATURES 合併問題。

使用方法：
    from services.diagnose_features import diagnose_feature_issue
    from models.model_loader import ModelLoader
    from config.config_loader import load_config
    
    config = load_config()
    model_loader = ModelLoader(config.model_config)
    data_service = DataService()
    
    df = data_service.fetch_stock_data("2330.TW", start="2020-01-01", end="2025-01-01")
    df = data_service.add_indicators(df)
    df = data_service.add_market_regime(df)
    
    model_data = {
        "model": model_loader.load(),
        "scaler": model_loader.get_scaler(),
        "features": model_loader.get_features(),
    }
    
    diagnose_feature_issue(df, model_data)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any

from train_model.market_regime import MKT_REGIME_FEATURES


def diagnose_feature_issue(
    df: pd.DataFrame,
    model_data: dict[str, Any],
    verbose: bool = True,
) -> dict[str, Any]:
    """
    終極特徵診斷：檢查特徵工程是否與訓練模型一致。
    
    Parameters
    ----------
    df : pd.DataFrame
        已計算特徵的 DataFrame（應包含所有12個特徵）。
    model_data : dict
        包含 "model", "scaler", "features" 的字典。
    verbose : bool
        是否輸出詳細診斷資訊。
    
    Returns
    -------
    dict
        診斷結果字典，包含：
        - features_status: 特徵存在狀態
        - mkt_regime_status: MKT_REGIME_FEATURES 狀態
        - feature_stats: 關鍵特徵統計
        - proba_stats: proba_strong 統計
        - issues: 發現的問題列表
    """
    FEATURES = model_data.get("features", [])
    model = model_data.get("model")
    scaler = model_data.get("scaler")
    
    issues = []
    results = {
        "features_status": {},
        "mkt_regime_status": {},
        "feature_stats": {},
        "proba_stats": {},
        "issues": issues,
    }
    
    if verbose:
        print("=" * 60)
        print("特徵工程終極診斷")
        print("=" * 60)
    
    # 1. 檢查12個特徵是否存在
    if FEATURES:
        missing_features = set(FEATURES) - set(df.columns)
        extra_features = set(df.columns) - set(FEATURES)
        
        results["features_status"] = {
            "total": len(FEATURES),
            "present": len(FEATURES) - len(missing_features),
            "missing": list(missing_features),
            "extra": list(extra_features),
        }
        
        if verbose:
            print(f"\n✅ 特徵數量: {len(FEATURES)}")
            print(f"   存在: {results['features_status']['present']}/{len(FEATURES)}")
            if missing_features:
                print(f"   ❌ 缺失: {missing_features}")
                issues.append(f"缺少特徵: {missing_features}")
            if extra_features:
                print(f"   ⚠️  多餘: {len(extra_features)} 個（不影響）")
    else:
        if verbose:
            print("\n⚠️ 模型未保存特徵列表，無法驗證特徵一致性")
        issues.append("模型未保存特徵列表")
    
    # 2. MKT_REGIME特徵檢查（關鍵！）
    mkt_cols = [col for col in df.columns if any(mkt in col.lower() for mkt in ['mkt', 'market'])]
    mkt_features_present = [f for f in MKT_REGIME_FEATURES if f in df.columns]
    mkt_features_missing = [f for f in MKT_REGIME_FEATURES if f not in df.columns]
    
    results["mkt_regime_status"] = {
        "required": len(MKT_REGIME_FEATURES),
        "present": len(mkt_features_present),
        "missing": mkt_features_missing,
        "nan_ratio": None,
    }
    
    if verbose:
        print(f"\n=== 大盤特徵診斷（關鍵！）===")
        print(f"MKT_REGIME特徵: {mkt_features_present}")
        print(f"缺失: {mkt_features_missing}")
    
    if mkt_features_missing:
        if verbose:
            print(f"🚨 致命錯誤：MKT_REGIME_FEATURES 缺失 {len(mkt_features_missing)} 個！")
        issues.append(f"MKT_REGIME_FEATURES缺失: {mkt_features_missing}")
    elif mkt_features_present:
        # 檢查 NaN 比例
        mkt_nan_ratio = df[mkt_features_present].isna().mean().mean()
        results["mkt_regime_status"]["nan_ratio"] = float(mkt_nan_ratio)
        
        if verbose:
            print(f"NaN比例: {mkt_nan_ratio:.1%}")
        
        if mkt_nan_ratio > 0.1:
            if verbose:
                print(f"⚠️ 警告：MKT_REGIME NaN比例過高（{mkt_nan_ratio:.1%}）")
            issues.append(f"MKT_REGIME NaN比例過高: {mkt_nan_ratio:.1%}")
    
    # 3. 關鍵特徵統計
    key_features = ['return_1', 'rsi_14', 'volume_spike', 'price_strength']
    if verbose:
        print(f"\n=== 關鍵特徵分佈 ===")
    
    for feat in key_features:
        if feat in df.columns:
            stats = {
                "mean": float(df[feat].mean()),
                "std": float(df[feat].std()),
                "min": float(df[feat].min()),
                "max": float(df[feat].max()),
            }
            results["feature_stats"][feat] = stats
            
            if verbose:
                print(f"{feat:15}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
                      f"range=[{stats['min']:.3f}, {stats['max']:.3f}]")
        elif verbose:
            print(f"{feat:15}: ❌ 缺失")
    
    # 4. 模型預測分佈（終極驗證）
    if model and scaler and FEATURES:
        try:
            # 確保特徵順序一致
            X = df[FEATURES].fillna(0)
            if set(X.columns) != set(FEATURES):
                X = X[[f for f in FEATURES if f in X.columns]]
            
            X_scaled = scaler.transform(X)
            proba = model.predict_proba(X_scaled)
            
            # 找到類別1的索引
            classes = list(getattr(model, "classes_", []))
            idx_strong = classes.index(1) if 1 in classes else 0
            proba_strong = proba[:, idx_strong]
            
            proba_stats = {
                "mean": float(np.mean(proba_strong)),
                "std": float(np.std(proba_strong)),
                "min": float(np.min(proba_strong)),
                "max": float(np.max(proba_strong)),
                "above_055": float(np.mean(proba_strong > 0.55)),
                "above_060": float(np.mean(proba_strong > 0.60)),
                "above_040": float(np.mean(proba_strong > 0.40)),
            }
            results["proba_stats"] = proba_stats
            
            if verbose:
                print(f"\n=== proba_strong 終極診斷 ===")
                print(f"平均值: {proba_stats['mean']:.3f}")
                print(f"標準差: {proba_stats['std']:.3f}")
                print(f"範圍: [{proba_stats['min']:.3f}, {proba_stats['max']:.3f}]")
                print(f">0.55比例: {proba_stats['above_055']:.1%}")
                print(f">0.60比例: {proba_stats['above_060']:.1%}")
                print(f">0.40比例: {proba_stats['above_040']:.1%}")
            
            # 診斷問題
            if proba_stats["max"] < 0.45:
                if verbose:
                    print(f"\n🚨 致命錯誤：proba全<0.45，特徵工程徹底失敗！")
                issues.append("proba_strong全<0.45")
            elif proba_stats["max"] < 0.55:
                if verbose:
                    print(f"\n⚠️ 警告：proba全<0.55，交易信號過少")
                issues.append("proba_strong全<0.55")
            elif proba_stats["above_055"] < 0.01:
                if verbose:
                    print(f"\n⚠️ 警告：>0.55比例過低（{proba_stats['above_055']:.1%}）")
                issues.append(f">0.55比例過低: {proba_stats['above_055']:.1%}")
            else:
                if verbose:
                    print(f"\n✅ 特徵正常，模型預測合理！")
        except Exception as e:
            if verbose:
                print(f"\n❌ 模型預測失敗：{str(e)}")
            issues.append(f"模型預測失敗: {str(e)}")
    else:
        if verbose:
            print(f"\n⚠️ 無法執行模型預測（缺少 model/scaler/features）")
        issues.append("無法執行模型預測")
    
    results["issues"] = issues
    
    if verbose:
        print("\n" + "=" * 60)
        if issues:
            print(f"發現 {len(issues)} 個問題：")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("✅ 未發現問題，特徵工程正常！")
        print("=" * 60)
    
    return results
