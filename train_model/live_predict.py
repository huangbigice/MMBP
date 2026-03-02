"""
③ live_predict.py（依模型修改指南）

負責：今天資料、產出明日預測；不計算任何未來報酬。
"""

import sys
from pathlib import Path

# 確保可從同目錄載入 train
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import pandas as pd

from train import FEATURES, MODEL_PATH, get_today_features


def load_model_artifacts():
    """載入模型、scaler、特徵清單。"""
    data = joblib.load(MODEL_PATH)
    return data["model"], data["scaler"], data["features"]


def main(stock_list_path="stock_list.json"):
    model, scaler, features = load_model_artifacts()
    if features != FEATURES:
        features = list(features)

    today_df, last_date = get_today_features(stock_list_path=stock_list_path)
    if len(today_df) == 0:
        print("無今日特徵資料，請檢查 stock_list 與網路")
        return

    X = scaler.transform(today_df[FEATURES].fillna(0))
    proba = model.predict_proba(X)
    pred = model.predict(X)

    today_df = today_df.copy()
    today_df["proba_strong"] = proba[:, 1]
    today_df["proba_weak"] = proba[:, 0]
    today_df["pred"] = pred

    # 明日預測：以今日特徵預測的是「明日」相對強弱
    print("=== 明日預測（基於", last_date.strftime("%Y-%m-%d"), "資料）===")
    print("（不計算任何未來報酬）\n")

    out = today_df[["ticker", "industry", "proba_strong", "proba_weak", "pred"]].copy()
    out["強弱"] = out["pred"].map({0: "弱勢", 1: "強勢", 2: "趨勢不明"})
    out = out.sort_values("proba_strong", ascending=False).reset_index(drop=True)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
