"""
Walk-forward 滾動重訓：依時間視窗切資料、訓練、存檔。

每 3 個月用最近 3~5 年資料重訓（adaptive learner），可覆寫單一模型檔或產出多版本。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from train_model.market_regime import MKT_REGIME_FEATURES


# 與 train_model第五版 一致的特徵列表（個股技術 + 基本面 + 大盤 regime）
TRAIN_FEATURES = [
    "open", "high", "low", "close", "volume",
    "ma5", "ma20", "ma60", "ma120", "ma240",
    "return_1", "return_5",
    "rsi_120", "rsi_240", "rsi_420",
    "ema120", "ema240", "ema420", "ema200",
    "fund_score",
    *MKT_REGIME_FEATURES,
]


@dataclass
class TrainWindow:
    """單一訓練視窗的起訖日。"""
    train_start: str
    train_end: str


def generate_training_windows(
    data_start: str,
    data_end: str,
    train_years: int = 5,
    roll_months: int = 3,
) -> list[TrainWindow]:
    """
    產生滾動訓練視窗：每個視窗為 train_years 年，每隔 roll_months 月往前滾一次。

    Parameters
    ----------
    data_start : str
        資料區間起日 (YYYY-MM-DD)。
    data_end : str
        資料區間迄日 (YYYY-MM-DD)。
    train_years : int
        每窗訓練年數。
    roll_months : int
        滾動間隔月數。

    Returns
    -------
    list[TrainWindow]
        視窗列表；每個視窗的 train_end 從 data_end 往前推。
    """
    end_d = pd.Timestamp(data_end)
    start_d = pd.Timestamp(data_start)
    windows: list[TrainWindow] = []
    # 從資料迄日往前滾，每次往前 roll_months 月
    current_end = end_d
    while True:
        current_start = current_end - pd.DateOffset(years=train_years)
        if current_start < start_d:
            break
        windows.append(
            TrainWindow(
                train_start=current_start.strftime("%Y-%m-%d"),
                train_end=current_end.strftime("%Y-%m-%d"),
            )
        )
        current_end = current_end - pd.DateOffset(months=roll_months)
    return windows


def train_one_window(
    panel_df: pd.DataFrame,
    train_start: str,
    train_end: str,
    model_path: str | Path,
    *,
    features: list[str] | None = None,
    n_estimators: int = 400,
    max_depth: int = 10,
    random_state: int = 42,
    class_weight: str = "balanced",
) -> RandomForestClassifier:
    """
    在單一時間視窗內切訓練集、擬合 RF、存檔。

    Parameters
    ----------
    panel_df : pd.DataFrame
        已含 label 與 TRAIN_FEATURES 的 panel（如 model_input_multi.csv）。
    train_start : str
        訓練區間起日 (YYYY-MM-DD)。
    train_end : str
        訓練區間迄日 (YYYY-MM-DD)，含當日。
    model_path : str | Path
        模型存檔路徑。
    features : list[str] | None
        特徵欄位；None 時用 TRAIN_FEATURES。
    n_estimators, max_depth, random_state, class_weight
        RandomForestClassifier 參數。

    Returns
    -------
    RandomForestClassifier
        擬合後的模型。
    """
    feat = features or TRAIN_FEATURES
    panel_df = panel_df.copy()
    panel_df["date"] = pd.to_datetime(panel_df["date"])
    mask = (panel_df["date"] >= train_start) & (panel_df["date"] <= train_end)
    train = panel_df.loc[mask].dropna(subset=feat + ["label"])
    if train.empty:
        raise ValueError(
            f"視窗 {train_start} ~ {train_end} 內無有效訓練樣本，請檢查資料或區間。"
        )

    X = train[feat]
    y = train["label"]
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight=class_weight,
    )
    model.fit(X, y)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model


def run_latest_window(
    csv_path: str | Path,
    model_path: str | Path,
    train_years: int = 5,
    **rf_kwargs: object,
) -> RandomForestClassifier:
    """
    用「最近 train_years 年」資料訓練一次並覆寫模型檔。
    適合排程每 3 個月執行一次。

    Parameters
    ----------
    csv_path : str | Path
        model_input_multi.csv 路徑（需已含相對標籤與 regime 特徵）。
    model_path : str | Path
        輸出的模型路徑（如 rf_model_multi.pkl）。
    train_years : int
        使用最近幾年資料訓練。
    **rf_kwargs
        傳給 train_one_window 的 RF 參數。

    Returns
    -------
    RandomForestClassifier
        擬合後的模型。
    """
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    data_end = df["date"].max().strftime("%Y-%m-%d")
    data_start = (df["date"].max() - pd.DateOffset(years=train_years)).strftime("%Y-%m-%d")
    return train_one_window(
        df,
        train_start=data_start,
        train_end=data_end,
        model_path=model_path,
        **rf_kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward 滾動重訓：依視窗訓練並存檔。"
    )
    parser.add_argument(
        "--csv",
        default="model_input_multi.csv",
        help="Panel CSV 路徑（需含 label 與 regime 特徵）",
    )
    parser.add_argument(
        "--model",
        default="rf_model_multi.pkl",
        help="輸出的模型路徑（--single 時覆寫此檔）",
    )
    parser.add_argument(
        "--train-years",
        type=int,
        default=5,
        help="每窗訓練年數",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="僅用最近 train-years 年訓練一次並覆寫 --model",
    )
    parser.add_argument(
        "--roll-months",
        type=int,
        default=3,
        help="多視窗時滾動間隔月數（僅 --all 時使用）",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="產出多個視窗模型（檔名加 _YYYYMM 後綴）",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"找不到 {csv_path}，請先執行 train_model第五版 產出含相對標籤與 regime 的 CSV。"
        )

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    data_start = df["date"].min().strftime("%Y-%m-%d")
    data_end = df["date"].max().strftime("%Y-%m-%d")

    if args.single:
        run_latest_window(
            csv_path=csv_path,
            model_path=args.model,
            train_years=args.train_years,
        )
        print(f"已以最近 {args.train_years} 年資料訓練並存檔：{args.model}")
        return

    if args.all:
        windows = generate_training_windows(
            data_start=data_start,
            data_end=data_end,
            train_years=args.train_years,
            roll_months=args.roll_months,
        )
        base = Path(args.model).stem
        parent = Path(args.model).parent
        for w in windows:
            suffix = w.train_end[:7].replace("-", "")  # YYYYMM
            out_path = parent / f"{base}_{suffix}.pkl"
            train_one_window(df, w.train_start, w.train_end, out_path)
            print(f"視窗 {w.train_start} ~ {w.train_end} → {out_path}")
        return

    # 預設：等同 --single
    run_latest_window(
        csv_path=csv_path,
        model_path=args.model,
        train_years=args.train_years,
    )
    print(f"已以最近 {args.train_years} 年資料訓練並存檔：{args.model}")


if __name__ == "__main__":
    main()
