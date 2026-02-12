# 模型與策略版本說明

本文件記載目前上線之模型與策略版本、訓練／驗證區間、主要假設與變更日誌，供模型風險管理與稽核對照。

---

## 當前版本

| 項目 | 版本／日期 |
|------|------------|
| **模型版本** (model_version) | 1.0.0 |
| **策略版本** (strategy_version) | 1.0.0 |
| **上線日期** (model_effective_date) | 2026-02-01 |

程式內單一來源：`config/versioning.py`。部署時可透過環境變數覆寫：`MODEL_VERSION`、`STRATEGY_VERSION`、`MODEL_EFFECTIVE_DATE`。

---

## 訓練／驗證區間

- **訓練資料**：依實際訓練腳本 `train_model/train_model第五版.py` 所使用之區間與標的為準（如 yfinance 歷史區間、標的列表）。
- **驗證**：依該腳本內之切分或驗證方式為準。
- 若已固定區間，可在此填寫例如：「訓練區間 2020-01-01 ～ 2025-12-31，標的見 `stock_list.json`。」

---

## 主要假設

- **模型**：Random Forest 三類標籤——不建議持有 / 長期持有 / 觀望；特徵集與 `DataService.required_features()` 一致。
- **評分邏輯**（`system_rating.py`）：系統權重——模型機率 0.50、基本面 0.25、技術面 0.25；門檻 THRESH_BUY=0.60、THRESH_HOLD=0.50。
- **回測**（`services/backtest_service.py`）：訊號由模型＋基本面門檻產生；倉位為 2 ATR 移動停損；可選波動率目標倉位；手續費 0.15%；無滑價假設。

---

## 變更日誌

以日期由新到舊排列。

| 日期 | 版本（模型 / 策略） | 變更說明 |
|------|--------------------|----------|
| 2026-02-01 | 1.0.0 / 1.0.0 | 初版上線：模型與策略版本管理上線；API 回應帶入 model_version、strategy_version、model_effective_date；新增 `GET /api/v1/model-info` 與本說明文件。 |

---

*完整 API 說明見 [README.md](README.md)。*
