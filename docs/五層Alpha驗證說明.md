# 五層 Alpha 驗證說明

本專案實作五層驗證，用於檢驗策略是否為「統計幻覺」或僅是 beta，並要求風險調整後報酬需打贏簡單基準。

## 執行方式

在專案根目錄、已安裝依賴的環境下：

```bash
uv run five_layer_tests.py
```

可選參數：

- `--symbol 2330.TW`：標的
- `--start 2018-01-01`：回測區間起日
- `--end 2025-12-31`：回測區間迄日
- `--wf-start 2010-01-01`：Walk-forward 整體區間起日（第二層）
- `--no-plot`：不產出 Layer 2 的 Sharpe 分佈圖

或只跑單層（在 Python 中）：

```python
from validation.five_layer_tests import (
    run_layer1_null_test,
    run_layer2_walk_forward_stability,
    run_layer3_regime_survival,
    run_layer4_benchmark_iq,
    run_layer5_economic_plausibility,
)
# 需先建好 BacktestService, DataService
run_layer1_null_test(backtest_service, symbol="2330.TW", start="2018-01-01", end="2025-12-31")
```

---

## 第一層：Null Model 殺戮測試

- **目的**：確認真實模型不是「隨機」或「永遠觀望」的運氣。
- **做法**：同一回測、同一成本、同一風控下跑三種：
  1. **真實模型**：現有模型預測
  2. **打亂訊號**：將 `proba_buy` 打亂（等同標籤打亂的隨機訊號）
  3. **永遠觀望**：不進場（position 恆為 0）
- **通過條件**：真實模型的 Sharpe 與年化報酬需顯著優於 2 和 3；否則可視為統計幻覺。

---

## 第二層：Walk-forward 穩定性測試

- **目的**：檢查樣本外表現是否穩定，避免 overfit。
- **做法**：時間滾動切（例如 2010–2015 訓練 → 測 2016，2011–2016 → 測 2017 …），每段算一次 Sharpe，看**分佈**而非單一一條 equity curve。
- **通過條件**：Sharpe 的變異不宜過大；負 Sharpe 比例 < 50%。若超過一半時間為負，視為典型 overfit。

---

## 第三層：Regime 生存測試

- **目的**：確認策略在不同市場狀態下不會「一邊賺、一邊爆」。
- **做法**：依多頭/空頭（regime）、高波動/低波動（以 `mkt_vol_60d` 分位數切）切分，看各區間之年化報酬與最大回撤。
- **通過條件**：最差 regime 的最大回撤與年化報酬不可「爆炸」（例如最大回撤 < -35%、年化 < -20%）。若多頭賺、空頭爆，可能只是加槓桿的 beta。

---

## 第四層：Benchmark 智商測試

- **目的**：在風險調整後報酬上必須打贏簡單規則，否則經濟上無存在意義。
- **基準**：Buy & Hold、Momentum（252 日報酬>0）、MA crossover（ma20>ma60）、RSI(14)>50。同一區間、同一手續費。
- **通過條件**：真實策略的 Sharpe 不低於上述所有基準。

---

## 第五層：Economic Plausibility 自評

- **目的**：說清楚 alpha 的**經濟來源**。
- **做法**：填寫 `docs/economic_plausibility_填寫.md`，說明來源（行為偏誤、流動性溢價、風險補償、資訊延遲等）與可驗證方式。
- **通過條件**：若只能回答「模型學出來的」，在專業投資世界裡視為不存在；需人工判定。

---

## 程式位置

- 測試邏輯：`validation/five_layer_tests.py`
- 回測服務擴充（null_mode）：`services/backtest_service.py` 之 `run_backtest(..., null_mode=..., shuffle_seed=...)`、`get_backtest_df(...)`
- 執行腳本：`run_five_layer_tests.py`
- 經濟來源自評範本：`docs/economic_plausibility_填寫.md`
