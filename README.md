## 專案簡介

**MMBP**（後端專案名稱在 `pyproject.toml` 中為 `make-money`）是一個針對 **股票投資決策支援** 所設計的 **機器學習＋量化分析後端服務**。  
它主要提供：

- **股票歷史資料抓取與特徵工程**（透過 `yfinance` 及自訂技術指標計算）。
- **機器學習模型推論**（使用已訓練好的 `scikit-learn` 模型檔）。
- **系統化投資評分與建議**（結合技術面指標、基本面評分、系統評分）。
- **REST API / Streaming API 介面**，供前端（例如 `FMFA` React 前端）呼叫：
  - 取得歷史股價與指標資料。
  - 取得最新的投資建議與各類別機率。
  - 透過 LLM（Ollama）進行投資分析對話（SSE 串流）。

整體來說，**MMBP** 扮演「智慧投資決策引擎」的角色，前端僅負責呈現與互動，而所有資料處理、模型推論與評分邏輯皆集中在此後端服務中。

### 模型與策略版本

為符合模型風險管理與可驗證、可追蹤之需求，系統為模型與評分／策略邏輯設定版本號與上線日期。目前使用之 **模型版本**、**策略版本** 與 **上線日期** 由 `config/versioning.py` 定義，並可透過環境變數 `MODEL_VERSION`、`STRATEGY_VERSION`、`MODEL_EFFECTIVE_DATE` 覆寫。預測與回測 API 回應中會帶入 `model_version`、`strategy_version`、`model_effective_date`；另提供 `GET /api/v1/model-info` 查詢目前版本。完整說明（含訓練區間、主要假設與變更日誌）見 [MODEL_VERSION.md](MODEL_VERSION.md)。

---

## 核心功能與架構概觀

### 1. API 服務入口 (`main.py`)

- 使用 **FastAPI** 作為 Web Framework。
- 在 `lifespan` 生命周期中完成：
  - 設定載入 (`ConfigLoader`)：包含模型路徑與審計目錄／保留天數等設定。
  - Logger 初始化 (`LoggerManager`)。
  - 模型載入器 (`ModelLoader`) 建立。
  - 資料服務 (`DataService`) 與預測服務 (`PredictionService`) 建立。
  - 審計日誌啟動時清理：刪除超過 `AUDIT_RETENTION_DAYS` 的舊審計檔。
  - 將上述實例掛載到 `app.state`，供路由層方便存取。
- 設定 FastAPI 應用：
  - `title="make_money API"`
  - `version="0.1.0"`
  - 掛載審計中介層 (`AuditMiddleware`)：對關鍵 API 記錄 client_ip、X-Client-Id、參數與結果摘要至 JSONL。
  - 掛載 API router：`app.include_router(api_router)`。

### 2. 路由層 (`api/routes.py`)

提供對外的 HTTP API，包括：

- **健康檢查**

  - `GET /api/v1/health`
  - 回傳 `HealthResponse`，可供前端或監控系統確認服務是否正常運作。

- **模型／策略版本**

  - `GET /api/v1/model-info`
  - 回傳目前 `model_version`、`strategy_version`、`model_effective_date`，以及可選的 `training_interval`、`assumptions`，供稽核與前端顯示「目前使用版本」。

- **LLM 投資助理對話（SSE 串流）**

  - `POST /api/v1/chat/stream`
  - Request Body：`ChatStreamRequest`
    - `symbol`: 股票代碼。
    - `message`: 使用者提問。
    - `context`: 前端整理好的技術指標／數據摘要文字。
  - 行為：
    - 透過 `OllamaLoader` 與本地/遠端 Ollama 服務交互。
    - 使用系統提示詞（繁體中文，強調風險與不保證獲利）。
    - 以 **Server-Sent Events (SSE)** 格式，一段一段地把 LLM 回覆串流回前端。
  - 回應：
    - `text/event-stream`，每段以 `data: ...` 形式傳送，最後送出 `event: done` 代表結束。

- **取得最新預測與投資建議**

  - `POST /api/v1/predict`
  - Request Body：`PredictionRequest`
    - e.g. `symbol`, `period`（預設 `"10y"`）。
  - 行為：
    - 從 `PredictionService` 取得最新一筆資料的預測結果：
      - 類別機率分布
      - 系統評分 / 技術面評分 / 基本面評分
      - 買進機率 `proba_buy`
      - 建議文字（例如：`"長期持有"`, `"觀望"`, `"不建議持有"`）
  - 回應：`PredictionResponse`（含 `model_version`、`strategy_version`、`model_effective_date`）

- **取得歷史股價資料**

  - `GET /api/v1/stock/{symbol}/data?period=10y`
  - 行為：
    - 透過 `DataService.fetch_stock_data` 從 `yfinance` 抓取歷史資料。
    - 包括：`date, open, high, low, close, volume` 等欄位。
  - 回應：`StockDataResponse`
    - 包含資料列數 `rows` 與序列化後的 `data`。

- **取得含指標的歷史資料**

  - `GET /api/v1/stock/{symbol}/indicators?period=10y`
  - 行為：
    - 先抓原始股價資料，再使用 `DataService.add_indicators` 加入技術指標。
  - 回應：`IndicatorsResponse`
    - 包含所有技術指標欄位（MA、RSI、EMA、報酬率等）。

- **單資產回測（含波動率目標倉位）**

  - `GET /api/v1/backtest?symbol=2330.TW&start=2020-01-01&end=2024-12-31`
  - 可選查詢參數：
    - `use_vol_targeting=true`（預設）：啟用波動率目標倉位，使策略波動貼近目標。
    - `target_vol_annual=0.10`：目標年化波動率（如 10%）。
    - `vol_lookback=20`：波動率估計滾動天數。
    - `max_leverage=1.0`：單一資產最大槓桿。
  - 行為：依模型訊號與 2 ATR 移動停損產生倉位，再依 ex-ante 波動率動態調整權重（Volatility Targeting），利於風控與客戶溝通。
  - 回應：`BacktestResponse`（年化報酬、波動率、最大回撤、夏普、權益曲線等；含 `model_version`、`strategy_version`、`model_effective_date`）。

- **多資產組合回測**

  - `POST /api/v1/backtest/portfolio`
  - Request Body：`PortfolioBacktestRequest`
    - `symbols`: 標的代碼列表（如 `["2330.TW", "2454.TW"]`）。
    - `start` / `end`：可選日期區間。
    - `target_vol_annual`、`vol_lookback`、`max_leverage`：單資產波動率目標參數。
    - `target_portfolio_vol_annual`：組合層級目標年化波動率。
    - `max_single_weight`：單一資產權重上限（如 0.40）。
  - 行為：各標的先做波動率目標倉位回測，再以逆波動率權重（等風險貢獻風格）聚合，並可選組合層級波動率目標與單一資產曝險上限。
  - 回應：`BacktestResponse`（`symbol="PORTFOLIO"`；含 `model_version`、`strategy_version`、`model_effective_date`）。

路由內輔助函式：

- `get_prediction_service(request)` / `get_data_service(request)`：
  - 從 `request.app.state` 安全取得對應 Service，如未初始化則拋出錯誤。
- `_df_to_records(df)`：
  - 將 `pandas.DataFrame` 轉為 list[dict]，並把 `NaN/NaT` 轉成 `None`，方便 JSON 序列化。
- `_sse_pack_data` / `_sse_event`：
  - 將文字打包成符合 SSE 規範的字串格式。

### 3. 資料服務 (`services/data_service.py`)

`DataService` 負責 **資料下載與技術指標計算 (Feature Engineering)**：

- **`FeatureConfig`**
  - `ma_windows`: `(5, 20, 60, 120, 240)` → 用於計算不同期間移動平均線。
  - `rsi_windows`: `(120, 240, 420)` → RSI 指標視窗長度。
  - `ema_spans`: `(120, 240, 420, 200)` → 指數移動平均線 span。

- **`fetch_stock_data(symbol, period="10y")`**
  - 使用 `yfinance.download(symbol, period=period)` 取得歷史資料。
  - 處理多重欄位索引（MultiIndex）情況，只保留第一層欄位名稱。
  - 轉換欄位名稱為：
    - `Open` → `open`
    - `High` → `high`
    - `Low` → `low`
    - `Close` → `close`
    - `Volume` → `volume`
  - 新增 `date` 欄位（由 `Date` 轉換而來），並依日期排序。

- **`add_indicators(df)`**
  - 在原始 DataFrame 上加入各種技術指標：
    - 移動平均：`ma5`, `ma20`, `ma60`, `ma120`, `ma240`。
    - 報酬率：`return_1`, `return_5`。
    - RSI：`rsi_120`, `rsi_240`, `rsi_420`（透過 `train_model.train_model第五版.compute_rsi`）。
    - EMA：`ema120`, `ema240`, `ema420`, `ema200`。

- **`required_features()`**
  - 回傳模型輸入所需特徵欄位名稱列表，包括：
    - 價格與成交量（`open`, `high`, `low`, `close`, `volume`）
    - 所有 MA / RSI / EMA 欄位
    - 報酬率欄位
    - `fund_score`（基本面分數，由其他模組補上）

### 4. 預測服務 (`services/prediction_service.py`)

`PredictionService` 封裝了從資料到最終投資建議的整個流程：

- **流程步驟**
  1. 使用 `ModelLoader` 載入已訓練的模型（例如 `rf_model_2330.pkl`）。
  2. 呼叫 `DataService.fetch_stock_data` 取得歷史股價。
  3. 呼叫 `DataService.add_indicators` 計算技術指標。
  4. 透過 `get_latest_fund_score(symbol)` 取得基本面分數，寫入 `df["fund_score"]`。
  5. 取得模型需要的所有特徵欄位 `features = required_features()`。
  6. 取最新一筆資料 `df_latest = df.iloc[-1:]` 並檢查是否有缺失值。
  7. 呼叫模型的 `predict_proba(X)` 計算各類別機率。
  8. 使用 `LABEL_MAP` 將類別數值（0/1/2）轉成人類可讀的標籤：
     - `0: "不建議持有"`
     - `1: "長期持有"`
     - `2: "觀望"`
  9. 確保模型輸出包含類別 `1`，以計算 `proba_buy`（長期持有機率）。
  10. 將 `proba_buy` 與其他欄位送入 `system_rating`，計算：
      - `system_score`
      - `tech_score`
      - `fund_score`
      - 最終建議文字 `recommendation`

- **回傳型別 `PredictionResult`**
  - `symbol`: 股票代碼。
  - `probabilities`: 各建議類別對應機率字典。
  - `system_score`: 綜合系統分數。
  - `tech_score`: 技術面評分。
  - `fund_score`: 基本面評分。
  - `proba_buy`: 長期持有機率。
  - `recommendation`: 最終文字建議。

### 5. 模型載入與設定 (`models/model_loader.py`, `config/config_loader.py`)

- **`ConfigLoader` / `AppConfig`**
  - 從環境變數或預設值載入模型路徑與審計設定：
    - 模型路徑：環境變數 `MODEL_PATH`，預設專案根目錄下 `rf_model_2330.pkl`
    - 審計目錄：`AUDIT_LOG_DIR`，預設 `logs/audit`
    - 審計保留天數：`AUDIT_RETENTION_DAYS`，預設 180
    - API Key 雜湊用密鑰：`AUDIT_HMAC_SECRET`（可選）
  - 統一集中管理設定，避免程式中多處硬編路徑。

- **`ModelLoader`**
  - 接收 `ModelLoadConfig(model_path=...)`。
  - 內建快取機制：第一次載入後會把模型存在記憶體，後續呼叫 `load()` 不需重新讀檔。
  - 若模型檔不存在，會拋出 `FileNotFoundError`。

### 6. LLM 投資助理 (`models/ollama_loader.py`)

雖然此檔案未在此完整展示，但從 `routes.py` 可看出：

- 使用 `OllamaLoader` 與本機或遠端的 Ollama Server 連線。
- 透過 `stream_chat` 取得逐段 LLM 回覆（適合 SSE 串流）。
- 系統提示詞強調：
  - 使用繁體中文。
  - 針對技術指標與數據進行分析。
  - 加入風險提示，避免保證獲利的用語。

### 7. 審計日誌 (Audit log)

為滿足法遵與內稽「可追溯、可舉證」需求，關鍵 API 呼叫會寫入審計日誌：

- **納入審計的 API**
  - `POST /api/v1/predict`
  - `GET /api/v1/backtest`
  - `POST /api/v1/backtest/portfolio`
  - `GET /api/v1/stock/{symbol}/indicators`
  - `GET /api/v1/stock/{symbol}/data`

- **記錄內容（不存完整 request/response）**
  - 時間（UTC ISO8601）、`request_id`、來源 IP（`client_ip`）、`X-Client-Id`（`client_id`）、`X-Api-Key` 雜湊（若有）。
  - 方法、路徑、查詢/body 參數摘要、`symbol`/`symbols`。
  - 回應狀態碼、延遲（ms）、結果摘要（如 `recommendation`、`system_score`、`annualized_return`、`rows` 等），失敗時含 `error_type`、`error_message`。

- **寫入位置與格式**
  - 目錄：由環境變數 `AUDIT_LOG_DIR` 指定，預設為專案根目錄下 `logs/audit`。
  - 檔名：`audit-YYYY-MM-DD.jsonl`（每日一檔，JSON Lines，每行一筆 JSON）。

- **保留期限與清理**
  - 由 `AUDIT_RETENTION_DAYS` 指定（預設 180 天）。服務啟動時會自動刪除超過保留天數的舊檔。

- **存取權限建議**
  - 寫入時新檔權限為 `0600`（僅 owner 可讀寫）。建議僅部署主機上的 ops／稽核帳號可讀取審計目錄，以利風控／法遵審查。

- **環境變數（見下方「設定環境變數與模型檔」）**
  - `AUDIT_LOG_DIR`、`AUDIT_RETENTION_DAYS`、`AUDIT_HMAC_SECRET`（用於 API Key 雜湊，不存明文）。

---

## 技術棧與相依套件

根據 `pyproject.toml`，主要技術與依賴如下：

- **語言與執行環境**
  - Python `>= 3.10`

- **Web Framework**
  - `fastapi[standard] >= 0.128.0`

- **HTTP / 非同步工具**
  - `httpx >= 0.28.1`

- **資料處理與科學運算**
  - `numpy >= 2.2.6`
  - `pandas >= 2.3.3`

- **機器學習與模型序列化**
  - `scikit-learn >= 1.7.2`
  - `joblib >= 1.5.3`

- **金融資料來源**
  - `yfinance >= 1.0`

- **視覺化（主要用於訓練腳本 / 分析）**
  - `matplotlib >= 3.10.8`

其他模組（未在 `pyproject` 中列出，但專案中使用）：

- `python-dotenv`：在 `model_loader.py`、`config_loader.py` 中透過 `load_dotenv()` 載入 `.env`。

---

## 環境需求與安裝

### 1. Python 與虛擬環境

- 建議使用 Python `3.10` ~ `3.13`。
- 建議在專案根目錄下建立虛擬環境，避免套件與其他專案衝突。

以 `python3` 為例：

```bash
git clone https://github.com/huangbigice/MMBP.git
cd /MMBP

python3 -m venv .venv
source .venv/bin/activate  # macOS / Linux
# 若為 Windows (PowerShell)
# .venv\Scripts\Activate.ps1
```
以 `uv` 為例：

```bash
git clone https://github.com/huangbigice/MMBP.git
cd /MMBP

uv venv --python 3.10.6
source .venv/bin/activate  # macOS / Linux
# 若為 Windows (PowerShell)
# .venv\Scripts\Activate.ps1
```


> 專案中已包含 `.venv/` 目錄，但建議在你自己的環境重新建立，以確保與本機 Python 版本相容，只需刪除 `.venv` 即可。

### 2. 安裝依賴套件

專案使用 `pyproject.toml` 管理依賴，可使用 `pip` or `uv` 直接安裝：

```bash
pip install -U pip
pip install -e .
```
or

```bash
uv sync
```
若你只想快速安裝主要依賴，也可以：

```bash
pip install fastapi[standard] httpx joblib matplotlib numpy pandas scikit-learn yfinance python-dotenv
```

（建議以 `pip install -e .` 為主，確保與 `pyproject.toml` 同步。）

### 3. 設定環境變數與模型檔

在專案根目錄建立 `.env` 檔（若尚未設定），至少需指定：

```env
MODEL_PATH=/absolute/path/to/your_model.pkl
```

可選的審計日誌相關變數（法遵／內稽用）：

```env
AUDIT_LOG_DIR=/path/to/logs/audit   # 審計 JSONL 目錄，預設為專案根目錄下 logs/audit
AUDIT_RETENTION_DAYS=180            # 審計檔保留天數，預設 180
AUDIT_HMAC_SECRET=your_secret       # 用於 X-Api-Key 雜湊（HMAC-SHA256），不存明文；可選
```

說明：

- 若未設定 `MODEL_PATH`，系統會預設尋找專案根目錄下的 `rf_model_2330.pkl`：
  - 由 `ConfigLoader` 中的：
    - `default_model_path = Path(__file__).resolve().parents[1] / "rf_model_2330.pkl"`
  - 若無此檔案則在啟動預測時會噴出 `FileNotFoundError`。
- 建議把正式要使用的模型檔（例如用特定股票或多股票訓練好的 `pkl` 檔）放到專案根目錄或指定路徑，並在 `.env` 指向。

> 若你同時有訓練腳本（位於 `train_model/` ），請先使用該腳本訓練並輸出模型，再把輸出檔案路徑填入 `MODEL_PATH`。

---

## 啟動方式

### 1. 使用 FastAPI 官方建議指令（開發環境）

專案中終端目前顯示有執行：

```bash
fastapi dev main.py
```

此指令會：

- 啟動開發伺服器（預設 `http://127.0.0.1:8000`，實際請看終端輸出）。
- 自動偵測程式變更並重新載入（類似 `uvicorn --reload`）。

### 2. 透過 `uvicorn` 啟動

你也可以改用 `uvicorn` 直接啟動：

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

說明：

- `main:app`：指向 `main.py` 中的 `app = FastAPI(...)` 實例。
- `--reload`：程式變更時自動重新啟動（適用開發環境）。
- `--host 0.0.0.0`：允許外部機器透過 IP 訪問（例如前端在同一局域網上）。

### 3. 檢查 API 是否啟動成功

啟動後可在瀏覽器或 `curl` 測試：

- 健康檢查：

  ```bash
  curl http://localhost:8000/api/v1/health
  ```

- 取得歷史資料：

  ```bash
  curl "http://localhost:8000/api/v1/stock/2330/data?period=1y"
  ```

- 取得預測結果：

  ```bash
  curl -X POST "http://localhost:8000/api/v1/predict" \
    -H "Content-Type: application/json" \
    -d '{"symbol": "2330.TW", "period": "10y"}'
  ```

若收到合法 JSON 回應，代表後端服務啟動並正常運作。

---

## 與前端（FMFA）整合方式

典型的整合方式如下：

- **前端設定 API Base URL**
  - 在前端專案（`FMFA`）的 `.env.development` 等檔案中設定：

    ```env
    VITE_API_BASE_URL=http://localhost:8000
    ```

- **前端呼叫範例**
  - 取得含指標的股價資料：
    - `GET {VITE_API_BASE_URL}/api/v1/stock/{symbol}/indicators?period=10y`
  - 取得最新預測與建議：
    - `POST {VITE_API_BASE_URL}/api/v1/predict`
  - 啟用 LLM 串流聊天：
    - `POST {VITE_API_BASE_URL}/api/v1/chat/stream`，並以 `EventSource` / `fetch + ReadableStream` 處理 SSE。

- **錯誤處理建議**
  - 若 API 回傳 400/500，前端可顯示對應訊息：
    - 400：多為輸入參數或資料不足（例如特徵缺失）。
    - 500：多為內部錯誤（例如模型檔不存在、外部服務異常）。

---

## 目錄結構重點說明

專案根目錄下重要檔案與資料夾：

- `main.py`：FastAPI 服務入口。
- `api/`：
  - `routes.py`：所有對外 API 路由。
  - `schemas.py`：Pydantic 模型定義（請參考實際檔案結構）。
- `services/`：
  - `data_service.py`：資料抓取與技術指標計算。
  - `prediction_service.py`：完整預測與系統評分流程。
  - `backtest_service.py`：單資產／組合回測，含波動率目標倉位與多資產風控。
  - `risk/`：風險與倉位模組。
    - `position_sizing.py`：波動率目標倉位（Volatility Targeting）。
    - `portfolio.py`：多資產對齊、逆波動率權重、組合層級波動率目標。
- `models/`：
  - `model_loader.py`：模型載入與快取。
  - `ollama_loader.py`：LLM 互動封裝。
- `config/`：
  - `config_loader.py`：集中管理設定值，如模型路徑。
- `train_model/`：
  - `train_model第五版.py` 等：模型訓練與指標計算程式，供離線訓練使用。
- `system_rating.py`：系統評分邏輯（整合技術面與基本面）。
- `fundamentals.py` / `news.py` / `stock_code.py` 等：
  - 與基本面、新聞、或股票代碼處理相關模組（視實際內容而定）。
- `pyproject.toml`：專案設定與依賴。
- `uv.lock`：與 `uv` / `pip` 鎖定檔有關，記錄安裝狀態。

---

## 開發建議與注意事項

- **資料品質**
  - 由於依賴 `yfinance`，若遇到個股資料缺漏或停牌，可能導致指標計算結果有缺失值。
  - `PredictionService` 已對最新一筆資料的缺失值進行檢查，若有缺失則會拋出 `ValueError`。

- **模型版本管理**
  - 建議為不同訓練版本的模型使用清楚命名，例如：
    - `rf_model_2330_v1.pkl`
    - `rf_model_multi-symbol_v2.pkl`
  - 並在 `.env` 中指定正確路徑，必要時加上版本註記。

- **日誌與監控**
  - `LoggerManager`（位於 `logger/logger_manager.py`）負責統一管理日誌設定。
  - 啟動與關閉時會記錄：
    - `"=== make_money API starting ==="`
    - `"=== make_money API shutdown ==="`
  - 建議在關鍵錯誤處加入適量 log 以便追蹤。

- **安全性**
  - 若未來對外公開服務，建議：
    - 加入 API Key / JWT 等認證機制。
    - 在反向代理層（如 Nginx）設定 HTTPS。

---

## 常見問題 (FAQ)

### Q1. 啟動時出現 `Model file not found`？

- 檢查項目：
  - `.env` 中的 `MODEL_PATH` 是否正確。
  - 指定路徑下是否真的存在該 `.pkl` 模型檔。
  - 檔案權限是否允許讀取。

### Q2. `predict` API 回傳 400，訊息為「最新一天特徵有缺失值」？

- 可能原因：
  - 最近一筆股價資料在某些指標上為 `NaN`，例如：
    - 資料長度不足以計算長期均線或 RSI。
  - 該股票近期剛上市或長時間停牌。
- 建議作法：
  - 調整 `period` 或指標視窗長度。
  - 在前端提示使用者換用資料較完整的股票或拉長觀察區間。

### Q3. `yfinance` 抓不到特定股票資料？

- 檢查：
  - 股票代碼是否需要加市場後綴（例如台股常為 `.TW`）。
  - 手動在 Yahoo Finance 網站確認該代碼是否有效。

### Q4. SSE 串流聊天前端收不到資料？

- 檢查：
  - 後端是否有正確啟動 Ovllama / LLM 服務。
  - 前端是否有使用 `EventSource` 或正確處理 `text/event-stream`。
  - 伺服器或代理層是否有對 SSE 做特殊設定（例如超時或壓縮）。

---

## 總結

**MMBP / make-money** 是此投資系統中的 **後端決策與分析核心**，負責整合：

- 金融資料抓取與技術指標運算。
- 機器學習模型預測。
- 系統化評分與建議產生。
- LLM 投資助理對話。

只要依照本文件：

1. 建立虛擬環境並安裝依賴。  
2. 準備並設定模型檔與 `.env`。  
3. 啟動 FastAPI 服務並接上前端。  

即可快速在本機或伺服器上啟動一套完整的智慧股票投資分析後端。  
若你需要，我也可以協助撰寫更詳細的 API 文件（例如 OpenAPI 註解補強或獨立的 API Spec 說明）。 

