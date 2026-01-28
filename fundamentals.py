import yfinance as yf
import pandas as pd

# 設定股票代號 (台積電為 2330.TW)
# yfinance 使用國際代碼，台灣股票需加上 .TW
ticker_symbol = "3481.TW"
stock = yf.Ticker(ticker_symbol)

# 獲取財務報表資料 (income_stmt 代表損益表)
# yfinance 抓到的資料通常是最近四季的季報或最近幾年的年報
financials = stock.financials 

# 獲取在外流通股數
# info 字典包含了許多公司資訊
shares_outstanding = stock.info.get('sharesOutstanding')

if not financials.empty and shares_outstanding is not None:
    # 取得最新一期的淨利 (通常在第一欄)
    # 'Net Income' 或 'Net Income Common Stockholders' 欄位名稱可能因公司而異
    try:
        net_income = financials.loc['Net Income'].iloc[0]
    except KeyError:
        # 嘗試另一種可能的欄位名稱
        net_income = financials.loc['Net Income Common Stockholders'].iloc[0]

    # 計算 EPS: 淨利 / 在外流通股數
    eps = net_income / shares_outstanding

    print(f"公司名稱: {stock.info['shortName']}")
    print(f"最新一期淨利: {net_income:,.2f}")
    print(f"在外流通股數: {shares_outstanding:,}")
    print(f"計算得出的 EPS (每股盈餘): **{eps:.2f}**")
else:
    print("無法獲取財務數據或股數資料。")

