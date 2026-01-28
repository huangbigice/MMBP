from config.config_loader import ConfigLoader
from db.db_repository import DuckdbRepository
from data.data_fetcher import DataFetcher
from indicator.indicator_processor import KBarProcessor
from reportation.report_generator import ReportGenerator
from logger.logger_manager import LoggerManager 
from datetime import datetime, timedelta
def main():
    # 1. 初始化資源
    config = ConfigLoader()
    logger = LoggerManager().get_logger()
    logger.info("=== 交易系統啟動 ===")

    # 連線至資料庫並初始化資料庫Table
    db = DuckdbRepository(config = config, logger = logger)
    
    # 測試用資料
    stock_code = "2330"
    base_date_str = "2025-12-31"
    base_date_dt = datetime.strptime(base_date_str, "%Y-%m-%d")
    start_date = (base_date_dt - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = base_date_str
    
    try:
        # 2. 取得原始資料Open,High,Low,Close,Volume
        datafetcher = DataFetcher(config = config,logger = logger,db = db)
        raw_data = datafetcher.main(stock_code, start_date, end_date)
        db.save_dataframe(raw_data) # 回存至資料庫
        
        # # 3. 計算技術指標
        # logger.info("開始執行數據處理程序...")
        # processor = KBarProcessor(config = config, logger = logger,db = db)
        # processor.run()

        # # 4. 產出報告
        # logger.info("開始產出分析報告...")
        # reporter = ReportGenerator(config = config, logger = logger,db = db)
        # reporter.export()
        
        # logger.info("=== 任務執行成功完成 ===")

    except Exception as e:
        # 捕捉任何執行中的錯誤並記錄
        logger.error(f"系統執行過程中發生未預期錯誤: {str(e)}", exc_info=True)

    finally:
        # 確保資料庫連線被關閉（如果 Repository 有實作 close 方法）
        if hasattr(db, 'close'):
            db.close()
            logger.info("資料庫連線已安全關閉。")

if __name__ == "__main__":
    main()