import logging
from logging.handlers import TimedRotatingFileHandler
import os
import sys
import shutil
from datetime import datetime

# --- 全局配置 ---
LOG_DIR = "logs"
HISTORY_DIR = os.path.join(LOG_DIR, "history")
LOG_FILENAME = "app.log"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

# --- 全局初始化标志 ---
_is_logger_initialized = False
logger = logging.getLogger("DeepInsightOtakuApp")

def setup_logger():
    """
    配置并返回一个 logger 实例。
    使用全局标志确保此函数体内的代码在整个应用生命周期中只执行一次，
    即使在多进程环境（如 Flask aiohttp 或 PyTorch DataLoader）中也是如此。
    """
    global _is_logger_initialized
    if _is_logger_initialized:
        return

    # --- 启动时归档上一次的日志 ---
    def archive_previous_log_on_startup():
        """在应用启动时，将上一次运行的日志文件归档。"""
        source_path = os.path.join(LOG_DIR, LOG_FILENAME)
        if os.path.exists(source_path) and os.path.getsize(source_path) > 0:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                dest_filename = f"{os.path.splitext(LOG_FILENAME)[0]}_{timestamp}.log"
                dest_path = os.path.join(HISTORY_DIR, dest_filename)
                
                shutil.move(source_path, dest_path)
                print(f"成功归档上一次的日志到: {dest_path}")
            except Exception as e:
                print(f"归档上一次的日志时出错: {e}")

    archive_previous_log_on_startup()

    # --- 自定义日志格式 ---
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] [%(process)d] - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # --- 自定义处理器和轮换逻辑 ---
    class UnbufferedTimedRotatingFileHandler(TimedRotatingFileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()

    def custom_rotator(source, dest):
        history_dest = os.path.join(HISTORY_DIR, os.path.basename(dest))
        if os.path.exists(history_dest):
            os.remove(history_dest)
        if os.path.exists(source):
            try:
                shutil.move(source, history_dest)
            except Exception as e:
                print(f"Error rotating log file: {e}")

    # --- Logger 设置 ---
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    # --- 文件处理器 ---
    file_handler = UnbufferedTimedRotatingFileHandler(
        os.path.join(LOG_DIR, LOG_FILENAME),
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    file_handler.rotator = custom_rotator

    # --- 控制台处理器 ---
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(log_format, date_format))

    # --- 添加处理器 ---
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    # --- 标记为已初始化 ---
    _is_logger_initialized = True
    
    # --- 初始日志 ---
    logger.info("=" * 50)
    logger.info("Logger initialized with real-time flushing and history rotation.")
    logger.info("=" * 50)

def get_latest_log_file():
    """获取当前正在写入的日志文件路径。"""
    return os.path.join(LOG_DIR, LOG_FILENAME)
