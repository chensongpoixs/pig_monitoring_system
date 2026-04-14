import logging
import os
from config.config import config

class Logger:
    def __init__(self):
        # 创建日志目录
        log_dir = os.path.dirname(config.LOG_FILE)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(config.LOG_FILE),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('PigMonitoringSystem')
    
    def info(self, message):
        """
        记录信息日志
        
        Args:
            message: 日志消息
        """
        self.logger.info(message)
    
    def warning(self, message):
        """
        记录警告日志
        
        Args:
            message: 日志消息
        """
        self.logger.warning(message)
    
    def error(self, message):
        """
        记录错误日志
        
        Args:
            message: 日志消息
        """
        self.logger.error(message)
    
    def critical(self, message):
        """
        记录严重错误日志
        
        Args:
            message: 日志消息
        """
        self.logger.critical(message)

# 创建全局日志实例
logger = Logger()
