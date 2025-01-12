"""
日志工具模块

This module provides logging utilities for the system.
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

from finglm_v1.config.settings import settings

def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
    max_bytes: int = 10_000_000,  # 10MB
    backup_count: int = 5
) -> None:
    """设置日志系统
    
    Args:
        level: 日志级别，默认使用配置中的级别
        log_file: 日志文件路径，默认使用配置中的路径
        max_bytes: 单个日志文件最大字节数
        backup_count: 保留的日志文件数量
    """
    level = level or settings.LOG_LEVEL
    log_file = log_file or settings.LOG_FILE
    
    # 创建日志目录
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建格式化器
    formatter = logging.Formatter(settings.LOG_FORMAT)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 添加文件处理器
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # 设置第三方库的日志级别
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    
    root_logger.info(f"Logging system initialized. Level: {level}, File: {log_file}")
