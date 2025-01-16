"""
配置文件模块

This module contains all the configuration settings for the FinanceQA system.
"""
from pathlib import Path
from typing import Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """系统配置类
    
    包含所有系统级别的配置参数，使用pydantic进行验证
    """
    
    # 基础路径配置
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    ASSETS_DIR: Path = BASE_DIR / "assets"

    # 系统配置
    MAX_WORKERS: int = 8
    
    # 数据文件路径
    DATA_DICTIONARY_PATH: Path = ASSETS_DIR / "data_dictionary.xlsx"
    ALL_TABLES_SCHEMA_PATH: Path = ASSETS_DIR / "all_tables_schema.txt"
    
    # LLM配置
    LLM_API_KEY: str = "e505e01b08f94aa1adb66f0930050523.3GREH5cSdPoASBcI"
    LLM_MODEL: str = "glm-4-plus"
    LLM_BASE_URL: str = "https://open.bigmodel.cn/api/paas/v4/"

    # 数据库配置
    DB_ACCESS_TOKEN: str = "3a6ab7f347954c9e96a5092b1b45d2c3"
    
    # 向量存储配置
    VECTOR_STORE_MODEL: str = "Conan-embedding-v1"
    VECTOR_STORE_TYPE: str = "faiss"
    VECTOR_STORE_PATH: Path = BASE_DIR / "vector_store"

    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: Path = BASE_DIR / "logs" / "app.log"
    
    # 缓存配置
    CACHE_TYPE: str = "memory"
    CACHE_TTL: int = 3600  # 1小时
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )

settings = Settings()
