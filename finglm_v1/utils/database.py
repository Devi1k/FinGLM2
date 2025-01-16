"""
数据库接口模块

提供与数据库交互的基本功能，包括执行SQL查询等。
"""
from typing import Optional, Dict, Any, Union
import json
import logging
import time
import requests
from functools import wraps
from pydantic import BaseModel

from finglm_v1.config import settings


logger = logging.getLogger(__name__)


class DataBaseError(Exception):
    """数据库操作异常"""
    pass


def retry_on_error(max_retries=3, delay=1):
    """重试装饰器
    
    Args:
        max_retries: 最大重试次数
        delay: 重试间隔(秒)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.RequestException, DataBaseError) as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (attempt + 1)  # 指数退避
                        logger.warning(f"第{attempt + 1}次调用失败，{wait_time}秒后重试: {str(e)}")
                        time.sleep(wait_time)
                    continue
            logger.error(f"达到最大重试次数({max_retries})，调用失败: {str(last_error)}")
            raise last_error
        return wrapper
    return decorator


class DatabaseClient:
    """数据库接口类
    
    提供与数据库交互的基本功能，包括执行SQL查询等。
    
    Attributes:
        base_url: API基础URL
        access_token: 访问令牌
    """
    
    def __init__(
            self,
            access_token: str,
            base_url: str = "https://comm.chatglm.cn/finglm2/api/query"
        ) -> None:
        """初始化数据库接口
        
        Args:
            access_token: API访问令牌
            base_url: API基础URL，默认为官方接口地址
        """
        self.base_url = base_url
        self.access_token = access_token
    
    @retry_on_error(max_retries=3, delay=1)
    def execute(
            self,
            sql: str,
            limit: int = 15
        ) -> Dict[str, Any]:
        """执行SQL查询
        
        Args:
            sql: SQL查询语句
            limit: 返回结果的最大行数，默认为15
            
        Returns:
            Dict[str, Any]: 查询结果
            
        Raises:
            DataBaseError: 当API调用失败或返回结果无效时抛出
        """
        url = f"{self.base_url}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token}"
        }
        data = {
            "sql": sql,
            "limit": limit
        }
        
        try:
            logger.info(f"Executing SQL query: {sql}")
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()  # 检查HTTP状态码
            
            result = response.json()
            logger.debug(f"Query result: {result}")
            return result
            
        except requests.exceptions.RequestException as e:
            error_msg = f"API请求失败: {str(e)}"
            logger.error(error_msg)
            raise DataBaseError(error_msg)
        except json.JSONDecodeError as e:
            error_msg = f"解析API响应失败: {str(e)}"
            logger.error(error_msg)
            raise DataBaseError(error_msg)
        except Exception as e:
            error_msg = f"执行查询时发生错误: {str(e)}"
            logger.error(error_msg)
            raise DataBaseError(error_msg)


if __name__ == "__main__":
    db = DatabaseClient(access_token=settings.DATABASE_ACCESS_TOKEN)
    result = db.execute("SELECT COUNT(*) AS Year2021DisclosureCount FROM astockeventsdb.lc_credit WHERE DATE(InfoPublDate) BETWEEN '2021-01-01' AND '2021-12-31'")
    print(result)