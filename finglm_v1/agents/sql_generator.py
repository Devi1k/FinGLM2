"""
SQL生成代理模块

This module provides SQL generation capabilities based on natural language understanding.
"""
from typing import List, Dict, Any, Optional
import logging
import json
from datetime import datetime

from pydantic import BaseModel, Field

from finglm_v1.core.types import QuestionUnderstanding, LLMError
from finglm_v1.agents.llm_client import LLMClient

logger = logging.getLogger(__name__)

class SQLTemplate(BaseModel):
    """SQL模板模型"""
    template: str = Field(..., description="SQL模板")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="参数")
    description: str = Field(..., description="模板描述")

class SQLGenerator:
    """SQL生成代理
    
    负责：
    1. SQL模板管理
    2. 参数替换
    3. SQL生成和验证
    
    Attributes:
        llm: LLM客户端
        templates: SQL模板字典
    """
    
    def __init__(self, llm: LLMClient) -> None:
        """初始化SQL生成代理
        
        Args:
            llm: LLM客户端
        """
        self.llm = llm
        logger.info("Initialized SQLGenerator")
    
    async def _format_table_info(self, understanding: QuestionUnderstanding) -> str:
        """格式化表信息
        
        Args:
            understanding: 问题理解结果
            
        Returns:
            str: 格式化的表信息
        """
        table_info = []
        for table in understanding.relevant_tables:
            fields = "\n".join(
                f"- {field['name']}: {field['description']}。示例: {field['example']}"
                for field in table.fields
            )
            table_info.append(
                f"表名: {table.chinese_name} ({table.english_name})\n"
                f"描述: {table.description}\n"
                f"字段:\n{fields}\n"
            )
        return "\n".join(table_info)
    
    async def generate_sql(
        self,
        question: str,
        understanding: QuestionUnderstanding
    ) -> str:
        """生成SQL查询
        
        Args:
            understanding: 问题理解结果
            
        Returns:
            str: 生成的SQL查询
            
        Raises:
            LLMError: LLM调用失败
        """
        try:
            # TODO 处理 SQL 生成 
            # 构建提示
            table_info = await self._format_table_info(understanding)
            
            prompt = (
                "任务：根据问题理解和表结构信息生成SQL查询。\n\n"
                "用户问题："
                f"{question}\n\n"
                "表结构信息：\n"
                f"{table_info}\n\n"
                "问题理解：\n"
                f"{understanding.model_dump_json(indent=2)}\n\n"
                "要求：\n"
                "1. 生成标准的SQL查询语句\n"
                "2. 使用正确的表和字段名\n"
                "3. 确保查询性能\n"
                "4. 处理好表之间的关联\n\n"
                "5. 查询年度报告时禁止使用 GROUP BY InnerCode\n"
                "6. 时间字段使用 DATE() 函数进行比较\n"
                "7. 优先使用已有的统计字段而不是自行计算\n\n"
                "8. 只需要输出SQL语句，不需要其他解释。"
                "查询参考：\n"
                "1. 查询近一个月最高价时，优先使用 HighPriceRM (近一月最高价(元)) 字段\n"
                "2. 查询近一月最低价时，直接使用 LowPriceRM 字段\n"
                "3. 查询行业数量示例：\n"
                "   SELECT count(*) as 风电零部件_2021\n"
                "   FROM AStockIndustryDB.LC_ExgIndustry\n"
                "   WHERE ThirdIndustryName LIKE '%风电零部件%'\n"
                "     AND YEAR(InfoPublDate) = 2021\n"
                "     AND IfPerformed = 1\n"
                "4. 查询最新年度报告机构持股示例：\n"
                "   SELECT *\n"
                "   FROM AStockShareholderDB.LC_StockHoldingSt\n"
                "   WHERE DATE(EndDate) = 'YYYY-12-31'\n"
                "     AND UpdateTime = (\n"
                "       SELECT MAX(UpdateTime)\n"
                "       FROM AStockShareholderDB.LC_StockHoldingSt\n"
                "       WHERE DATE(EndDate) = 'YYYY-12-31'\n"
                "     )\n"
                "   ORDER BY InstitutionsHoldings DESC\n"
                "   LIMIT 1\n"
                "5. 查询新高信息使用 AStockMarketQuotesDB.CS_StockPatterns：\n"
                "   - 近半年新高：使用 IfHighestHPriceRMSix 字段\n"
                "   - 近一周新高：使用 IfHighestHPriceRW 字段\n"
                "6. 半年度报告查询条件：\n"
                "   YEAR(EndDate) = YYYY AND InfoSource = '半年度报告'\n\n"
                
            )
            
            # 生成SQL
            response = await self.llm.generate_with_context(
                prompt
            )
            

            sql = response.content.strip()
            
            # 验证SQL（可以添加更多验证逻辑）
            # if not sql.lower().startswith("select"):
            #     raise ValueError("Generated SQL must be a SELECT statement")
            
            logger.debug(f"Generated SQL: {sql}")
            return sql
            
        except Exception as e:
            logger.error(f"Failed to generate SQL: {str(e)}")
            raise LLMError(f"SQL generation failed: {str(e)}")
