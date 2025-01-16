"""
答案生成代理模块

This module provides answer generation capabilities based on SQL query results.
"""
from typing import List, Dict, Any, Optional
import logging
import json
from datetime import datetime
import pandas as pd
import re

from pydantic import BaseModel, Field

from finglm_v1.core.types import QuestionUnderstanding, LLMError
from finglm_v1.agents.llm_client import LLMClient

logger = logging.getLogger(__name__)

class AnswerTemplate(BaseModel):
    """答案模板模型"""
    template: str = Field(..., description="答案模板")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="参数")
    description: str = Field(..., description="模板描述")

class AnswerGenerator:
    """答案生成代理
    
    负责：
    1. 数据分析和处理
    2. 答案生成和格式化
    3. 数据可视化建议
    
    Attributes:
        llm: LLM客户端
        templates: 答案模板字典
    """
    
    def __init__(self, llm: LLMClient) -> None:
        """初始化答案生成代理
        
        Args:
            llm: LLM客户端
        """
        self.llm = llm
        logger.info("Initialized AnswerGenerator")
    
    def _format_data(self, data: pd.DataFrame) -> str:
        """格式化数据
        
        Args:
            data: 查询结果数据
            
        Returns:
            str: 格式化的数据描述
        """
        # 数据基本信息
        info = [
            f"数据形状: {data.shape[0]}行 x {data.shape[1]}列",
            f"列名: {', '.join(data.columns)}",
        ]
        
        # 数值列统计
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        if not numeric_cols.empty:
            stats = data[numeric_cols].agg(['min', 'max', 'mean', 'std']).round(2)
            info.append("\n数值列统计:")
            for col in numeric_cols:
                info.append(
                    f"- {col}:\n"
                    f"  最小值: {stats.loc['min', col]}\n"
                    f"  最大值: {stats.loc['max', col]}\n"
                    f"  平均值: {stats.loc['mean', col]}\n"
                    f"  标准差: {stats.loc['std', col]}"
                )
        
        # 分类列统计
        categorical_cols = data.select_dtypes(include=['object']).columns
        if not categorical_cols.empty:
            info.append("\n分类列统计:")
            for col in categorical_cols:
                value_counts = data[col].value_counts()
                if len(value_counts) <= 5:  # 只显示少量唯一值的列
                    info.append(
                        f"- {col}:\n"
                        f"  唯一值: {', '.join(str(x) for x in value_counts.index[:5])}\n" # pyright: ignore
                        f"  计数: {', '.join(str(x) for x in value_counts.values[:5])}"
                    )
        
        return "\n".join(info)

    async def generate_answer(
        self,
        data: Dict[str, Any],
        sql: str,
        understanding: QuestionUnderstanding
    ) -> str:
        """生成答案
        
        Args:
            data: 查询结果数据
            understanding: 问题理解结果
            
        Returns:
            str: 生成的答案
            
        Raises:
            LLMError: LLM调用失败
        """
        try:
            # 构建提示
            prompt = (
                "任务描述:\n"
                "您的任务是根据提供的用户查询、数据表信息、SQL查询语句及其结果,生成自然语言解释。"
                "您需要分析查询结果是否完全回答了用户问题,并根据分析给出相应回复。\n\n"
                
                "USER QUERY:\n"
                f"{understanding.question}\n\n"
                
                "QUERY RESULT:\n"
                f"{json.dumps(data, indent=2)}\n\n"

                "SQL QUERY:\n"
                f"{sql}\n\n"

                "QUERY RESULT:\n"
                f"{self._format_data(pd.DataFrame(data))}\n\n"
                
                "处理步骤:\n"
                "1. 分析用户查询意图和SQL查询的对应关系\n"
                "2. 理解SQL查询逻辑和结果含义\n"
                "3. 用自然语言解释查询结果\n"
                "4. 反思分析是否完全回答了用户问题\n"
                "5. 根据反思结果输出回复:\n"
                "   - 如果未完全回答问题:生成临时回复,说明当前结果和存在的问题\n"
                "   - 如果完全回答问题:生成最终回复,并以<|FINISH|>结束\n\n"
                
                "输出格式:\n"
                "ANALYSIS:\n"
                "<分析用户问题和SQL查询的对应关系>\n\n"
                
                "EXPLANATION:\n"
                "<用自然语言解释查询结果>\n\n"
                
                "REFLECTION:\n"
                "<反思是否完全回答了用户问题>\n\n"
                
                "RESPONSE:\n"
                "<根据反思结果生成的回复>\n"
                "[如果是最终回复则添加<|FINISH|>]"
            )
            
            # 生成答案
            response = await self.llm.generate_with_context(
                prompt,
                system_message=(
                    "你是一个专业的数据分析师，擅长解读数据并提供洞察。"
                    "请基于给定的数据信息和问题理解，生成准确、有见地的答案。"
                )
            )

            # 提取 RESPONSE 部分
            content = response.content.strip()
            response_match = re.search(r'RESPONSE:\s*(.*?)(?:\s*$|\s)', content, re.DOTALL)
            if response_match:
                answer = response_match.group(1).strip()
            else:
                logger.warning("No RESPONSE section found in LLM output")
                answer = ""  # 如果没找到 RESPONSE 部分，使用完整内容

            logger.debug(f"Generated answer: {answer}")
            return answer
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {str(e)}")
            raise LLMError(f"Answer generation failed: {str(e)}")
