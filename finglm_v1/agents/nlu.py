"""
自然语言理解代理模块

This module provides natural language understanding capabilities for the FinanceQA system.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import json
from datetime import datetime, timedelta
import re
from pathlib import Path

from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

from finglm_v1.config.settings import settings
from finglm_v1.core.types import (
    QuestionContext,
    QuestionUnderstanding,
    TableInfo,
    LLMError,
)
from finglm_v1.agents.llm_client import LLMClient, Message
from finglm_v1.utils.vector_store import TableVectorStore

logger = logging.getLogger(__name__)


class Entity(BaseModel):
    """实体模型"""

    table_name: Optional[str] = Field(None, description="表名")
    entity_value: Optional[str] = Field(None, description="实体值")
    confidence: float = Field(default=1.0, description="置信度")

    @property
    def is_valid(self) -> bool:
        """检查实体是否有效（至少包含表名或值之一）"""
        return bool(self.table_name or self.entity_value)


class TimeRange(BaseModel):
    """时间范围模型"""

    start: datetime = Field(..., description="开始时间")
    end: datetime = Field(..., description="结束时间")

    @classmethod
    def from_relative(cls, relative_desc: str) -> "TimeRange":
        """从相对时间描述创建时间范围

        Args:
            relative_desc: 相对时间描述，如"近一个月"、"过去7天"等

        Returns:
            TimeRange: 时间范围实例
        """
        now = datetime.now()

        # 解析常见的时间表达
        if "月" in relative_desc:
            months = int(re.search(r"\d+", relative_desc).group())
            start = now - timedelta(days=30 * months)
        elif "天" in relative_desc or "日" in relative_desc:
            days = int(re.search(r"\d+", relative_desc).group())
            start = now - timedelta(days=days)
        elif "年" in relative_desc:
            years = int(re.search(r"\d+", relative_desc).group())
            start = now.replace(year=now.year - years)
        else:
            raise ValueError(f"Unsupported time description: {relative_desc}")

        return cls(start=start, end=now)


class ParserAgent:
    """问题理解代理

    负责：
    1. 问题改写和上下文整合
    2. 实体识别和链接
    3. 相关表识别
    4. 字段映射
    5. 时间表达式标准化

    Attributes:
        llm: LLM客户端
        vector_store: 向量存储
        data_dictionary: 数据字典DataFrame
        table_schemas: 表结构信息
    """

    def __init__(
        self,
        llm: LLMClient,
        vector_store: TableVectorStore,
        data_dictionary: pd.DataFrame,
        table_schemas: str,
    ) -> None:
        """初始化解析代理

        Args:
            llm: LLM客户端
            vector_store: 向量存储
            data_dictionary: 数据字典DataFrame
            table_schemas: 表结构信息
        """
        self.llm = llm
        self.vector_store = vector_store
        self.data_dictionary = data_dictionary
        self.table_schemas = self._parse_table_schemas(table_schemas)

        logger.info("Initialized ParserAgent")

    def _parse_table_schemas(self, schema_text: str) -> Dict[str, TableInfo]:
        """解析表结构文本
        Args:
            schema_text: 表结构文本

        Returns:
            Dict[str, TableInfo]: 表结构信息字典
        """
        schemas: Dict[str, TableInfo] = {}
        current_table = None

        for line in schema_text.split("\n"):
            # 处理表头
            if line.startswith("==="):
                table_name = line.split(" ")[1]
                current_table = table_name
                schemas[current_table] = TableInfo(
                    chinese_name="",  # 从数据字典中补充
                    english_name=current_table,
                    description="",  # 从数据字典中补充
                    fields=[],
                )
                continue

            # 跳过无关行
            if (
                not current_table
                or "注释" in line
                or "-" * 20 in line
                or not line.strip()
            ):
                continue

            # 解析字段
            parts = line.split(None, 2)
            if len(parts) >= 2:
                field_name, field_comment, field_example = (
                    parts[0].strip(),
                    parts[1].strip(),
                    parts[2].strip(),
                )
                schemas[current_table].fields.append(
                    {
                        "name": field_name,
                        "description": field_comment,
                        "example": field_example,
                    }
                )

        # 补充表的中文名和描述
        for table in schemas.values():
            df_row = self.data_dictionary[
                self.data_dictionary["库表名英文"] == table.english_name
            ]
            if not df_row.empty:
                table.chinese_name = df_row["库表名中文"].iloc[0]
                table.description = df_row["表描述"].iloc[0]

        return schemas

    async def _rewrite_question(
        self, question: str, context: Optional[QuestionContext]
    ) -> str:
        """改写问题

        结合上下文信息改写问题，使其更完整和明确

        Args:
            question: 原始问题
            context: 上下文信息

        Returns:
            str: 改写后的问题
        """
        if not context or not context.history:
            return question

        prompt = (
            "任务说明:\n"
            "你是一个对话重写助手。请基于以下多轮对话的上下文，重写第{len(context.history)}轮对话内容，使其更清晰明确。\n\n"
            "要求:\n"
            "1. 保持原始对话的核心意图和语气不变\n"
            '2. 将代词("这个"、"那个"等)替换为明确的名词\n'
            "3. 补充必要的上下文信息\n"
            "4. 确保重写后的内容独立可理解\n"
            "5. 保持对话的自然流畅性\n\n"
            "历史对话:\n"
            f"{json.dumps(context.history[:-1], ensure_ascii=False, indent=2)}\n\n"
            "当前需要重写的对话:\n"
            f"内容: {question}\n\n"
            "请重写上述对话内容，确保清晰明确且保持原意。"
        )

        response = await self.llm.generate_with_context(prompt)

        return response.content.strip()

    async def _extract_entities(
        self, question: str, table_info: List[TableInfo]
    ) -> List[Entity]:
        """提取实体

        从问题中提取实体，并进行标准化

        Args:
            question: 问题文本
            table_info: 相关表信息

        Returns:
            List[Entity]: 实体列表
        """
        prompt = (
            "任务说明:\n"
            "你是一个数据库查询分析助手。请分析给定的查询问题，完成表选择和实体抽取任务，并以JSON格式输出结果。\n\n"
            "数据库表信息:\n"
            f"{json.dumps([t.model_dump() for t in table_info], ensure_ascii=False, indent=2)}\n\n"
            "查询问题:\n"
            f"{question}\n\n"
            "请以下列JSON格式输出分析结果：\n"
            "{\n"
            '    "required_tables": [\n'
            "        {\n"
            '            "table_name": "表名",\n'
            '            "necessity": "primary/auxiliary",\n'
            '            "reason": "选择该表的原因",\n'
            "        }\n"
            "    ],\n"
            '    "entities": {\n'
            '        "main_entity": [\n'
            "            {\n"
            '                "value": "实体值",\n'
            '                "confidence": 0.9\n'
            "            }\n"
            "        ]\n"
            "    }\n"
            "}\n\n"
            "要求：\n"
            "1. 确保输出为有效的JSON格式\n"
            "2. 所有文本字段使用UTF-8编码\n"
            "3. 时间格式统一使用ISO 8601标准(YYYY-MM-DD)\n"
            "4. confidence值在0-1之间，表示提取的确信度\n"
            "5. 如果某个字段不存在，使用null而不是空字符串"
        )

        response = await self.llm.generate_with_context(prompt)

        try:
            # 尝试从响应中提取JSON字符串
            content = response.content.strip()

            # 正则模式匹配两种格式：
            # 1. ```json ... ``` 格式
            # 2. 纯JSON格式
            json_pattern = r"```(?:json)?\s*({[\s\S]*?})\s*```|^({[\s\S]*})$"

            match = re.search(json_pattern, content)
            if not match:
                logger.error("No valid JSON found in response")
                return []

            # 获取匹配到的JSON字符串（优先使用markdown格式，如果没有则使用纯JSON格式）
            json_str = (match.group(1) or match.group(2)).strip()

            # 解析JSON
            result = json.loads(json_str)
            entities = []

            # 处理表实体和主实体
            for table in result.get("required_tables", []):
                entity = Entity(table_name=table.get("table_name"))  # pyright: ignore
                if entity.is_valid:
                    entities.append(entity)

            for entity_data in result.get("entities", {}).get("main_entity", []):
                entity = Entity(
                    entity_value=entity_data.get("value")
                )  # pyright: ignore
                if entity.is_valid:
                    entities.append(entity)

            return [e for e in entities if e.is_valid]

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")
            return []

    async def parse(
        self, question: str, context: Optional[QuestionContext] = None
    ) -> QuestionUnderstanding:
        """解析用户问题

        Args:
            question: 用户问题
            context: 可选的上下文信息
            table_representations: 可选的表示例列表

        Returns:
            QuestionUnderstanding: 问题理解结果

        Raises:
            LLMError: LLM调用失败
            ValueError: 输入参数错误
        """
        try:
            # 1. 问题改写
            if context is not None:
                rewritten_question = await self._rewrite_question(question, context)
            else:
                rewritten_question = question
            logger.debug(f"Rewritten question: {rewritten_question}")

            # 2. 相关表识别
            search_results = self.vector_store.search(rewritten_question, k=25)
            relevant_tables = []

            for result in search_results:
                idx = result.index  # 使用 VectorSearchResult 的 index 属性
                table_name_en = self.data_dictionary["库表名英文"].iloc[idx].lower()
                table_name_cn = self.data_dictionary["库表名中文"].iloc[idx]
                table_desc = self.data_dictionary["表描述"].iloc[idx]

                if table_name_en in self.table_schemas:
                    table_info = self.table_schemas[table_name_en]
                    # 更新表信息
                    table_info.chinese_name = table_name_cn
                    table_info.description = table_desc
                    relevant_tables.append(table_info)

            logger.debug(
                f"Relevant tables: {[t.chinese_name for t in relevant_tables]}"
            )

            # 3. 实体提取
            entities = await self._extract_entities(rewritten_question, relevant_tables)
            logger.debug(f"Extracted entities: {entities}")

            # 4. 构建理解结果
            understanding = QuestionUnderstanding(
                question=rewritten_question,
                entities=[{"entity_value": entity.entity_value} for entity in entities],
                relevant_tables=[
                    TableInfo(
                        chinese_name=table.chinese_name,
                        english_name=table.english_name,
                        description=table.description,
                        fields=table.fields,
                    )
                    for entity in entities
                    if entity.table_name
                    for table in relevant_tables
                    if entity.table_name == table.english_name
                ],
            )

            return understanding

        except Exception as e:
            logger.error(f"Failed to parse question: {str(e)}")
            raise
