"""
核心类型定义模块

This module contains all the core type definitions used across the system.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime

class QuestionContext(BaseModel):
    """问题上下文模型
    
    包含处理问题所需的上下文信息
    """
    
    history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="历史对话记录"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="元数据信息"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="上下文创建时间"
    )

class TableInfo(BaseModel):
    """表信息模型"""
    
    chinese_name: str = Field(..., description="表中文名")
    english_name: str = Field(..., description="表英文名")
    description: str = Field(..., description="表描述")
    fields: List[Dict[str, str]] = Field(..., description="字段信息")

class QuestionUnderstanding(BaseModel):
    """问题理解结果模型"""
    question: str = Field(..., description="问题文本")

    entities: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="识别出的实体"
    )
    relevant_tables: List[TableInfo] = Field(
        default_factory=list,
        description="相关表信息"
    )

class QueryResult(BaseModel):
    """查询结果模型"""
    
    data: List[Dict[str, Any]] = Field(..., description="查询数据")
    columns: List[str] = Field(..., description="列名")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="元数据"
    )

class FinanceQAError(Exception):
    """金融问答系统基础异常类"""
    pass

class QuestionProcessingError(FinanceQAError):
    """问题处理错误"""
    pass

class DatabaseError(FinanceQAError):
    """数据库错误"""
    pass

class LLMError(FinanceQAError):
    """LLM调用错误"""
    pass
