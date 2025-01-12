"""
金融问答系统

This package provides a question answering system for financial data.
"""

from finglm_v1.core.system import FinanceQASystem
from finglm_v1.core.types import (
    QuestionContext,
    QuestionUnderstanding,
    QueryResult,
    TableInfo
)

__all__ = [
    'FinanceQASystem',
    'QuestionContext',
    'QuestionUnderstanding',
    'QueryResult',
    'TableInfo'
]
