"""
对话管理模块

This module provides dialogue history management and context tracking capabilities.
"""
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
from collections import deque

from pydantic import BaseModel, Field

from finglm_v1.core.types import QuestionContext

logger = logging.getLogger(__name__)

class DialogueTurn(BaseModel):
    """对话轮次模型"""
    question_id: str = Field(..., description="问题ID")
    question: str = Field(..., description="用户问题")
    answer: str = Field(..., description="系统回答")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Dialogue(BaseModel):
    """对话会话模型"""
    dialogue_id: str = Field(..., description="对话ID")
    turns: List[DialogueTurn] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DialogueManager:
    """对话管理器
    
    提供对话历史管理和上下文跟踪功能：
    1. 内存存储
    2. 上下文窗口
    3. 会话恢复
    
    Attributes:
        max_history: 每个对话保留的最大轮次数
        context_window: 上下文窗口大小
        dialogues: 对话存储字典
    """
    
    def __init__(
        self,
        max_history: int = 100,
        context_window: int = 5
    ) -> None:
        """初始化对话管理器
        
        Args:
            max_history: 最大历史记录数
            context_window: 上下文窗口大小
        """
        self.max_history = max_history
        self.context_window = context_window
        self.dialogues: Dict[str, Dialogue] = {}
        
        logger.info(
            f"Initialized DialogueManager with "
            f"max_history={max_history}, context_window={context_window}"
        )
    
    async def get_context(self, question_id: str) -> QuestionContext:
        """获取问题上下文
        
        Args:
            question_id: 问题ID
            
        Returns:
            QuestionContext: 问题上下文
        """
        try:
            # 解析对话ID
            dialogue_id = question_id.split("-")[0]
            
            # 获取对话
            dialogue = self.dialogues.get(dialogue_id)
            if not dialogue:
                return QuestionContext()
                
            # 获取最近的对话轮次
            recent_turns = dialogue.turns[-self.context_window:]
            
            # 构建上下文
            context = QuestionContext(
                history=[
                    {
                        "role": "human" if i % 2 == 0 else "assistant",
                        "content": turn.question if i % 2 == 0 else turn.answer
                    }
                    for i, turn in enumerate(recent_turns)
                ],
                metadata={
                    "dialogue_id": dialogue_id,
                    "question_id": question_id,
                    "turn_count": len(dialogue.turns)
                }
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get context for {question_id}: {str(e)}")
            return QuestionContext()
    
    async def update_history(
        self,
        question_id: str,
        question: str,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """更新对话历史
        
        Args:
            question_id: 问题ID
            question: 用户问题
            answer: 系统回答
            metadata: 元数据
        """
        try:
            # 解析对话ID
            dialogue_id = question_id.split("-")[0]
            
            # 获取或创建对话
            dialogue = self.dialogues.get(dialogue_id)
            if not dialogue:
                dialogue = Dialogue(dialogue_id=dialogue_id)
                self.dialogues[dialogue_id] = dialogue
            
            # 添加新的对话轮次
            dialogue.turns.append(DialogueTurn(
                question_id=question_id,
                question=question,
                answer=answer,
                metadata=metadata or {}
            ))
            
            # 限制历史记录数量
            if len(dialogue.turns) > self.max_history:
                dialogue.turns = dialogue.turns[-self.max_history:]
            
            logger.debug(
                f"Updated history for dialogue {dialogue_id}, "
                f"total turns: {len(dialogue.turns)}"
            )
            
        except Exception as e:
            logger.error(
                f"Failed to update history for {question_id}: {str(e)}"
            )
            raise
