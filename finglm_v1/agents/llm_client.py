"""
LLM客户端模块

This module provides a client for interacting with Large Language Models.
"""
from typing import List, Dict, Any, Optional, Union
import logging
import asyncio
from functools import lru_cache
from datetime import datetime
import json
import hashlib

from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from zhipuai import ZhipuAI
from openai import OpenAI
from zhipuai.types.chat.chat_completion import Completion
from finglm_v1.config.settings import settings
from finglm_v1.core.types import LLMError

logger = logging.getLogger(__name__)

class Message(BaseModel):
    """聊天消息模型"""
    role: str = Field(..., description="消息角色(system/user/assistant)")
    content: str = Field(..., description="消息内容")
    name: Optional[str] = Field(None, description="可选的消息发送者名称")
    
    class Config:
        frozen = True

class ChatResponse(BaseModel):
    """聊天响应模型"""
    content: str = Field(..., description="响应内容")
    role: str = Field(default="assistant", description="响应角色")
    finish_reason: Optional[str] = Field(None, description="结束原因")
    usage: Dict[str, int] = Field(default_factory=dict, description="token使用统计")
    
    @classmethod
    def from_openai_response(cls, response) -> "ChatResponse":
        """从OpenAI响应创建ChatResponse实例"""
        choice = response.choices[0]
        return cls(
            content=choice.message.content,
            role=choice.message.role,
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        )

class LLMClient:
    """LLM客户端类
    
    提供与大语言模型交互的接口，支持：
    1. 异步调用
    2. 自动重试
    3. 结果缓存
    4. 详细日志
    5. 错误处理
    
    Attributes:
        client: ZhipuAI客户端实例
        model: 使用的模型名称
        max_retries: 最大重试次数
        cache_size: LRU缓存大小
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 3,
        cache_size: int = 1000
    ) -> None:
        """初始化LLM客户端
        
        Args:
            api_key: 智谱AI API密钥，默认从配置获取
            model: 模型名称，默认从配置获取
            max_retries: 最大重试次数
            cache_size: LRU缓存大小
        
        Raises:
            LLMError: 初始化失败时抛出
        """
        try:
            self.client = OpenAI(api_key=api_key or settings.LLM_API_KEY,base_url=settings.LLM_BASE_URL)
            self.model = model or settings.LLM_MODEL
            self.max_retries = max_retries
            
            logger.info(
                f"Initialized LLMClient with model={self.model}, "
                f"max_retries={max_retries}, cache_size={cache_size}"
            )
        except Exception as e:
            raise LLMError(f"Failed to initialize LLMClient: {str(e)}")
    
    def _generate_cache_key(self, messages: List[Message], **kwargs) -> str:
        """生成缓存键
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Returns:
            str: 缓存键
        """
        # 将消息和参数序列化为JSON字符串
        cache_data = {
            "messages": [
                msg.model_dump() if isinstance(msg, Message) 
                else msg for msg in messages
            ],
            "model": self.model,
            **{k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool, type(None)))}
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        
        # 计算SHA256哈希
        return hashlib.sha256(cache_str.encode()).hexdigest()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def generate(
        self,
        messages: List[Union[Message, Dict[str, str]]],
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        timeout: float = 30.0,
        **kwargs: Any
    ) -> ChatResponse:
        """生成LLM响应
        
        Args:
            messages: 消息列表，每个消息可以是Message对象或字典
            temperature: 温度参数
            top_p: top-p采样参数
            timeout: 超时时间(秒)
            **kwargs: 传递给底层API的其他参数
            
        Returns:
            ChatResponse: 聊天响应
            
        Raises:
            LLMError: LLM调用失败
            asyncio.TimeoutError: 调用超时
        """
        try:
            # 转换消息格式
            processed_messages: List[Message] = []
            for msg in messages:
                if isinstance(msg, dict):
                    processed_messages.append(Message(**msg))
                else:
                    processed_messages.append(msg)
            
            # 记录请求
            logger.info(
                f"Generating response for messages={processed_messages}, "
                f"temperature={temperature}, top_p={top_p}"
            )
            
            # 异步调用 LLM
            loop = asyncio.get_running_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.client.chat.completions.create(
                        model=self.model,
                        messages=[msg.model_dump() for msg in processed_messages],
                        temperature=temperature,
                        top_p=top_p,
                        timeout=timeout,
                        **kwargs
                    )
                ),
                timeout=timeout
            )
            
            # 转换响应
            chat_response = ChatResponse.from_openai_response(response)
            
            # 记录响应
            logger.debug(
                f"Generated response: content_length={len(chat_response.content)}, "
                f"usage={chat_response.usage}"
            )
            
            return chat_response
            
        except asyncio.TimeoutError:
            logger.error(f"LLM request timed out after {timeout}s")
            raise
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise LLMError(f"LLM generation failed: {str(e)}")
    
    async def generate_with_context(
        self,
        user_message: str,
        context: Optional[List[Message]] = None,
        system_message: Optional[str] = None
    ) -> ChatResponse:
        """使用上下文生成响应
        
        Args:
            user_message: 用户消息
            context: 可选的上下文消息列表
            system_message: 可选的系统消息
            
        Returns:
            ChatResponse: 聊天响应
        """
        messages: List[Message] = []
        
        # 添加系统消息
        if system_message:
            messages.append(Message(role="system", content=system_message)) # pyright: ignore
            
        # 添加上下文消息
        if context:
            messages.extend(context)
            
        # 添加用户消息
        messages.append(Message(role="user", content=user_message)) # pyright: ignore
        
        return await self.generate(messages) # pyright: ignore


if __name__ == "__main__":
    # 测试 LLMClient 的 generate 功能
    async def test_generate():
        llm_client = LLMClient(model="glm-4-flash", api_key=settings.LLM_API_KEY)
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="What is the capital of France?")
        ]
        
        try:
            response = await llm_client.generate(messages)
            print(f"Generated response: {response.content}")
            print(f"Usage: {response.usage}")
        except LLMError as e:
            print(f"Error occurred: {e}")

    # 运行测试
    asyncio.run(test_generate())
