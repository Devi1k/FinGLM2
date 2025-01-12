"""
金融问答系统核心模块

This module contains the main FinanceQASystem class that orchestrates all components.
"""

from typing import Optional, Dict, Any, List
import logging
import pandas as pd
from pathlib import Path

from finglm_v1.config.settings import settings
from finglm_v1.core.types import QuestionContext, QuestionUnderstanding, QueryResult
from finglm_v1.agents.answer_generator import AnswerGenerator
from finglm_v1.agents.llm_client import LLMClient
from finglm_v1.agents.nlu import ParserAgent
from finglm_v1.agents.sql_generator import SQLGenerator
from finglm_v1.utils.dialogue_manager import DialogueManager
from finglm_v1.utils.vector_store import TableVectorStore
from finglm_v1.utils.database import DatabaseClient
from finglm_v1.utils.logging import setup_logging

logger = logging.getLogger(__name__)


class FinanceQASystem:
    """金融问答系统主类

    该类负责协调各个组件完成问答流程，包括：
    1. 问题理解
    2. SQL生成
    3. 数据查询
    4. 答案生成

    Attributes:
        data_dictionary: DataFrame, 包含数据字典信息
        database_table_map: Dict[str, str], 中英文表名映射
        llm: LLMClient, LLM客户端
        dialogue_manager: DialogueManager, 对话管理器
        vector_store: TableVectorStore, 向量存储
        parser_agent: ParserAgent, 问题理解组件
        sql_agent: SQLGenerator, SQL生成组件
        answer_agent: AnswerGenerator, 答案生成组件
        db: DatabaseClient, 数据库客户端
    """

    def __init__(self) -> None:
        """初始化问答系统"""
        setup_logging()
        logger.info("Initializing FinanceQASystem...")
        self._load_data_dictionary()
        self._initialize_components()
        logger.info("FinanceQASystem initialized successfully")

    def _load_data_dictionary(self) -> None:
        """加载数据字典

        从配置的路径加载数据字典Excel文件，处理成系统所需的格式
        """
        try:
            self.data_dictionary = pd.read_excel(
                settings.DATA_DICTIONARY_PATH, sheet_name="库表关系"
            )
            self.data_dictionary["库表名中文"] = (
                self.data_dictionary["库名中文"] + "." + self.data_dictionary["表中文"]
            )
            self.data_dictionary["库表名英文"] = (
                self.data_dictionary["库名英文"] + "." + self.data_dictionary["表英文"]
            )
            self.data_dictionary["representation"] = (
                "库表名："
                + self.data_dictionary["库表名中文"]
                + "，注释："
                + self.data_dictionary["表描述"]
            )
            self.database_table_map = self.data_dictionary.set_index("库表名中文")[
                "库表名英文"
            ].to_dict()

            with open(settings.ALL_TABLES_SCHEMA_PATH, "r") as f:
                self.all_tables_schema = f.read()

            logger.info("Data dictionary loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load data dictionary: {str(e)}")
            raise

    def _initialize_components(self) -> None:
        """初始化系统组件

        创建并初始化所有必要的系统组件，包括LLM客户端、对话管理器等
        """
        try:
            self.llm = LLMClient()
            self.dialogue_manager = DialogueManager()
            self.vector_store = TableVectorStore(
                list(self.data_dictionary["representation"])
            )
            self.parser_agent = ParserAgent(
                self.llm,
                self.vector_store,
                self.data_dictionary,
                self.all_tables_schema,
            )
            self.sql_agent = SQLGenerator(self.llm)
            self.answer_agent = AnswerGenerator(self.llm)
            self.db = DatabaseClient(access_token=settings.DB_ACCESS_TOKEN)
            logger.info("All components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise

    async def process_question(
        self,
        question_id: str,
        question: str,
        *,
        context: Optional[QuestionContext] = None,
    ) -> str:
        """处理用户问题

        Args:
            question_id: 问题唯一标识符
            question: 用户问题文本
            context: 可选的上下文信息

        Returns:
            str: 生成的答案文本

        Raises:
            QuestionProcessingError: 问题处理过程中的错误
            DatabaseError: 数据库查询错误
        """
        logger.info(f"Processing question {question_id}: {question}")
        max_iteration = 3
        answer = ''
        try:
            # 1. 获取上下文
            if context is None:
                context = await self.dialogue_manager.get_context(question_id)
            for iter in range(max_iteration):
                # 2. 理解问题
                understanding = await self.parser_agent.parse(question, context)

                # 3. 生成查询
                sql = await self.sql_agent.generate_sql(question, understanding)

                # 4. 执行查询
                data = self.db.execute(sql)

                # 5. 生成答案
                answer = await self.answer_agent.generate_answer(data, understanding)

                if "<|FINISH|>" in answer:
                    answer = answer.split("<|FINISH|>")[0]
                    break
                
                # 6. 更新对话历史
                await self.dialogue_manager.update_history(question_id, question, answer)

            logger.info(f"Successfully processed question {question_id}")
            return answer

        except Exception as e:
            logger.error(f"Error processing question {question_id}: {str(e)}")
            raise
