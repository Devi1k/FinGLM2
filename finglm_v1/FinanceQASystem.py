from finglm_v1.answer_agent import AnswerAgent
from finglm_v1.llm_client import LLMClient
from finglm_v1.nlu import ParserAgent
from finglm_v1.sql_generator import SQLAgent
from finglm_v1.dialogue_manager import DialogueManager
from finglm_v1.vector_store import TableVectorStore



class FinanceQASystem:
    def __init__(self):
        self.llm = LLMClient()  # LLM API客户端
        self.dialogue_manager = DialogueManager()  # 对话管理
        self.vector_store = TableVectorStore()  # 内存向量存储
        self.parser_agent = ParserAgent(self.llm)  # 问题理解
        self.sql_agent = SQLAgent(self.llm)  # SQL生成
        self.answer_agent = AnswerAgent(self.llm)  # 答案生成
        
        
    def process_question(self, question_id: str, question: str) -> str:
        # 1. 获取上下文
        context = self.dialogue_manager.get_context(question_id)
        
        # 2. 理解问题
        understanding = self.parser_agent.parse(question, context)
        


        # 3. 生成查询
        sql = self.sql_agent.generate_sql(understanding)
        
        # 4. 执行查询
        data = self.db.execute(sql)
        
        # 5. 生成答案
        answer = self.answer_agent.generate_answer(data, understanding)
        
        # 6. 更新对话历史
        self.dialogue_manager.update_history(question_id, question, answer)
        
        return answer