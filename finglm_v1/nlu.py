from pydantic import fields


class ParserAgent:
    def __init__(self, llm_client):
        self.llm = llm_client
        
    def parse(self, question: str, context: dict) -> dict:
        # Embedding 根据表名，表解释检索需要的表
        # TODO: 根据问题（上下文）进行向量检索，获取相关的数据表


        # 构建prompt
        prompt = self.build_understanding_prompt(question, context)
        
        # 调用LLM生成实体以及对应的表
        response = self.llm.generate(prompt)
        
        # 解析响应
        understanding = self.parse_understanding(response)

        # 读取表schema，再次生成需要的对应 field
        prompt = self.build_fields_prompt(understanding['table'])
        response = self.llm.generate(prompt)
        fields = self.parse_fields(response)

        return {
            'table': understanding['table'],
            'entities': understanding['entities'],
            'fields': fields['fields'], 
        }
    
    def build_understanding_prompt(self, question: str, context: dict) -> str:
        return f"""
        历史对话: {context['history']}
        当前问题: {question}
        请提取:
        1. 问题意图
        2. 关键实体
        3. 上下文关联实体
        """