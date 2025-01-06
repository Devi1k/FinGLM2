class AnswerAgent:
    def __init__(self, llm_client):
        self.llm = llm_client
        
    def generate_answer(self, data: dict, understanding: dict) -> str:
        # 构建prompt
        prompt = self.build_answer_prompt(data, understanding)
        
        # 调用LLM
        response = self.llm.generate(prompt)
        
        # 格式化答案
        answer = self.format_answer(response)
        
        return answer
    
    def build_answer_prompt(self, data: dict, understanding: dict) -> str:
        return f"""
        问题意图: {understanding['intent']}
        查询数据: {data}
        请生成准确、简洁的答案
        """