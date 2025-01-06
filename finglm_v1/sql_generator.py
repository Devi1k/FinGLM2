class SQLAgent:
    def __init__(self, llm_client):
        self.llm = llm_client
        
    
    def generate_sql(self, understanding: dict) -> str:
        
        # 根据信息抽取结果生成对应 sql
        prompt = self.build_sql_prompt(understanding)
        sql = self.llm.generate(prompt)
        
        # TODO: 再次调用进行反思，优化生成的sql（包括表连接，条件等）

        
        return sql
    
    def build_sql_prompt(self, understanding: dict) -> str:
        conditions = []
        
        # 处理实体条件
        for entity in understanding['entities']:
            conditions.append(f"{entity['field']} = '{entity['value']}'")
            
        # 处理上下文关联条件
        for entity in understanding['context_entities']:
            conditions.append(f"{entity['field']} = '{entity['value']}'")
            
        return ' AND '.join(conditions)