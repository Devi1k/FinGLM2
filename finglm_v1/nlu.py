from pydantic import fields
from typing import List
import json
import re


class ParserAgent:
    def __init__(self, llm_client, vector_store):
        self.llm = llm_client
        self.vector_store = vector_store
    
    def _rewrite_question(self, question: str, context: dict) -> str:
        """Rewrite the question using context."""
        prompt = self.build_rewrite_prompt(question, context)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": ""},
        ]
        rewrite_response = self.llm.generate(messages)
        return rewrite_response

    def parse(self, question: str, context: dict, table_representation: List[str]) -> dict:
        # Embedding 根据表名，表解释检索需要的表

        # 改写问题，增强实体链指能力
        if context:
            question = self._rewrite_question(question, context)

        table_indices = self.vector_store.search(question)
        table_list = [table_representation[idx] for idx in table_indices]

        # 构建 prompt 抽取实体，表名
        prompt = self.build_understanding_prompt(question, table_list)

        # 调用LLM生成实体以及对应的表
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": ""},
        ]
        response = self.llm.generate(messages)

        # 解析响应
        understanding = self.parse_understanding(response)


        # TODO: 读取表schema，再次生成需要的对应 field
        prompt = self.build_fields_prompt(understanding["table"])
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": ""},
        ]
        response = self.llm.generate(messages)
        fields = self.parse_fields(response)

        return {
            "table": understanding["table"],
            "entities": understanding["entities"],
            "fields": fields["fields"],
        }

    def build_understanding_prompt(self, question: str, table_list: List[str]) -> str:


        prompt_template = """任务说明:
你是一个数据库查询分析助手。请分析给定的查询问题，完成表选择和实体抽取任务，并以JSON格式输出结果。

数据库表信息:
<<table_list>>

查询问题:
<<question>>

请以下列JSON格式输出分析结果：
{{
    "required_tables": [
        {
            "table_name": "首要查询表名/关联辅助表名",
            "reason": "选择该表的原因",
        }
    ],
    "entities": {{
        "main_entity": {{  # 查询主体
            "value": "实体值",
            "confidence": 0.9  # 提取可信度，0-1之间
        }},
}}

要求：
1. 确保输出为有效的JSON格式
2. 所有文本字段使用UTF-8编码
3. confidence值在0-1之间，表示提取的确信度
4. 表选择时考虑查询的完整性，包括直接和间接需要的表
5. 实体抽取时关注查询中的显式和隐式实体
6. 对选择和抽取结果提供清晰的解释
        """
        prompt = prompt_template.replace(
            "<<table_list>>", " ".join(table_list)
        ).replace("<<question>>", question)
        return prompt

    def build_rewrite_prompt(self, question: str, context: dict) -> str:
        context_list = context.get('history', [])
        context_list = context_list[-3:]
        prompt_template = """任务说明:
你是一个对话重写助手。请基于以下多轮对话的上下文，重写当前对话内容，使其更清晰明确。

要求:
1. 保持原始对话的核心意图和语气不变
2. 将代词("这个"、"那个"等)替换为明确的名词
3. 补充必要的上下文信息
4. 确保重写后的内容独立可理解
5. 保持对话的自然流畅性
历史对话:
<<context_list>>
当前需要重写的对话:
内容: <<question>>

请重写上述对话内容，确保清晰明确且保持原意。
"""
        prompt_template = prompt_template.replace("<<question>>", question).replace("<<context_list>>", " ".join(context_list))
        return prompt_template

    def parse_understanding(self, response: str) -> dict:
        """
        解析大模型输出的文本，提取JSON字符串并转换为Python字典。
        
        Args:
            response (str): 大模型的文本输出

        Returns:
            dict: 包含表和实体信息的字典
        """
        def get_empty_result():
            """返回空结果的辅助函数"""
            return {"table": [], "entities": {}}

        try:
            # 首先尝试匹配```json```格式
            code_block_pattern = r'```(?:json)?([\s\S]*?)```'
            code_block_match = re.search(code_block_pattern, response)
            
            if code_block_match:
                # 如果找到```json```格式，提取其中的内容
                json_str = code_block_match.group(1).strip()
            else:
                # 否则尝试直接匹配JSON对象
                json_pattern = r'({[\s\S]*})'
                json_match = re.search(json_pattern, response)
                if not json_match:
                    return get_empty_result()
                json_str = json_match.group(1)

            # 尝试解析JSON字符串
            data = json.loads(json_str)
            
            # 提取表和实体信息
            result = {
                "table": [{"name": table["table_name"], "reason": table["reason"]} 
                         for table in data.get("required_tables", [])],
                "entities": data.get("entities", {})
            }
            return result

        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            # 任何解析错误都返回空结果
            return get_empty_result()

    