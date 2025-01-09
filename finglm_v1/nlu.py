from pydantic import fields
from typing import List, Dict
import json
import re


class ParserAgent:
    def __init__(self, llm_client, vector_store, data_frame, table_dict, table_name_map):
        self.llm = llm_client
        self.vector_store = vector_store
        self.data_frame = data_frame
        self.table_dict = table_dict
        self.table_name_map = table_name_map

    def _rewrite_question(self, question: str, context: dict) -> str:
        """Rewrite the question using context."""
        prompt = self.build_rewrite_prompt(question, context)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": ""},
        ]
        rewrite_response = self.llm.generate(messages)
        return rewrite_response

    def parse(
        self, question: str, context: dict, table_representation: List[str]
    ) -> dict:
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

        # 读取表schema，再次生成需要的对应 field
        # prompt = self.build_fields_prompt(question, understanding["table"])
        # messages = [
        #     {"role": "system", "content": prompt},
        #     {"role": "user", "content": ""},
        # ]
        # response = self.llm.generate(messages)
        # fields = self.parse_understanding(response)

        # 从all_tables_schema中提取表结构信息
        table_info = {}
        current_table = None
        lines = self.all_tables_schema.split('\n')
        
        for line in lines:
            # 处理表头
            if line.startswith('==='):
                # 提取表名
                table_name = line.split()[0][4:]  # 去掉=== 获取表名
                current_table = table_name
                table_info[current_table] = {
                    'fields': []
                }
                continue

            # 跳过列标题行和分隔线
            if not current_table or '注释' in line or '-' * 20 in line or not line.strip():
                continue

            # 解析字段行
            parts = line.split(None, 2) # 按空格分割,最多分2次
            if len(parts) >= 2:
                field_name = parts[0].strip()
                field_comment = parts[1].strip()
                
                table_info[current_table]['fields'].append({
                    'field': field_name,
                    'comment': field_comment  
                })
                
        simplified_fields = {}
        # 根据理解的表和字段生成需要的fields
        for table in understanding["table"]:
            table_name = table["name"]
            if table_name in self.table_name_map:
                table_name_en = self.table_name_map[table_name]
                # 获取表的字段信息
                if table_name_en in table_info:
                    for field in table_info[table_name_en]['fields']:
                        simplified_fields['table_name'] = {
                            'field_name': field['field'],
                            'field_comment': field['comment']
                        }


        return {
            "table": understanding["table"],
            "entities": understanding["entities"],
            "fields": simplified_fields
        }
    
    def build_fields_prompt(self, question: str, table_list: List[str]) -> str:
        # 根据表名包含生成字段抽取所需的必要字段
        prompt = """任务：根据用户查询和给定的表结构信息，选择相关字段

系统角色设定：
你是一个专业的数据分析助手，擅长理解用户意图并从复杂的数据结构中选择最相关的字段。你会进行多轮思考，每一轮都可能发现新的见解。

输入信息：
1. 用户查询: <<query>>
2. 表结构信息：
<<tables_info>>

格式为：{
    "table1": {
        "description": "表的描述",
        "fields": [
            {"field": "字段名", "description": "字段描述"},
            ...
        ]
    },
    "table2": {
        "description": "表的描述",
        "fields": [
            {"field": "字段名", "description": "字段描述"},
            ...
        ]
    },
    ...
}

思考过程：
第一轮思考：
1. 分析用户查询
   - 提取核心关键词
   - 识别查询目标
   - 定位主要数据需求

2. 初步表选择
   - 根据表描述匹配相关表
   - 评估表之间的关联性
   - 记录初步发现

3. 字段初筛
   - 根据关键词匹配相关字段
   - 标记确定必需的字段
   - 记录待确认的字段

第二轮思考：
1. 深入分析
   - 考虑潜在的数据关联
   - 评估是否遗漏关键信息
   - 检查表间关系是否完整

2. 字段关系分析
   - 验证字段间的逻辑关系
   - 补充必要的关联字段
   - 移除冗余字段

...

输出要求：
必须输出标准的 JSON 格式，包含以下内容：
{
    "thinking_process": {
        "round1": {
            "keywords": ["关键词1", "关键词2", ...],
            "initial_findings": "初步发现描述", 
            "selected_tables": ["表1", "表2", ...]
        },
        "round2": {
            "new_insights": "新的见解描述",
            "field_relationships": "字段关系分析",
            "adjustments": "调整说明"
        },
        ...
    },
    "selected_fields": {
        "table_name1": {
            "fields": [
                {
                    "field_name": "字段名1",
                    "reason": "选择理由"
                },
                ...
            ]
        },
        "table_name2": {
            "fields": [
                {
                    "field_name": "字段名1", 
                    "reason": "选择理由"
                },
                ...
            ]
        }
    }
}

<|finish|>

注意：
1. 输出必须是合法的 JSON 格式
2. 所有字段名和表名必须与输入信息中的名称完全匹配
3. 必须包含完整的思考过程记录
4. 输出完成请以 <|finish|> 结束
"""
        # 构造 table_info 
        table_info = {}
        for table_name in table_list:
            
            # 获取库表名英文
            table_name_en = self.table_name_map[table_name]
            # 获取表描述
            description = self.data_frame.set_index('库表名中文')['表描述'].to_dict().get(table_name, "")
            # 获取字段信息
            table_fields = self.table_dict.get(table_name_en, [])
            
            table_info[table_name] = {
                "description": description,
                "fields": [
                    {"field": field["name"], "description": field["comment"]}
                    for field in table_fields
                ]
            }

        # 拼接 prompt，替换 <<query>> 和 <<tables_info>>
        prompt = prompt.replace(
            "<<query>>", question
        ).replace("<<tables_info>>", json.dumps(table_info, ensure_ascii=False, indent=2))
        return prompt

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
        context_list = context.get("history", [])
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
        prompt_template = prompt_template.replace("<<question>>", question).replace(
            "<<context_list>>", " ".join(context_list)
        )
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
            code_block_pattern = r"```(?:json)?([\s\S]*?)```"
            code_block_match = re.search(code_block_pattern, response)

            if code_block_match:
                # 如果找到```json```格式，提取其中的内容
                json_str = code_block_match.group(1).strip()
            else:
                # 否则尝试直接匹配JSON对象
                json_pattern = r"({[\s\S]*})"
                json_match = re.search(json_pattern, response)
                if not json_match:
                    return get_empty_result()
                json_str = json_match.group(1)

            # 尝试解析JSON字符串
            data = json.loads(json_str)

            # 提取表、实体和字段信息
            result = {
                "table": [
                    {"name": table["table_name"], "reason": table["reason"]}
                    for table in data.get("required_tables", [])
                ],
                "entities": data.get("entities", {}),
            }
            return result

        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            # 任何解析错误都返回空结果
            return get_empty_result()

if __name__ == "__main__":
    # TODO 完成单元测试
    parser_agent = ParserAgent(None, None, None, None, None)
    question = "查询近一个月最高价"
    context = {
        "history": ["用户：查询近一个月最高价", "助手：好的，请稍等，正在查询中..."]
    }
