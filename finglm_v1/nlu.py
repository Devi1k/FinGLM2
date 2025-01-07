from pydantic import fields
from typing import List


class ParserAgent:
    def __init__(self, llm_client, vector_store):
        self.llm = llm_client
        self.vector_store = vector_store

    def parse(self, question: str, context: dict, table_name: List[str]) -> dict:
        # Embedding 根据表名，表解释检索需要的表

        # 改写问题，增强实体链指能力
        if context:
            prompt = self.build_rewrite_prompt(question, context)
            messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': ''}]
            rewrite_response = self.llm.generate(messages)
            question = rewrite_response

        table_indices = self.vector_store.search(question)
        table_list = [table_name[idx] for idx in table_indices]


        # 构建 prompt 抽取实体，表名
        prompt = self.build_understanding_prompt(question, table_list)

        # 调用LLM生成实体以及对应的表
        messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': ''}]
        response = self.llm.generate(messages)

        # 解析响应
        understanding = self.parse_understanding(response)

        # 读取表schema，再次生成需要的对应 field
        prompt = self.build_fields_prompt(understanding["table"])
        messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': ''}]
        response = self.llm.generate(messages)
        fields = self.parse_fields(response)

        return {
            "table": understanding["table"],
            "entities": understanding["entities"],
            "fields": fields["fields"],
        }

    def build_understanding_prompt(
        self, question: str, table_list: List[str]
    ) -> str:
        # TODO: 同时完成实体识别和表名识别

        entities = []
        entities_str = self.entities_recognition(question)

        prompt_template = """
        我有如下数据库表{'库表名': <<table_list>>}
已查询获得事实：<<fact_1>>
我想回答问题"<<question>>"
        """
        prompt = prompt_template.replace("<<table_list>>", "\n".join(table_list)).replace(
            "<<question>>", question)
        return prompt


    def build_rewrite_prompt(self, question: str, context: dict) -> str:
        pass
    

    def parse_understanding(self, response: str) -> dict:
        pass

    def entities_recognition(self, question: str) -> List[str]:
        prompt = """你将会进行命名实体识别任务，并输出实体json，主要识别以下几种实体：
    公司名称，代码，基金名称。

    其中，公司名称可以是全称，简称，拼音缩写，代码包含股票代码和基金代码，基金名称包含债券型基金，
    以下是几个示例：
    user:唐山港集团股份有限公司是什么时间上市的（回答XXXX-XX-XX）
    当年一共上市了多少家企业？
    这些企业有多少是在北京注册的？
    assistant:```json
    [{"公司名称":"唐山港集团股份有限公司"}]
    ```
    user:JD的职工总数有多少人？
    该公司披露的硕士或研究生学历（及以上）的有多少人？
    20201月1日至年底退休了多少人？
    assistant:```json
    [{"公司名称":"JD"}]
    ```
    user:600872的全称、A股简称、法人、法律顾问、会计师事务所及董秘是？
    该公司实控人是否发生改变？如果发生变化，什么时候变成了谁？是哪国人？是否有永久境外居留权？（回答时间用XXXX-XX-XX）
    assistant:```json
    [{"代码":"600872"}]
    ```
    user:华夏鼎康债券A在2019年的分红次数是多少？每次分红的派现比例是多少？
    基于上述分红数据，在2019年最后一次分红时，如果一位投资者持有1000份该基金，税后可以获得多少分红收益？
    assistant:```json
    [{"基金名称":"华夏鼎康债券A"}]
    ```
    user:化工纳入过多少个子类概念？
    assistant:```json
    []
    ```
        """
        messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': question}]
        return self.llm.generate(messages)