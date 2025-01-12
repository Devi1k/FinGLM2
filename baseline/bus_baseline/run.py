import pandas as pd
import jieba
import re
import requests
import json
from zhipuai import ZhipuAI
from collections import Counter
from tqdm import tqdm
import time


Access_Token = 'd3cc78289acf4de296fba2b6d56a52eb'  # Competition team Token, used to access the competition database
MODEL = "glm-4-plus"  # Default large model used; this solution uses the GLM-4-PLUS model entirely
client = ZhipuAI(api_key='d25e7ccdd599490bb3113ad6ac8e515f.G0cQzHFMidedbXF5')

# Preprocess the competition questions here
question_data_path = r'../../assets/question.json'
df1 = pd.read_excel('../../assets/data_dictionary.xlsx', sheet_name='库表关系')
df2 = pd.read_excel('../../assets/data_dictionary.xlsx', sheet_name='表字段信息')
file_path = '../../assets/all_tables_schema.txt'

df1['库表名英文'] = df1['库名英文'] + '.' + df1['表英文']
df1['库表名中文'] = df1['库名中文'] + '.' + df1['表中文']

database_name = list(df1['库名中文'])
table_name = list(df1['表中文'])
table_name_en = list(df1['表英文'])
database_table_ch = list(df1['库表名中文'])
database_table_en = list(df1['库表名英文'])
database_table_en_zs = {'库表名': database_table_en, '对应中文注释说明': table_name}
database_table_map = df1.set_index('库表名中文')['库表名英文'].to_dict()

database_L = []
database_L_zh = []
for i in table_name_en:
    df3 = df2[df2['table_name'] == i]
    name = df1[df1['表英文'] == i]['库表名英文'].iloc[0]
    column_name = list(df3['column_name'])
    column_name_zh = list(df3['column_description'])
    column_name_2 = list(df3['注释'].dropna())

    dict_1 = {'数据表名': name, '列名': column_name, '注释': column_name_2}
    dict_2 = {'数据表名': name, '列名': column_name, '列名中文描述': column_name_zh, '注释': column_name_2}
    database_L.append(dict_1)
    database_L_zh.append(dict_2)




def create_chat_completion(messages, model=MODEL):
    """
    Create a chat completion using the provided messages and model.
    
    Parameters:
        messages (list): A list of message dictionaries to pass to the model.
        model (str): The model name to use.
    
    Returns:
        response (dict): The response from the chat completion endpoint.
    """
    response = client.chat.completions.create(
        model=model,
        stream=False,
        messages=messages
    )
    return response


def filter_table_comments(question, table_comments):
    """
    Filter a list of table comments based on the given question. 
    Uses jieba for segmentation and removes stopwords, returning only comments 
    that contain at least one of the segmented keywords.
    
    Parameters:
        question (str): The question text.
        table_comments (list): A list of comment strings to filter.
    
    Returns:
        filtered_comments (list): Filtered list of comments.
    """
    stopwords = ['？', '有', '的', '多少', '人', '（', '）']
    seg_list = list(jieba.cut(question, cut_all=False))
    filtered_seg_list = [word for word in seg_list if word not in stopwords]

    filtered_comments = []
    for comment in table_comments:
        if any(keyword in comment for keyword in filtered_seg_list):
            filtered_comments.append(comment)
    return filtered_comments


with open(file_path, 'r', encoding='utf-8') as file:
    content = file.read()
input_text = content


def parse_table_structures(input_text):
    """
    Parse the input text to extract table structures. 
    
    The format is expected as pairs: "table_name === table_structure".
    
    Parameters:
        input_text (str): The raw text containing table structures.
        
    Returns:
        tables_dict (dict): A dictionary where keys are table names and 
                            values are the associated table structures.
    """
    tables_text = input_text.split('===')[1:]
    tables_dict = {tables_text[i]: tables_text[i + 1] for i in range(0, len(tables_text), 2)}
    return tables_dict


def map_chinese_to_english_tables(chinese_names, english_names):
    """
    Map Chinese table names to their corresponding English table names.
    For each Chinese name, there is a matching English name 
    (case-insensitive comparison).
    
    Parameters:
        chinese_names (list): A list of Chinese table names.
        english_names (list): A list of English table names.
        
    Returns:
        name_map (dict): A dictionary mapping Chinese table names to English table names.
    """
    name_map = {}
    for cname in chinese_names:
        # Find the corresponding English name (case-insensitive match)
        english_match = [en for en in english_names if str(en).lower() == cname.lower()][0]
        name_map[cname] = english_match
    return name_map


def find_value_in_list_of_dicts(dict_list, key_to_match, value_to_match, key_to_return):
    """
    Search through a list of dictionaries and find the first dictionary where 
    the value of key_to_match equals value_to_match, then return the value 
    associated with key_to_return.
    
    Parameters:
        dict_list (list): A list of dictionaries to search through.
        key_to_match (str): The key whose value we want to match.
        value_to_match (str): The value we are looking for.
        key_to_return (str): The key whose value we want to return.
        
    Returns:
        (str): The value associated with key_to_return in the matching dictionary, 
               or an empty string if no match is found.
    """
    for dictionary in dict_list:
        if dictionary.get(key_to_match) == value_to_match:
            return dictionary.get(key_to_return)
    return ''


def get_table_schema(question=''):
    """
    Retrieve table schemas along with optional filtered field comments.
    If a question is provided, the comments will be filtered based on 
    question keywords.
    
    The function:
      1. Maps Chinese table names to English table names.
      2. For each table, retrieves its structure and finds associated comments.
      3. If a question is provided, filter the comments based on keywords extracted from the question.
    
    Parameters:
        question (str): The question text. If empty, no filtering is performed.
        
    Returns:
        table_maps (list): A list of dictionaries, each containing table schema information.
        {
            '数据表名': EnglishTableName,
            '数据表结构': TableStructure,
            '字段注释': FilteredComments (optional if question is provided)
        }
    """

    parsed_tables = parse_table_structures(input_text)

    # Clean up keys and values
    cleaned_tables = {
        k.replace(' ', '').replace('表结构', ''): v.replace('--', '')
        for k, v in parsed_tables.items()
    }

    # List of Chinese table names (keys)
    chinese_table_names = list(cleaned_tables.keys())

    name_map = map_chinese_to_english_tables(chinese_table_names, database_table_en)

    table_maps = []
    for cname, structure in cleaned_tables.items():
        english_name = name_map.get(cname)
        comments = find_value_in_list_of_dicts(database_L, '数据表名', english_name, '注释')

        if question == '':
            # No filtering, just return table name and structure
            table_map = {
                '数据表名': english_name,
                '数据表结构': structure
            }
        else:
            # Filter comments based on question
            filtered_comments = filter_table_comments(question, comments)
            table_map = {
                '数据表名': english_name,
                '数据表结构': structure,
                '字段注释': filtered_comments
            }

        table_maps.append(table_map)

    return table_maps


def find_json(text):
    """
    Attempt to extract and parse a JSON object from the provided text.
    The function tries up to three attempts using two patterns:
      1. A Markdown code block with ```json ... ```
      2. A more general JSON-like pattern using { and }

    If successful, returns the parsed JSON data.
    If parsing fails after all attempts, returns the original text.
    
    Parameters:
        text (str): The input text from which to extract JSON.
    
    Returns:
        dict or str: Parsed JSON dictionary if successful, else the original text.
    """
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        json_pattern = r"```json\n(.*?)\n```"
        match = re.search(json_pattern, text, re.DOTALL)
        if not match:
            json_pattern2 = r"({.*?})"
            match = re.search(json_pattern2, text, re.DOTALL)

        if match:
            json_string = match.group(1) if match.lastindex == 1 else match.group(0)
            # Remove Markdown formatting if present
            json_string = json_string.replace("```json\n", "").replace("\n```", "")
            try:
                data = json.loads(json_string)
                return data
            except json.JSONDecodeError as e:
                if attempt < max_attempts:
                    print(f"Attempt {attempt}: Failed to parse JSON, reason: {e}. Retrying...")
                else:
                    print(f"All {max_attempts} attempts to parse JSON failed. Returning original text.")
        else:
            if attempt < max_attempts:
                print(f"Attempt {attempt}: No JSON string found in the text. Retrying...")
            else:
                print("No matching JSON string found. Returning original text.")

        # If no match or no success in this attempt, return the original text
        return text


def dict_to_sentence(data):
    """
    Convert a dictionary into a descriptive sentence by enumerating key-value pairs.
    For example: {"name": "John", "age": 30} -> "name 是 John, age 是 30"
    
    Parameters:
        data (dict): The dictionary to convert.
        
    Returns:
        str: A sentence describing the dictionary keys and values.
    """
    try:
        if not isinstance(data, dict):
            raise ValueError("Input is not a dictionary")

        return ", ".join(f"{key} 是 {value}" for key, value in data.items())
    except Exception as e:
        print(f"Error in dict_to_sentence: {e}")
        return str(data)


def process_dict(d):
    """
    Recursively process a nested dictionary to produce a comma-separated description.
    For nested dictionaries, it processes them recursively and returns a descriptive string.
    
    For example:
        {
            "company": {
                "name": "ABC Corp",
                "location": "New York"
            },
            "year": 2021
        }
    might be processed into a string like:
        "company company 是 name 是 ABC Corp, location 是 New York, year 2021"
    
    Parameters:
        d (dict): A dictionary or another object to describe.
        
    Returns:
        str: A descriptive string.
    """

    def recursive_process(sub_dict):
        sentences = []
        for key, value in sub_dict.items():
            if isinstance(value, dict):
                # Process nested dictionary and wrap result in dict_to_sentence for formatting
                nested_result = recursive_process(value)
                sentences.append(dict_to_sentence({key: nested_result}))
            else:
                # Non-dict values are directly appended
                sentences.append(f"{key} {value}")
        return ", ".join(sentences)

    if not isinstance(d, dict):
        # If it's not a dictionary, just return its string representation
        return str(d)

    return recursive_process(d)


def run_conversation(question):
    """
    Run a conversation flow given a question by:
      1. Using run_conversation_xietong(question) to get an answer.
      2. Attempting to find and parse JSON from the answer using find_json.
      3. Converting the parsed result (or original text if parsing fails) into a descriptive sentence using process_dict.
    
    Parameters:
        question (str): The question to ask.
        
    Returns:
        str: The final processed answer as a descriptive string.
    """
    last_answer = run_conversation_xietong(question)
    parsed_data = find_json(last_answer)
    final_string = process_dict(parsed_data)
    return final_string


def clean_text(text):
    """
    Remove any parenthetical segments (including Chinese parentheses) and trim whitespace.
    For example, "This is a sentence(remark)" -> "This is a sentence"
    
    Parameters:
        text (str): The text to clean.
        
    Returns:
        str: The cleaned text.
    """
    pattern = r'[\(（][^\)）]*[\)）]'  # Pattern to match parentheses and their contents
    cleaned_text = re.sub(pattern, '', text).strip()
    return cleaned_text


def find_dict_by_element(dict_list, target_element):
    """
    Given a list of dictionaries, return all dictionaries where  '列名中文描述' contains the target_element.
    Parameters:
        dict_list (list): A list of dictionaries, each expected to have '列名中文描述' key.
        target_element (str): The element to search for.
        
    Returns:
        list: A list of dictionaries that contain target_element in '列名中文描述'.
    """
    return [d for d in dict_list if target_element in d.get('列名中文描述', [])]


def to_get_question_columns(question):
    """
    Given a question (string) and a global variable database_L_zh (list of dicts),
    find 列名 that correspond to 列名中文描述 mentioned in the question. 
    
    If any matching columns are found, return a message instructing the user to 
    use these column names directly for data querying. If none are found, return an empty string.
    
    Parameters:
        question (str): The input question text.
        
    Returns:
        str: A message with identified column names or an empty string if none found.
    """
    global database_L_zh
    L_num = []
    for items in database_L_zh:
        L_num += items['列名中文描述']

    # Get unique column descriptions
    L_num_new = [item for item, count in Counter(L_num).items() if count == 1]

    # Drop NaN if any
    series_num = pd.Series(L_num_new)
    L_num_new = list(series_num.dropna())

    # Remove known irrelevant items
    irrelevant_items = ['年度', '占比']
    for irr in irrelevant_items:
        if irr in L_num_new:
            L_num_new.remove(irr)

    matched_columns = []
    for col_desc in L_num_new:
        # Check if the column description or its cleaned version appears in the question
        if col_desc in question or clean_text(col_desc) in question:
            L_dict = find_dict_by_element(database_L_zh, col_desc)
            if not L_dict:
                continue
            # Create a mapping from Chinese description to English column name
            dict_zip = dict(zip(L_dict[0]['列名中文描述'], L_dict[0]['列名']))
            column_name = dict_zip[col_desc]
            data_table = L_dict[0]['数据表名']

            matched_columns.append({
                '数据库表': data_table,
                '列名': column_name,
                '列名中文含义': col_desc
            })

    if matched_columns:
        return f"已获得一部分数据库列名{matched_columns}，请充分利用获得的列名直接查询数据。"
    else:
        return ''


def replace_date_with_day(sql):
    """
    This function replaces instances of exact date conditions in a SQL 
    statement from a format like:
        TradingDate = 'YYYY-MM-DD'
    to:
        date(TradingDate) = 'YYYY-MM-DD'
    
    Parameters:
        sql (str): The original SQL statement.
        
    Returns:
        str: The modified SQL statement, or the original if no match is found.
    """
    # Regex pattern to match patterns like: ColumnName = 'YYYY-MM-DD'
    pattern = r"([.\w]+)\s*=\s*'(\d{4}-\d{2}-\d{2})'"

    def replace_func(match):
        column_name = match.group(1)
        date_value = match.group(2)
        return f"date({column_name}) = '{date_value}'"

    new_sql = re.sub(pattern, replace_func, sql)

    # If no change was made, return the original SQL
    return new_sql if new_sql != sql else sql


def extract_sql(text):
    """
    Extracts an SQL statement from a block of text enclosed in triple backticks:
        ```sql
        SELECT ...
        ```
    
    Parameters:
        text (str): The full text containing an SQL statement.
        
    Returns:
        str: The extracted SQL statement, or a message if not found.
    """
    sql_pattern = re.compile(r'```sql(.*?)```', re.DOTALL)
    match = sql_pattern.search(text)
    if match:
        # Strip leading and trailing whitespace from the matched SQL
        return match.group(1).strip()
    else:
        return "No SQL statement found."


def select_data(sql_text):
    """
    Sends the given SQL query to a specified endpoint and returns the JSON response.
    
    Parameters:
        sql_text (str): The SQL query to be executed.
        
    Returns:
        str: The JSON response from the API, formatted with indentation.
    """
    url = "https://comm.chatglm.cn/finglm2/api/query"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f'Bearer {Access_Token}'
    }
    data = {
        "sql": sql_text,  # e.g. SELECT * FROM constantdb.secumain LIMIT 10
        "limit": 15
    }
    response = requests.post(url, headers=headers, json=data)
    try:
        return json.dumps(response.json(), indent=2, ensure_ascii=False)
    except:
        return str(response.json())


def to_select(text):
    """
    High-level function that:
      1. Extracts SQL from the given text.
      2. Optimizes the extracted SQL by converting date columns to 'date(...)'.
      3. Executes the optimized SQL through select_data and returns the result.
    
    Parameters:
        text (str): The input text containing an SQL statement.
        
    Returns:
        str: The JSON response from the SQL query.
    """
    sql_statement = extract_sql(text)
    print('***********Extracted SQL****************')
    print(sql_statement)
    print('***********Extracted SQL****************')
    optimized_sql = replace_date_with_day(sql_statement)
    result = select_data(optimized_sql)
    return result



def exec_sql_s(sql):
    """
    Execute a given SQL query on a remote endpoint and return the result.
    Uses 'Access_Token' for authorization and limits the result to 10 rows.

    Parameters:
        sql (str): The SQL query to be executed.

    Returns:
        list: The query result as a list of rows (dictionaries), or None if not found.
    """
    headers = {
        "Authorization": f'Bearer {Access_Token}',
        "Accept": "application/json"
    }
    url = "https://comm.chatglm.cn/finglm2/api/query"

    response = requests.post(url, headers=headers, json={
        "sql": sql,
        "limit": 10
    })
    response_json = response.json()

    # If there's no 'data' field, print the full response for debugging
    if 'data' not in response_json:
        print(response_json)

    # Return 'data' if present
    return response_json.get('data', None)


def output_result(result, info_list):
    """
    Append the formatted JSON 'data' from the result into 'info_list'.

    Parameters:
        result (dict): The query result containing 'data'.
        info_list (list): The list to which formatted data will be appended.

    Returns:
        list: The updated info_list with the new data appended, if any.
    """
    if 'data' in result and len(result['data']) > 0:
        info_list.append(json.dumps(result['data'], ensure_ascii=False, indent=1) + '\n')
    return info_list


def process_company_name(value):
    """
    Given a company name (or related keyword), search in three tables:
    ConstantDB.SecuMain, ConstantDB.HK_SecuMain, ConstantDB.US_SecuMain.

    Attempts to match various company-related fields (e.g., ChiName, EngName, etc.)
    and returns all matching results along with the table where they were found.

    Parameters:
        value (str): The company name or related string to match.

    Returns:
        list: A list of tuples (result, table) where result is the matched data and table is the table name.
              If no matches found, prints a message and returns an empty list.
    """
    res_lst = []
    tables = ['ConstantDB.SecuMain', 'ConstantDB.HK_SecuMain', 'ConstantDB.US_SecuMain']
    columns_to_match = ['CompanyCode', 'SecuCode', 'ChiName', 'ChiNameAbbr',
                        'EngName', 'EngNameAbbr', 'SecuAbbr', 'ChiSpelling']
    columns_to_select = ['InnerCode', 'CompanyCode', 'SecuCode', 'ChiName', 'ChiNameAbbr',
                         'EngName', 'EngNameAbbr', 'SecuAbbr', 'ChiSpelling']

    # Escape single quotes to prevent SQL injection
    value = value.replace("'", "''")

    for table in tables:
        # For the US table, remove columns that may not be available
        local_match_cols = columns_to_match.copy()
        local_select_cols = columns_to_select.copy()
        if 'US' in table:
            if 'ChiNameAbbr' in local_match_cols:
                local_match_cols.remove('ChiNameAbbr')
            if 'ChiNameAbbr' in local_select_cols:
                local_select_cols.remove('ChiNameAbbr')
            if 'EngNameAbbr' in local_match_cols:
                local_match_cols.remove('EngNameAbbr')
            if 'EngNameAbbr' in local_select_cols:
                local_select_cols.remove('EngNameAbbr')

        # Build the WHERE clause with OR conditions for each column
        match_conditions = [f"{col} = '{value}'" for col in local_match_cols]
        where_clause = ' OR '.join(match_conditions)

        sql = f"""
        SELECT {', '.join(local_select_cols)}
        FROM {table}
        WHERE {where_clause}
        """
        result = exec_sql_s(sql)
        if result:
            res_lst.append((result, table))
    else:
        # The 'else' clause in a for loop runs only if no 'break' was encountered.
        # Here it just prints if no results were found.
        if not res_lst:
            print(f"未在任何表中找到公司名称为 {value} 的信息。")

    return res_lst


def process_code(value):
    """
    Given a code (e.g., a stock code), search the three tables and return matches.

    Parameters:
        value (str): The code to search for.

    Returns:
        list: A list of tuples (result, table) if found, else empty.
    """
    res_lst = []
    tables = ['ConstantDB.SecuMain', 'ConstantDB.HK_SecuMain', 'ConstantDB.US_SecuMain']
    columns_to_select = ['InnerCode', 'CompanyCode', 'SecuCode', 'ChiName', 'ChiNameAbbr',
                         'EngName', 'EngNameAbbr', 'SecuAbbr', 'ChiSpelling']

    value = value.replace("'", "''")  # Escape single quotes

    for table in tables:
        local_select_cols = columns_to_select.copy()
        if 'US' in table:
            if 'ChiNameAbbr' in local_select_cols:
                local_select_cols.remove('ChiNameAbbr')
            if 'EngNameAbbr' in local_select_cols:
                local_select_cols.remove('EngNameAbbr')

        sql = f"""
        SELECT {', '.join(local_select_cols)}
        FROM {table}
        WHERE SecuCode = '{value}'
        """
        result = exec_sql_s(sql)
        if result:
            res_lst.append((result, table))
    else:
        if not res_lst:
            print(f"未在任何表中找到代码为 {value} 的信息。")

    return res_lst


def process_items(item_list):
    """
    Given a list of items (dictionaries) from JSON extraction, attempt to process each based on its key:
    - If key is '基金名称' or '公司名称', use process_company_name.
    - If key is '代码', use process_code.
    - Otherwise, print an unrecognized key message.

    Parameters:
        item_list (list): A list of dictionaries like [{"公司名称": "XX公司"}, {"代码":"600872"}].

    Returns:
        tuple: (res, tables)
               res (str): A formatted string showing what was found.
               tables (list): A list of table names where matches were found.
    """
    res_list = []
    for item in item_list:
        key, value = list(item.items())[0]
        if key in ["基金名称", "公司名称"]:
            res_list.extend(process_company_name(value))
        elif key == "代码":
            res_list.extend(process_code(value))
        else:
            print(f"无法识别的键：{key}")

    # Filter out empty results
    res_list = [i for i in res_list if i]
    res = ''
    tables = []
    for result_data, table_name in res_list:
        tables.append(table_name)
        res += f"预处理程序通过表格：{table_name} 查询到以下内容：\n {json.dumps(result_data, ensure_ascii=False, indent=1)} \n"

    return res, tables


def extract_list_from_json(json_string):
    """
    Attempt to decode a JSON string representing a list.

    Parameters:
        json_string (str): The JSON string to decode.

    Returns:
        list or None: The decoded list, or None if decoding fails or not a list.
    """
    try:
        data = json.loads(json_string)
        if isinstance(data, list):
            return data
        else:
            print("解码的JSON数据不是一个列表")
            return None
    except json.JSONDecodeError as e:
        print(f"JSON解码错误: {e}")
        return None


def process_question(question):
    """
    Given a question, run it through a prompt to perform Named Entity Recognition (NER),
    extract entities (公司名称, 代码, 基金名称), parse the assistant's JSON response,
    and process the items to retrieve relevant information from the database.

    Parameters:
        question (str): The user question.

    Returns:
        tuple: (res, tables) where
               res (str) - Processed result details as a string.
               tables (list) - List of tables involved in the final result.
    """
    prompt = '''
    你将会进行命名实体识别任务，并输出实体json，主要识别以下几种实体：
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
    '''

    messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': question}]
    aa = create_chat_completion(messages)
    bb = find_json(aa.choices[0].message.content)
    return process_items(bb)


def way_string_2(question):
    way_string_2 = to_get_question_columns(question)
    way_string_2 += ">>查询参考："
    if "近一个月最高价" in question:
        way_string_2 += "查询近一个月最高价,你写的sql语句可以优先考虑表中已有字段HighPriceRM  近一月最高价(元)  "
    if "近一个月最低价" in question:
        way_string_2 += "查询近一月最低价(元),你写的sql语句直接调用已有字段LowPriceRM"
    if "行业" in question and ('多少只' in question or '几个' in question or '多少个' in question):
        way_string_2 += """查询某行业某年数量 示例sql语句:SELECT count(*) as 风电零部件_2021
            FROM AStockIndustryDB.LC_ExgIndustry
            where ThirdIndustryName like '%风电零部件%' and year(InfoPublDate)=2021 and IfPerformed = 1;"""
    if ('年度报告' in question and '最新更新' in question) or '比例合计' in question:
        way_string_2 += """特别重要一定注意，查询最新更新XXXX年年度报告，机构持有无限售流通A股数量合计InstitutionsHoldProp最多公司代码，优先使用查询sql语句，SELECT *
                            FROM AStockShareholderDB.LC_StockHoldingSt
                            WHERE date(EndDate) = 'XXXX-12-31'
                              AND UpdateTime = (
                                SELECT MAX(UpdateTime)
                                FROM AStockShareholderDB.LC_StockHoldingSt
                                WHERE date(EndDate) = 'XXXX-12-31'
                              ) order by InstitutionsHoldings desc limit 1 ，XXXX代表问题查询年度，sql语句禁止出现group by InnerCode;

                              查询最新更新XXXX年年度报告,公司机构持有无限售流通A股比例合计InstitutionsHoldProp是多少,优先使用查询sql语句，SELECT InstitutionsHoldProp
                            FROM AStockShareholderDB.LC_StockHoldingSt
                            WHERE date(EndDate) = 'XXXX-12-31'
                              AND UpdateTime = (
                                SELECT MAX(UpdateTime)
                                FROM AStockShareholderDB.LC_StockHoldingSt
                                WHERE date(EndDate) = 'XXXX-12-31'
                              ) order by InstitutionsHoldings desc limit 1 ，XXXX代表问题查询年度，sql语句禁止出现group by InnerCode;"""

    if '新高' in question:
        way_string_2 += """新高 要用AStockMarketQuotesDB.CS_StockPatterns现有字段
        
        查询今天是2021年01月01日，创近半年新高的股票有几只。示例sql语句:SELECT count(*)  FROM AStockMarketQuotesDB.CS_StockPatterns
                where  IfHighestHPriceRMSix=1 and date(TradingDay)='2021-01-01;
                判断某日 YY-MM-DD  InnerCode XXXXXX 是否创近一周的新高，查询结果1代表是,IfHighestHPriceRW字段可以根据情况灵活调整  SELECT   InnerCode,TradingDay,IfHighestHPriceRW  FROM AStockMarketQuotesDB.CS_StockPatterns
where  date(TradingDay)='2021-12-20' and InnerCode = '311490'
                
                """
    if '成交额' in question and '平均' in question:
        way_string_2 += """查询这家公司5日内平均成交额是多少。示例sql语句:SELECT count(*)  FROM AStockMarketQuotesDB.CS_StockPatterns
                where  IfHighestHPriceRMSix=1 and date(TradingDay)='2021-01-01"""
    if '半年度报告' in question:
        way_string_2 += """查询XXXX年半年度报告的条件为：year(EndDate) = XXXX and InfoSource='半年度报告'"""

    return way_string_2

    if '新高' in question:
        way_string_2 += """查询今天是2021年01月01日，创近半年新高的股票有几只示。示例sql语句:SELECT count(*)  FROM AStockMarketQuotesDB.CS_StockPatterns
                where  IfHighestHPriceRMSix=1 and date(TradingDay)='2021-01-01"""
    if '成交额' in question and '平均' in question:
        way_string_2 += """查询这家公司5日内平均成交额是多少。示例sql语句:SELECT count(*)  FROM AStockMarketQuotesDB.CS_StockPatterns
                where  IfHighestHPriceRMSix=1 and date(TradingDay)='2021-01-01"""

    return way_string_2


def run_conversation_until_complete(messages, max_rounds=6):
    """
    Test function to run a conversation loop until the assistant indicates completion.
    """
    last_response = None  # 用于存储最后一次对话的响应
    round_count = 0  # 对话轮数计数器
    response = create_chat_completion(messages)
    while True:
        if round_count >= max_rounds:
            break  # 如果对话轮数超过最大值，则退出循环

        question = response.choices[0].message.content
        select_result = to_select(question)
        messages.append({"role": "assistant", "content": question})
        messages.append({"role": "user", "content": str(select_result)})

        response = create_chat_completion(messages)

        last_response = response.choices[0].message.content  # 存储最后一次响应       
        if "全部完成" in response.choices[0].message.content: # pyright: ignore
            break  # 如果检测到“回答完成”，则停止循环
        round_count += 1  # 增加对话轮数计数
    return last_response  # 返回最后一次对话的内容


def run_conversation_xietong(question):
    content_p_1 = """我有如下数据库表{'库表名': ['AStockBasicInfoDB.LC_StockArchives',
  'AStockBasicInfoDB.LC_NameChange',
  'AStockBasicInfoDB.LC_Business',
  'AStockIndustryDB.LC_ExgIndustry',
  'AStockIndustryDB.LC_ExgIndChange',
  'AStockIndustryDB.LC_IndustryValuation',
  'AStockIndustryDB.LC_IndFinIndicators',
  'AStockIndustryDB.LC_COConcept',
  'AStockIndustryDB.LC_ConceptList',
  'AStockOperationsDB.LC_SuppCustDetail',
  'AStockShareholderDB.LC_SHTypeClassifi',
  'AStockShareholderDB.LC_MainSHListNew',
  'AStockShareholderDB.LC_SHNumber',
  'AStockShareholderDB.LC_Mshareholder',
  'AStockShareholderDB.LC_ActualController',
  'AStockShareholderDB.LC_ShareStru',
  'AStockShareholderDB.LC_StockHoldingSt',
  'AStockShareholderDB.LC_ShareTransfer',
  'AStockShareholderDB.LC_ShareFP',
  'AStockShareholderDB.LC_ShareFPSta',
  'AStockShareholderDB.LC_Buyback',
  'AStockShareholderDB.LC_BuybackAttach',
  'AStockShareholderDB.LC_LegalDistribution',
  'AStockShareholderDB.LC_NationalStockHoldSt',
  'AStockShareholderDB.CS_ForeignHoldingSt',
  'AStockFinanceDB.LC_AShareSeasonedNewIssue',
  'AStockFinanceDB.LC_ASharePlacement',
  'AStockFinanceDB.LC_Dividend',
  'AStockFinanceDB.LC_CapitalInvest',
  'AStockMarketQuotesDB.CS_StockCapFlowIndex',
  'AStockMarketQuotesDB.CS_TurnoverVolTecIndex',
  'AStockMarketQuotesDB.CS_StockPatterns',
  'AStockMarketQuotesDB.QT_DailyQuote',
  'AStockMarketQuotesDB.QT_StockPerformance',
  'AStockMarketQuotesDB.LC_SuspendResumption',
  'AStockFinanceDB.LC_BalanceSheetAll',
  'AStockFinanceDB.LC_IncomeStatementAll',
  'AStockFinanceDB.LC_CashFlowStatementAll',
  'AStockFinanceDB.LC_IntAssetsDetail',
  'AStockFinanceDB.LC_MainOperIncome',
  'AStockFinanceDB.LC_OperatingStatus',
  'AStockFinanceDB.LC_AuditOpinion',
  'AStockOperationsDB.LC_Staff',
  'AStockOperationsDB.LC_RewardStat',
  'AStockEventsDB.LC_Warrant',
  'AStockEventsDB.LC_Credit',
  'AStockEventsDB.LC_SuitArbitration',
  'AStockEventsDB.LC_EntrustInv',
  'AStockEventsDB.LC_Regroup',
  'AStockEventsDB.LC_MajorContract',
  'AStockEventsDB.LC_InvestorRa',
  'AStockEventsDB.LC_InvestorDetail',
  'AStockShareholderDB.LC_ESOP',
  'AStockShareholderDB.LC_ESOPSummary',
  'AStockShareholderDB.LC_TransferPlan',
  'AStockShareholderDB.LC_SMAttendInfo',
  'HKStockDB.HK_EmployeeChange',
  'HKStockDB.HK_StockArchives',
  'HKStockDB.CS_HKStockPerformance',
  'USStockDB.US_CompanyInfo',
  'USStockDB.US_DailyQuote',
  'PublicFundDB.MF_FundArchives',
  'PublicFundDB.MF_FundProdName',
  'PublicFundDB.MF_InvestAdvisorOutline',
  'PublicFundDB.MF_Dividend',
  'CreditDB.LC_ViolatiParty',
  'IndexDB.LC_IndexBasicInfo',
  'IndexDB.LC_IndexComponent',
  'InstitutionDB.LC_InstiArchive',
  'ConstantDB.SecuMain',
  'ConstantDB.HK_SecuMain',
  'ConstantDB.CT_SystemConst',
  'ConstantDB.QT_TradingDayNew',
  'ConstantDB.LC_AreaCode',
  'InstitutionDB.PS_EventStru',
  'ConstantDB.US_SecuMain',
  'InstitutionDB.PS_NewsSecurity'],
 '对应中文注释说明': ['公司概况',
  '公司名称更改状况',
  '公司经营范围与行业变更',
  '公司行业划分表',
  '公司行业变更表',
  '行业估值指标',
  '行业财务指标表',
  '概念所属公司表',
  '概念板块常量表',
  '公司供应商与客户',
  '股东类型分类表',
  '股东名单(新)',
  '股东户数',
  '大股东介绍',
  '公司实际控制人',
  '公司股本结构变动',
  '股东持股统计',
  '股东股权变动',
  '股东股权冻结和质押',
  '股东股权冻结和质押统计',
  '股份回购',
  '股份回购关联表',
  '法人配售与战略投资者',
  'A股国家队持股统计',
  '外资持股统计',
  'A股增发',
  'A股配股',
  '公司分红',
  '资金投向说明',
  '境内股票交易资金流向指标',
  '境内股票成交量技术指标',
  '股票技术形态表',
  '日行情表',
  '股票行情表现(新)',
  '停牌复牌表',
  '资产负债表_新会计准则',
  '利润分配表_新会计准则',
  '现金流量表_新会计准则',
  '公司研发投入与产出',
  '公司主营业务构成',
  '公司经营情况述评',
  '公司历年审计意见',
  '公司职工构成',
  '公司管理层报酬统计',
  '公司担保明细',
  '公司借贷明细',
  '公司诉讼仲裁明细',
  '重大事项委托理财',
  '公司资产重组明细',
  '公司重大经营合同明细',
  '投资者关系活动',
  '投资者关系活动调研明细',
  '员工持股计划',
  '员工持股计划概况',
  '股东增减持计划表',
  '股东大会出席信息',
  '港股公司员工数量变动表',
  '港股公司概况',
  '港股行情表现',
  '美股公司概况',
  '美股日行情',
  '公募基金概况',
  '公募基金产品名称',
  '公募基金管理人概况',
  '公募基金分红',
  '违规当事人处罚',
  '指数基本情况',
  '指数成份',
  '机构基本资料',
  '证券主表,包含字段InnerCode,CompanyCode,SecuCode,ChiName,ChiNameAbbr 代表中文名称缩写,EngName,EngNameAbbr,SecuAbbr 代表 证券简称,ListedDate',
  '港股证券主表，包含字段InnerCode,CompanyCode,SecuCode,ChiName,ChiNameAbbr 代表中文名称缩写,EngName,EngNameAbbr,SecuAbbr 代表 证券简称,ListedDate',
  '系统常量表',
  '交易日表(新)',
  '国家城市代码表',
  '事件体系指引表',
  '美股证券主表',
  '证券舆情表']}
已查询获得事实：<<fact_1>>
我想回答问题
"<<question>>"

如果已查询获得事实可以直接总结答案，需要是明确的答案数据不是需要查询数据库表，记得提示我：<全部完成，答案如下>,将答案总结以json格式给我。
如果不能直接总结答案，需要查询的数据库表,请从上面数据库表中筛选出还需要哪些数据库表，记得提示我：<需要查询的数据库表>,只返回需要数据列表,不要回答其他内容。"""

    content_p = content_p_1.replace('<<question>>', str(question)).replace('<<fact_1>>',
                                                                           str(process_question(question)))
    content_p = content_p + way_string_2(question)
    content_p_2 = """获取的表结构如下<list>,表结构中列名可以引用使用,表结构中数据示例只是参考不能引用。
我们现在开始查询当前问题，请你分步写出查询sql语句，我把查询结果告诉你，你再告诉我下一步，
注意如果我返回的结果为空或者错误影响下一步调用，请重新告诉我sql语句。
等你全部回答完成，不需要进行下一步调用时，记得提示我：<全部完成，答案如下>,将答案总结以json格式给我，只需要总结当前问题。
查询技巧:sql查询年度时优先使用year()函数。sql查询语句不需要注释，不然会报错。sql中日期条件格式应参考这样date(TradingDay) = 'YYYY-MM-DD'。尽量利用表格中已有的字段。"""

    # 执行函数部分
    messages = []
    messages.append({"role": "user", "content": "您好阿"})
    messages.append({"role": "user", "content": content_p})
    response = create_chat_completion(messages)
    if "全部完成" in response.choices[0].message.content: # pyright: ignore
        return response.choices[0].message.content  # 如果检测到“回答完成”，则停止循环
    messages.append({"role": "assistant", "content": response.choices[0].message.content})
    table_maps = get_table_schema(question)
    LL = [i for i in table_maps if i.get('数据表名') in response.choices[0].message.content]
    content_p_2 = content_p_2.replace('<list>', str(LL)) + way_string_2(question)
    messages.append({"role": "user", "content": content_p_2})  ###开始对话       
    last_answer = run_conversation_until_complete(messages, max_rounds=9)
    return last_answer


def get_answer(question):
    """
    Attempt to answer the given question by interacting with the 
    conversation model. If an error occurs, return a default error message.
    
    Parameters:
        question (str): The question that needs an answer.
        
    Returns:
        str: The answer string or an error message if an exception occurs.
    """
    try:
        print(f"Attempting to answer the question: {question}")
        last_answer = run_conversation(question)
        return last_answer
    except Exception as e:
        print(f"Error occurred while executing get_answer: {e}")
        return "An error occurred while retrieving the answer."


def question_rew(context_text, original_question):
    """
    Rewrite the given question to be clearer and more specific based on the provided context,
    without altering the original meaning or omitting any information.
    
    Parameters:
        context_text (str): The context text that the question is based on.
        original_question (str): The question to be rewritten.
        
    Returns:
        str: The rewritten question.
    """
    prompt = (
        f"根据这些内容'{context_text}',帮我重写这个问题”'{original_question}' ,让问题清晰明确，"
        "不改变原意，不要遗漏信息，特别是时间，只返回问题。\n"
        "以下是示例：\n"
        "user:根据这些内容'最新更新的2021年度报告中，机构持有无限售流通A股数量合计最多的公司简称是？  公司简称 帝尔激光',"
        "帮我重写这个问题'在这份报告中，该公司机构持有无限售流通A股比例合计是多少，保留2位小数？' ,让问题清晰明确，不改变原意，不要遗漏信息，特别是时间，"
        "只返回问题。\n"
        "assistant:最新更新的2021年度报告中,公司简称 帝尔激光 持有无限售流通A股比例合计是多少，保留2位小数？"
    )

    messages = [{"role": "user", "content": prompt}]
    response = create_chat_completion(messages)
    return response.choices[0].message.content


def main_answer(q_json_list, start_index=0, end_index=None):
    """
    Process a portion of a list of JSON objects, each containing a 'tid' and 'team' 
    where 'team' is a list of questions.
    
    For each JSON object in the specified range:
      1. Extract all questions from 'team'.
      2. If no previous Q&A history, use the question directly. Otherwise, 
         rewrite the question based on all previously answered content.
      3. Get the answer using get_answer and store it.
      4. Update the original structure with the answers.
    
    Parameters:
        q_json_list (list): List of data objects, each containing keys 'tid' and 'team'.
        start_index (int): The starting index of the list subset to process.
        end_index (int): The ending index (non-inclusive) of the list subset. 
                         If None, process until the end of q_json_list.
                         
    Returns:
        list: A list of processed dictionaries with updated answers.
    """
    if end_index is None or end_index > len(q_json_list):
        end_index = len(q_json_list)

    data_list_result = []
    for i in tqdm(range(start_index, end_index), desc="Processing JSON data in range"):
        item = q_json_list[i]
        start_time = time.time()

        # Extract questions
        questions_list = [(member["id"], member["question"]) for member in item["team"]]
        answers_dict = {}
        all_previous = ''

        # Iterate over all questions in the current item
        for question_id, question_text in questions_list:
            if all_previous == '':
                rewritten_question = question_text
            else:
                rewritten_question = question_rew(all_previous, question_text)

            answer = get_answer(rewritten_question)
            print(f'----------answer:{answer}')
            answers_dict[question_id] = answer
            all_previous += question_text + answer

        # Update original item with answers
        for member in item["team"]:
            member["answer"] = answers_dict.get(member["id"], "无答案")

        updated_data = {"tid": item["tid"], "team": item["team"]}
        data_list_result.append(updated_data)

        elapsed_time = time.time() - start_time
        print(f"Completed processing JSON index {i} in {elapsed_time:.2f} seconds")
        with open('result_zj.json', 'w', encoding='utf-8') as json_file:
            json.dump(data_list_result, json_file, ensure_ascii=False, indent=4)
    return data_list_result


if __name__ == "__main__":
    # Load input data
    with open(question_data_path, 'r', encoding='utf-8') as file:
        q_json_list = json.load(file)

    # Users can specify a range to process the corresponding subset of data
    # For example, from index 0 to 100 (excluding 100), processing the first 100 JSON entries
    start_idx = 0
    end_idx = 101  # Specify processing data in the range 0-101

    results = main_answer(q_json_list, start_index=start_idx, end_index=end_idx)

    # Write the processing results to a file
    with open('result.json', 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)