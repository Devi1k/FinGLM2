import json
from FinanceQASystem import FinanceQASystem

if __name__ == "__main__":
    question_file_path = "assets/question.json"
    finance_qa_system = FinanceQASystem()
    with open(question_file_path, "r", encoding="utf-8") as file:
        q_json_list = json.load(file)
    for q_json in q_json_list:
        finance_qa_system.process_question(q_json["id"], q_json["question"])
    # TODO：导出结果

      