from openai import OpenAI

class DialogueManager:
    """
    对话管理,涉及大模型需要的 messages 上下文
    """
    def __init__(self):
        self.history = {}  # 存储对话历史
        
    def get_context(self, question_id: str) -> dict:
        # 获取相关的历史对话
        dialogue_history = self.history.get(question_id.split('-')[0], [])
        return {
            'history': dialogue_history,
            'current_id': question_id
        }
    
    def update_history(self, question_id: str, question: str, answer: str):
        # 更新对话历史
        dialogue_id = question_id.split('-')[0]
        if dialogue_id not in self.history:
            self.history[dialogue_id] = []
        self.history[dialogue_id].append({
            'question_id': question_id,
            'question': question,
            'answer': answer
        })