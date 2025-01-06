class TableVectorStore:
    def __init__(self):
        self.vectors = []  # 存储向量数据
    
    def building_vector(self):
        """Initialize the vector store by creating an empty list"""

        
        
        
        
    def add_vector(self, text: str, vector: List[float], metadata: dict):
        self.vectors.append({
            'text': text,
            'vector': vector,
            'metadata': metadata
        })
    
    def search(self, query_vector: List[float], top_k: int = 5) -> List[dict]:
        # 简单的向量相似度搜索
        scores = [(i, cosine_similarity(query_vector, item['vector']))
                 for i, item in enumerate(self.vectors)]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return [self.vectors[i] for i, _ in scores[:top_k]]