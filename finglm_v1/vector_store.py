import os
from typing import List
from sentence_transformers import SentenceTransformer


class TableVectorStore:
    def __init__(self, representation):

        self.embedding_model_path = "model_path/Conan-embedding-v1"
        """
        如果向量数据为空，需要初始化。否则，直接加载向量数据
        """
        self.vectors_path = "model_path/faiss_index.bin"
        self.embedding_model = SentenceTransformer(self.embedding_model_path)
        self.vectors = self.building_vector(representation)

    def building_vector(self, representation):
        """Initialize the vector store by creating an empty list"""
        if os.path.exists(self.vectors_path):
            return self.load_vectors()
        else:
            # Encode the representation
            encoded_vectors = self.embedding_model.encode(representation)

            # Initialize and add vectors to the FAISS index
            import faiss

            dimension = encoded_vectors.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(encoded_vectors)  # pyright:ignore

            # Save the FAISS index
            faiss.write_index(index, self.vectors_path)
            return index

    def load_vectors(self):
        import faiss

        index = faiss.read_index(self.vectors_path)
        return index

    def search(self, question: str, top_k: int = 5) -> List[int]:
        # 简单的向量相似度搜索
        query_vector = self.embedding_model.encode(question)
        distances, indices = self.vectors.search(query_vector.reshape(1, -1), k=top_k) # pyright: ignore
        return indices.tolist()


if __name__ == "__main__":
    import pandas as pd

    data_dictionary_path = "assets/data_dictionary.xlsx"
    all_tables_schema_path = "assets/all_tables_schema.txt"
    df1 = pd.read_excel(data_dictionary_path, sheet_name="库表关系")
    df1["库表名中文"] = df1["库名中文"] + "." + df1["表中文"]
    df1["库表名英文"] = df1["库名英文"] + "." + df1["表英文"]
    df1["representation"] = "库表名:" + df1["库表名中文"] + "。注释:" + df1["表描述"]
    vector_store = TableVectorStore(df1["representation"])
    search_res = vector_store.search("今天是2021年12月24日，创近半年新高的股票有几只？",25)
    print('\n'.join(df1.iloc[search_res[0]]["库表名中文"].tolist()))
