"""
向量存储模块

This module provides vector storage and similarity search capabilities.
"""

import os
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import json
import numpy as np
from datetime import datetime
import asyncio
import pickle

from pydantic import BaseModel, Field
import faiss
from sentence_transformers import SentenceTransformer
import numpy.typing as npt

from finglm_v1.config.settings import settings

logger = logging.getLogger(__name__)


class VectorSearchResult(BaseModel):
    """向量搜索结果模型"""

    index: int = Field(..., description="原始数据索引")
    score: float = Field(..., description="相似度得分")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class VectorStoreMetadata(BaseModel):
    """向量存储元数据模型"""

    created_at: datetime = Field(default_factory=datetime.now)
    dimension: int = Field(..., description="向量维度")
    count: int = Field(..., description="向量数量")
    model_name: str = Field(..., description="编码模型名称")
    extra: Dict[str, Any] = Field(default_factory=dict, description="额外信息")


class TableVectorStore:
    """表向量存储

    提供高效的向量存储和相似度搜索功能：
    1. 支持多种索引类型
    2. 异步操作支持
    3. 元数据管理
    4. 增量更新
    5. 持久化存储

    Attributes:
        index: FAISS索引实例
        embedding_model: 句子编码模型
        metadata: 向量存储元数据
    """

    def __init__(
        self,
        texts: Optional[List[str]] = None,
        *,
        model_name: Optional[str] = None,
        index_type: str = "flat",
        dimension: Optional[int] = None,
        load_path: Optional[Path] = None,
        device: str = "cpu",
    ) -> None:
        """初始化向量存储

        Args:
            texts: 文本列表
            model_name: 编码模型名称
            index_type: 索引类型 ("flat", "ivf", "hnsw")
            dimension: 向量维度
            load_path: 加载路径
            device: 设备类型

        Raises:
            ValueError: 参数错误
            RuntimeError: 初始化失败
        """
        try:
            self.device = device
            self.model_name = model_name or settings.VECTOR_STORE_MODEL

            # 初始化编码模型
            logger.info(f"Loading embedding model: {self.model_name}")
            self.embedding_model = SentenceTransformer(
                os.path.join(settings.BASE_DIR, "model_path", self.model_name)
            )
            if device == "cuda":
                self.embedding_model.to("cuda")

            # 加载或创建索引
            if load_path:
                self._load(load_path)
            else:
                if not texts:
                    raise ValueError("Either texts or load_path must be provided")
                self._create_index(texts, index_type, dimension)

            logger.info(
                f"Initialized TableVectorStore with {self.metadata.count} vectors, "
                f"dimension={self.metadata.dimension}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise RuntimeError(f"Vector store initialization failed: {str(e)}")

    def _create_index(
        self, texts: List[str], index_type: str, dimension: Optional[int] = None
    ) -> None:
        """创建FAISS索引

        Args:
            texts: 文本列表
            index_type: 索引类型
            dimension: 向量维度
        """
        # 编码文本
        vectors = self.encode(texts)
        dimension = vectors.shape[1]

        # 创建索引
        if index_type == "flat":
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "ivf":
            nlist = min(int(np.sqrt(len(texts))), 100)  # 聚类中心数量
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.index.train(vectors)  # pyright: ignore
        elif index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(dimension, 32)  # 32为M参数
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

        # 添加向量
        self.index.add(vectors)  # pyright: ignore

        # 创建元数据
        self.metadata = VectorStoreMetadata(
            dimension=dimension,  # pyright: ignore
            count=len(texts),
            model_name=self.model_name,
        )

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """编码文本

        Args:
            texts: 单个文本或文本列表

        Returns:
            np.ndarray: 向量数组
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.embedding_model.encode(
            texts, convert_to_tensor=False, normalize_embeddings=True
        )
        return embeddings.astype(np.float32)

    def search(
        self, query: str, k: int = 5, threshold: float = 0.6
    ) -> List[VectorSearchResult]:
        """搜索相似向量

        Args:
            query: 查询文本
            k: 返回结果数量
            threshold: 相似度阈值

        Returns:
            List[VectorSearchResult]: 搜索结果列表
        """
        try:
            # 编码查询
            query_vector = self.encode(query)

            # 搜索
            distances, indices = self.index.search(
                query_vector.reshape(1, -1).astype(np.float32), k
            )  # pyright: ignore

            # 转换距离为相似度分数
            scores = 1 / (1 + distances[0])

            # 过滤低于阈值的结果
            results = []
            for idx, score in zip(indices[0], scores):
                if score >= threshold:
                    results.append(
                        VectorSearchResult(
                            index=int(idx),
                            score=float(score),
                            metadata=self.metadata.extra.get("text_metadata", {}).get(
                                str(idx), {}
                            ),
                        )
                    )
            return results

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    async def add_texts(
        self, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[int]:
        """添加文本到索引

        Args:
            texts: 文本列表
            metadata: 元数据列表

        Returns:
            List[int]: 新增向量的索引列表
        """
        try:
            # 编码文本
            vectors = self.encode(texts)

            # 添加到索引
            start_idx = self.metadata.count
            self.index.add(vectors.astype(np.float32))  # pyright: ignore

            # 更新元数据
            new_indices = list(range(start_idx, start_idx + len(texts)))
            self.metadata.count += len(texts)

            if metadata:
                text_metadata = self.metadata.extra.get("text_metadata", {})
                for idx, meta in zip(new_indices, metadata):
                    text_metadata[str(idx)] = meta
                self.metadata.extra["text_metadata"] = text_metadata

            return new_indices

        except Exception as e:
            logger.error(f"Failed to add texts: {str(e)}")
            return []

    def save(self, save_dir: Path) -> None:
        """保存向量存储

        Args:
            save_dir: 保存目录
        """
        try:
            save_dir.mkdir(parents=True, exist_ok=True)

            # 保存索引
            index_path = save_dir / "faiss_index.bin"
            faiss.write_index(self.index, str(index_path))

            # 保存元数据
            metadata_path = save_dir / "metadata.json"
            metadata_path.write_text(
                self.metadata.model_dump_json(indent=2), encoding="utf-8"
            )

            logger.info(f"Saved vector store to {save_dir}")

        except Exception as e:
            logger.error(f"Failed to save vector store: {str(e)}")
            raise

    def _load(self, load_path: Path) -> None:
        """加载向量存储

        Args:
            load_path: 加载路径
        """
        try:
            # 加载索引
            index_path = load_path / "faiss_index.bin"
            self.index = faiss.read_index(str(index_path))

            # 加载元数据
            metadata_path = load_path / "metadata.json"
            metadata_dict = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.metadata = VectorStoreMetadata(**metadata_dict)

            logger.info(f"Loaded vector store from {load_path}")

        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
            raise

    def __len__(self) -> int:
        """返回向量数量"""
        return self.metadata.count
