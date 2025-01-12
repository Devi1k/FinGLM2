"""
金融问答系统运行脚本

This script runs the FinanceQASystem with concurrent question processing.
"""
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from finglm_v1.config.settings import settings
from finglm_v1.core.types import QuestionContext
from finglm_v1.core.system import FinanceQASystem
from finglm_v1.utils.logging import setup_logging

logger = logging.getLogger(__name__)

class QuestionProcessor:
    """问题处理器类
    
    负责并发处理问题并保存结果
    
    Attributes:
        qa_system: FinanceQASystem实例
        max_workers: 最大工作线程数
        output_dir: 输出目录
    """
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        output_dir: Optional[Path] = None
    ) -> None:
        """初始化问题处理器
        
        Args:
            max_workers: 最大工作线程数，默认为CPU核心数
            output_dir: 输出目录路径，默认为'outputs'
        """
        self.qa_system = FinanceQASystem()
        self.max_workers = max_workers or settings.MAX_WORKERS
        self.output_dir = output_dir or Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        
    async def process_all_tids(self, data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """并发处理所有TID组
        
        Args:
            data: 问题数据，包含多个TID组
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: 处理结果，按TID组织
        """
        # 创建每个TID的处理任务
        tasks = [
            self.process_single_tid(item["tid"], item["team"])
            for item in data
        ]
        
        # 并发执行所有TID的处理
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 整理结果
        processed_results = {}
        for tid_data, result in zip(data, results):
            tid = tid_data["tid"]
            if isinstance(result, Exception):
                logger.error(f"Failed to process TID {tid}: {str(result)}")
                processed_results[tid] = [{
                    "id": q["id"],
                    "question": q["question"],
                    "answer": f"处理失败: {str(result)}"
                } for q in tid_data["team"]]
            else:
                processed_results[tid] = result
                
        return processed_results
        
    async def process_single_tid(
        self,
        tid: str,
        questions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """处理单个TID下的所有问题
        
        Args:
            tid: 问题组ID
            questions: 问题列表
            
        Returns:
            List[Dict[str, Any]]: 处理结果列表
        """
        results = []
        context = None  # 初始化上下文
        
        try:
            # 按顺序处理问题，保持对话上下文
            for question in questions:
                answer = await self._process_single_question(
                    question["id"],
                    question["question"],
                    context
                )
                
                result = {
                    "id": question["id"],
                    "question": question["question"],
                    "answer": answer
                }
                results.append(result)
                
                # 更新上下文
                if context is None:
                    context = QuestionContext()
                context.history.append({
                    "human": question["question"],
                    "assistant": answer
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Error processing TID {tid}: {str(e)}")
            raise
            
    async def _process_single_question(
        self,
        question_id: str,
        question: str,
        context: Optional[QuestionContext] = None
    ) -> str:
        """处理单个问题
        
        Args:
            question_id: 问题ID
            question: 问题内容
            context: 对话上下文
            
        Returns:
            str: 答案
        """
        try:
            return await self.qa_system.process_question(
                question_id,
                question,
                context=context
            )
        except Exception as e:
            logger.error(f"Error processing question {question_id}: {str(e)}")
            raise
            
    def save_results(self, results: Dict[str, List[Dict[str, Any]]]) -> Path:
        """保存处理结果
        
        Args:
            results: 处理结果，按TID组织
            
        Returns:
            Path: 保存的文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"results_{timestamp}.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Results saved to {output_file}")
        return output_file

async def main() -> None:
    """主函数"""
    try:
        setup_logging()
        logger.info("Starting question processing...")
        
        # 读取问题文件
        question_file = Path("assets/question.json")
        with open(question_file, "r", encoding="utf-8") as f:
            questions = json.load(f)
            
        # 创建处理器并处理问题
        processor = QuestionProcessor()
        results = await processor.process_all_tids(questions)
        
        # 保存结果
        output_file = processor.save_results(results)
        logger.info(f"Processing completed. Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())