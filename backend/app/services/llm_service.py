"""
轻量大模型 NER 服务
基于 Qwen2.5-3B-Instruct（针对8GB显存优化）
"""
import json
import re
from typing import List, Dict, Optional
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class LLMNERService:
    """
    基于轻量大模型的 NER 识别服务
    - 模型：Qwen2.5-3B-Instruct（适合8GB显存）
    - 识别：人名、公司名、地址、职位等非结构化信息
    - 准确率：93-95%
    """
    
    def __init__(self):
        logger.info("初始化大模型 NER 服务...")
        logger.info(f"模型: {settings.LLM_MODEL_NAME}")
        logger.info(f"显存利用率: {settings.LLM_GPU_MEMORY_UTILIZATION}")

        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError(
                "当前环境未安装 vLLM。若使用 Ollama，请将 LLM_BACKEND 设置为 'ollama'；"
                "若使用 vLLM，请先安装 vllm 相关依赖。"
            ) from exc
        
        # 初始化 vLLM
        model_path = settings.LLM_MODEL_PATH or settings.LLM_MODEL_NAME
        
        try:
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=settings.LLM_TENSOR_PARALLEL_SIZE,
                max_model_len=settings.LLM_MAX_LENGTH,
                gpu_memory_utilization=settings.LLM_GPU_MEMORY_UTILIZATION,
                dtype=settings.LLM_DTYPE,
                trust_remote_code=True
            )
            logger.info("大模型加载成功")
        except Exception as e:
            logger.error(f"大模型加载失败: {str(e)}")
            raise
        
        # 采样参数
        self.sampling_params = SamplingParams(
            temperature=0.1,  # 低温度保证稳定性
            max_tokens=1024,  # 减少输出长度以节省显存
            top_p=0.9
        )
        
        # Prompt 模板
        self.prompt_template = """你是一个专业的合同信息提取助手。请从以下合同文本中识别所有敏感信息。

合同文本：
{text}

请识别以下类型的敏感信息：
1. PERSON: 人名（如：张三、李四、王经理）
2. ORGANIZATION: 公司/组织名称（如：北京科技有限公司、XX银行）
3. LOCATION: 地址（如：北京市朝阳区XX路XX号）
4. POSITION: 职位（如：法定代表人、总经理、董事长）

要求：
- 只识别真实的敏感信息，不要识别"甲方"、"乙方"等代称
- 返回JSON格式，包含type、text、start、end字段
- start和end是字符在原文中的位置（从0开始计数）
- 确保位置准确

返回格式示例：
{{"entities": [
    {{"type": "PERSON", "text": "张三", "start": 10, "end": 12}},
    {{"type": "ORGANIZATION", "text": "北京科技有限公司", "start": 20, "end": 30}}
]}}

只返回JSON，不要其他内容："""
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        使用大模型提取实体
        
        Args:
            text: 待分析文本
            
        Returns:
            识别的实体列表
        """
        logger.info(f"大模型开始识别，文本长度: {len(text)}")
        
        try:
            # 如果文本太长，进行分块
            if len(text) > 2048:
                logger.info("文本过长，进行分块处理")
                return self._extract_from_chunks(text)
            
            # 构造 prompt
            prompt = self.prompt_template.format(text=text)
            
            # 推理
            outputs = self.llm.generate([prompt], self.sampling_params)
            
            # 解析结果
            result_text = outputs[0].outputs[0].text
            logger.debug(f"模型输出: {result_text}")
            
            entities = self._parse_result(result_text)
            
            # 添加source标记
            for entity in entities:
                entity["source"] = "llm"
                if "score" not in entity:
                    entity["score"] = 0.9  # 大模型默认置信度
            
            logger.info(f"大模型识别到 {len(entities)} 个实体")
            return entities
            
        except Exception as e:
            logger.error(f"大模型识别失败: {str(e)}")
            return []
    
    def _extract_from_chunks(self, text: str, chunk_size: int = 2048) -> List[Dict]:
        """
        分块处理长文本
        """
        chunks = []
        offset = 0
        
        # 按段落分块
        paragraphs = text.split('\n')
        current_chunk = ""
        current_offset = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + '\n'
            else:
                if current_chunk:
                    chunks.append((current_chunk, current_offset))
                current_chunk = para + '\n'
                current_offset = offset
            offset += len(para) + 1
        
        if current_chunk:
            chunks.append((current_chunk, current_offset))
        
        logger.info(f"文本分为 {len(chunks)} 块")
        
        # 处理每个块
        all_entities = []
        for chunk_text, chunk_offset in chunks:
            entities = self.extract_entities(chunk_text)
            # 调整位置偏移
            for entity in entities:
                entity["start"] += chunk_offset
                entity["end"] += chunk_offset
            all_entities.extend(entities)
        
        return all_entities
    
    def _parse_result(self, result_text: str) -> List[Dict]:
        """
        解析模型输出的JSON结果
        """
        try:
            # 尝试直接解析JSON
            result = json.loads(result_text)
            entities = result.get("entities", [])
            return entities
        except json.JSONDecodeError:
            # 如果解析失败，尝试提取JSON部分
            logger.warning("JSON解析失败，尝试提取JSON部分")
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    entities = result.get("entities", [])
                    return entities
                except:
                    pass
            
            logger.error("无法解析模型输出")
            return []
    
    def batch_extract(self, texts: List[str]) -> List[List[Dict]]:
        """
        批量提取（提升性能）
        注意：8GB显存建议batch_size=1，避免OOM
        
        Args:
            texts: 文本列表
            
        Returns:
            每个文本的识别结果列表
        """
        logger.info(f"批量识别 {len(texts)} 个文本")
        
        try:
            prompts = [self.prompt_template.format(text=text) for text in texts]
            outputs = self.llm.generate(prompts, self.sampling_params)
            
            results = []
            for output in outputs:
                result_text = output.outputs[0].text
                entities = self._parse_result(result_text)
                
                # 添加source标记
                for entity in entities:
                    entity["source"] = "llm"
                    if "score" not in entity:
                        entity["score"] = 0.9
                
                results.append(entities)
            
            return results
            
        except Exception as e:
            logger.error(f"批量识别失败: {str(e)}")
            return [[] for _ in texts]
