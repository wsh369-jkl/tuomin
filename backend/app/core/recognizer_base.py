"""
核心基类 - 识别器基类
所有识别器必须继承此类
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RecognizerResult:
    """识别结果"""
    entity_type: str  # 实体类型
    start: int  # 起始位置
    end: int  # 结束位置
    score: float  # 置信度 (0-1)
    text: str  # 识别的文本
    source: str  # 识别来源
    metadata: Optional[Dict] = None  # 额外元数据
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "score": self.score,
            "text": self.text,
            "source": self.source,
            "metadata": self.metadata or {}
        }


class BaseRecognizer(ABC):
    """
    识别器基类
    所有识别器（正则、ML、LLM、自定义）都继承此类
    """
    
    def __init__(
        self,
        name: str,
        supported_entities: List[str],
        supported_language: str = "zh",
        version: str = "1.0.0"
    ):
        self.name = name
        self.supported_entities = supported_entities
        self.supported_language = supported_language
        self.version = version
        self.enabled = True
        
        logger.debug(f"初始化识别器: {self.name}, 支持实体: {self.supported_entities}")
    
    @abstractmethod
    async def analyze(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        **kwargs,
    ) -> List[RecognizerResult]:
        """
        分析文本，识别敏感信息
        
        Args:
            text: 待分析文本
            entities: 要识别的实体类型列表（None表示识别所有支持的类型）
            
        Returns:
            识别结果列表
        """
        pass
    
    def supports_entity(self, entity_type: str) -> bool:
        """检查是否支持某个实体类型"""
        return entity_type in self.supported_entities
    
    def enable(self):
        """启用识别器"""
        self.enabled = True
        logger.info(f"识别器 {self.name} 已启用")
    
    def disable(self):
        """禁用识别器"""
        self.enabled = False
        logger.info(f"识别器 {self.name} 已禁用")
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, entities={self.supported_entities})"
