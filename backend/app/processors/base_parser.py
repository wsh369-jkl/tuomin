"""
文档解析器基类
"""
from abc import ABC, abstractmethod
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

class BaseParser(ABC):
    """
    文档解析器基类
    """
    
    @abstractmethod
    async def parse(self, file_path: str, **kwargs: Any) -> Dict:
        """
        解析文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            {
                "text": "文档文本内容",
                "metadata": {
                    "pages": 页数,
                    "format": 格式,
                    ...
                }
            }
        """
        pass
    
    @abstractmethod
    def supports(self, file_extension: str) -> bool:
        """
        检查是否支持该文件格式
        """
        pass
