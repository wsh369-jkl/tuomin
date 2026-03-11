"""
文档解析器工厂
"""
from typing import Any, Dict, Optional
from app.processors.base_parser import BaseParser
from app.processors.pdf_parser import PDFParser
from app.processors.docx_parser import DOCXParser
from app.processors.txt_parser import TXTParser
import os
import logging

logger = logging.getLogger(__name__)

class DocumentParser:
    """
    文档解析器工厂
    根据文件类型自动选择合适的解析器
    """
    
    def __init__(self):
        # 注册所有解析器
        self.parsers = [
            PDFParser(),
            DOCXParser(),
            TXTParser()
        ]
    
    async def parse(self, file_path: str, **kwargs: Any) -> Dict:
        """
        解析文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            {
                "text": "文档文本内容",
                "metadata": {...}
            }
        """
        # 获取文件扩展名
        _, ext = os.path.splitext(file_path)
        
        # 查找支持的解析器
        parser = self._get_parser(ext)
        
        if parser is None:
            raise ValueError(f"不支持的文件格式: {ext}")
        
        # 解析文档
        result = await parser.parse(file_path, **kwargs)
        
        return result
    
    def _get_parser(self, file_extension: str) -> Optional[BaseParser]:
        """
        根据文件扩展名获取解析器
        """
        for parser in self.parsers:
            if parser.supports(file_extension):
                return parser
        return None
    
    def supports(self, file_extension: str) -> bool:
        """
        检查是否支持该文件格式
        """
        return self._get_parser(file_extension) is not None
