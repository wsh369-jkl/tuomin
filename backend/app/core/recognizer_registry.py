"""
识别器注册表
管理所有识别器的注册、查询、启用/禁用
"""
from typing import List, Dict, Optional
from app.core.recognizer_base import BaseRecognizer, RecognizerResult
import logging

logger = logging.getLogger(__name__)


class RecognizerRegistry:
    """
    识别器注册表
    - 注册/注销识别器
    - 按实体类型查询识别器
    - 批量执行识别
    """
    
    def __init__(self):
        self._recognizers: Dict[str, BaseRecognizer] = {}
        logger.info("初始化识别器注册表")
    
    def add_recognizer(self, recognizer: BaseRecognizer):
        """
        注册识别器
        
        Args:
            recognizer: 识别器实例
        """
        if recognizer.name in self._recognizers:
            logger.warning(f"识别器 {recognizer.name} 已存在，将被覆盖")
        
        self._recognizers[recognizer.name] = recognizer
        logger.info(f"注册识别器: {recognizer.name}, 支持实体: {recognizer.supported_entities}")
    
    def remove_recognizer(self, name: str):
        """
        注销识别器
        
        Args:
            name: 识别器名称
        """
        if name in self._recognizers:
            del self._recognizers[name]
            logger.info(f"注销识别器: {name}")
        else:
            logger.warning(f"识别器 {name} 不存在")
    
    def get_recognizer(self, name: str) -> Optional[BaseRecognizer]:
        """
        获取识别器
        
        Args:
            name: 识别器名称
            
        Returns:
            识别器实例，不存在返回 None
        """
        return self._recognizers.get(name)
    
    def get_recognizers_for_entity(self, entity_type: str) -> List[BaseRecognizer]:
        """
        获取支持某个实体类型的所有识别器
        
        Args:
            entity_type: 实体类型
            
        Returns:
            识别器列表
        """
        recognizers = [
            r for r in self._recognizers.values()
            if r.enabled and r.supports_entity(entity_type)
        ]
        logger.debug(f"实体类型 {entity_type} 有 {len(recognizers)} 个识别器")
        return recognizers
    
    def get_all_recognizers(self) -> List[BaseRecognizer]:
        """获取所有识别器"""
        return list(self._recognizers.values())
    
    def get_enabled_recognizers(self) -> List[BaseRecognizer]:
        """获取所有启用的识别器"""
        return [r for r in self._recognizers.values() if r.enabled]
    
    def get_supported_entities(self) -> List[str]:
        """获取所有支持的实体类型"""
        entities = set()
        for recognizer in self._recognizers.values():
            if recognizer.enabled:
                entities.update(recognizer.supported_entities)
        return sorted(list(entities))
    
    async def analyze(
        self, 
        text: str, 
        entities: Optional[List[str]] = None,
        recognizer_names: Optional[List[str]] = None,
        **kwargs,
    ) -> List[RecognizerResult]:
        """
        使用注册的识别器分析文本
        
        Args:
            text: 待分析文本
            entities: 要识别的实体类型列表（None表示所有）
            recognizer_names: 要使用的识别器名称列表（None表示所有）
            
        Returns:
            识别结果列表
        """
        logger.info(f"开始分析，文本长度: {len(text)}")
        
        # 确定要使用的识别器
        if recognizer_names:
            recognizers = [
                self._recognizers[name] 
                for name in recognizer_names 
                if name in self._recognizers and self._recognizers[name].enabled
            ]
        else:
            recognizers = self.get_enabled_recognizers()
        
        logger.info(f"使用 {len(recognizers)} 个识别器")
        
        # 执行识别
        all_results = []
        for recognizer in recognizers:
            try:
                results = await recognizer.analyze(
                    text,
                    entities,
                    **{**kwargs, "existing_results": list(all_results)},
                )
                all_results.extend(results)
                logger.debug(f"识别器 {recognizer.name} 识别到 {len(results)} 个实体")
            except Exception as e:
                logger.error(f"识别器 {recognizer.name} 执行失败: {str(e)}")
        
        logger.info(f"总共识别到 {len(all_results)} 个实体")
        return all_results
    
    def get_statistics(self) -> Dict:
        """获取注册表统计信息"""
        return {
            "total_recognizers": len(self._recognizers),
            "enabled_recognizers": len(self.get_enabled_recognizers()),
            "supported_entities": self.get_supported_entities(),
            "recognizers": [
                {
                    "name": r.name,
                    "enabled": r.enabled,
                    "entities": r.supported_entities,
                    "version": r.version
                }
                for r in self._recognizers.values()
            ]
        }
