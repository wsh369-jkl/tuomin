"""
脱敏操作注册表
管理所有脱敏操作的注册、查询
"""
from typing import Dict, Optional
from app.core.operator_base import BaseOperator, MaskOperator, ReplaceOperator, RedactOperator, HashOperator, EncryptOperator
import logging

logger = logging.getLogger(__name__)


class OperatorRegistry:
    """
    脱敏操作注册表
    - 注册/注销脱敏操作
    - 按名称查询操作
    - 提供默认操作
    """
    
    def __init__(self):
        self._operators: Dict[str, BaseOperator] = {}
        logger.info("初始化脱敏操作注册表")
        
        # 注册默认操作
        self._register_default_operators()
    
    def _register_default_operators(self):
        """注册默认的脱敏操作"""
        default_operators = [
            MaskOperator(),
            ReplaceOperator(),
            RedactOperator(),
            HashOperator(),
            EncryptOperator()
        ]
        
        for operator in default_operators:
            self.add_operator(operator)
    
    def add_operator(self, operator: BaseOperator):
        """
        注册脱敏操作
        
        Args:
            operator: 脱敏操作实例
        """
        if operator.name in self._operators:
            logger.warning(f"脱敏操作 {operator.name} 已存在，将被覆盖")
        
        self._operators[operator.name] = operator
        logger.info(f"注册脱敏操作: {operator.name}")
    
    def remove_operator(self, name: str):
        """
        注销脱敏操作
        
        Args:
            name: 操作名称
        """
        if name in self._operators:
            del self._operators[name]
            logger.info(f"注销脱敏操作: {name}")
        else:
            logger.warning(f"脱敏操作 {name} 不存在")
    
    def get_operator(self, name: str) -> Optional[BaseOperator]:
        """
        获取脱敏操作
        
        Args:
            name: 操作名称
            
        Returns:
            脱敏操作实例，不存在返回 None
        """
        return self._operators.get(name)
    
    def get_all_operators(self) -> Dict[str, BaseOperator]:
        """获取所有脱敏操作"""
        return self._operators.copy()
    
    def get_operator_names(self) -> list:
        """获取所有操作名称"""
        return list(self._operators.keys())
    
    def get_statistics(self) -> Dict:
        """获取注册表统计信息"""
        return {
            "total_operators": len(self._operators),
            "operators": [
                {
                    "name": op.name,
                    "type": op.__class__.__name__
                }
                for op in self._operators.values()
            ]
        }
