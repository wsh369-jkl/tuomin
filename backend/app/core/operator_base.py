"""
核心基类 - 脱敏操作基类
所有脱敏操作必须继承此类
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OperatorConfig:
    """脱敏操作配置"""
    operator_name: str  # 操作名称
    params: Dict[str, Any]  # 操作参数
    
    def to_dict(self) -> Dict:
        return {
            "operator_name": self.operator_name,
            "params": self.params
        }


class BaseOperator(ABC):
    """
    脱敏操作基类
    所有脱敏操作（遮蔽、替换、加密等）都继承此类
    """
    
    def __init__(self, name: str):
        self.name = name
        logger.debug(f"初始化脱敏操作: {self.name}")
    
    @abstractmethod
    def operate(self, text: str, start: int, end: int, entity_type: str, **params) -> str:
        """
        执行脱敏操作
        
        Args:
            text: 原始文本
            start: 实体起始位置
            end: 实体结束位置
            entity_type: 实体类型
            **params: 额外参数
            
        Returns:
            脱敏后的文本片段
        """
        pass
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """验证参数是否有效"""
        return True
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"


class MaskOperator(BaseOperator):
    """遮蔽操作 - 用 * 遮蔽部分字符"""
    
    def __init__(self):
        super().__init__("mask")
    
    def operate(self, text: str, start: int, end: int, entity_type: str, **params) -> str:
        """
        遮蔽操作
        
        参数:
            masking_char: 遮蔽字符（默认 *）
            chars_to_mask: 要遮蔽的字符数
            from_end: 是否从末尾开始遮蔽
            keep_start: 保留前几个字符
            keep_end: 保留后几个字符
            mask_email: 是否按邮箱结构脱敏
        """
        original = text[start:end]
        masking_char = params.get("masking_char", "*")
        keep_start = params.get("keep_start")
        keep_end = params.get("keep_end")
        mask_email = params.get("mask_email", False)
        chars_to_mask = params.get("chars_to_mask", len(original) - 4)
        from_end = params.get("from_end", False)

        if mask_email and "@" in original:
            username, domain = original.split("@", 1)
            if not username:
                return masking_char + "@" + domain
            return username[:1] + masking_char * max(1, len(username) - 1) + "@" + domain

        if keep_start is not None or keep_end is not None:
            keep_start = max(0, int(keep_start or 0))
            keep_end = max(0, int(keep_end or 0))
            if keep_start + keep_end >= len(original):
                return original
            masked_length = len(original) - keep_start - keep_end
            return (
                original[:keep_start]
                + masking_char * masked_length
                + original[len(original) - keep_end:]
            )
        
        if len(original) <= 4:
            return masking_char * len(original)
        
        if from_end:
            # 从末尾遮蔽
            keep_start = len(original) - chars_to_mask
            return original[:keep_start] + masking_char * chars_to_mask
        else:
            # 从开头遮蔽
            keep_end = len(original) - chars_to_mask
            return masking_char * chars_to_mask + original[keep_end:]


class ReplaceOperator(BaseOperator):
    """替换操作 - 用占位符替换"""
    
    def __init__(self):
        super().__init__("replace")
    
    def operate(self, text: str, start: int, end: int, entity_type: str, **params) -> str:
        """
        替换操作
        
        参数:
            new_value: 新值（默认 [实体类型]）
        """
        new_value = params.get("new_value", f"[{entity_type}]")
        return new_value


class RedactOperator(BaseOperator):
    """删除操作 - 完全删除"""
    
    def __init__(self):
        super().__init__("redact")
    
    def operate(self, text: str, start: int, end: int, entity_type: str, **params) -> str:
        """删除操作"""
        return ""


class HashOperator(BaseOperator):
    """哈希操作 - 用哈希值替换"""
    
    def __init__(self):
        super().__init__("hash")
    
    def operate(self, text: str, start: int, end: int, entity_type: str, **params) -> str:
        """
        哈希操作
        
        参数:
            hash_type: 哈希类型（md5, sha256）
        """
        import hashlib
        
        original = text[start:end]
        hash_type = params.get("hash_type", "md5")
        
        if hash_type == "md5":
            return hashlib.md5(original.encode()).hexdigest()[:8]
        elif hash_type == "sha256":
            return hashlib.sha256(original.encode()).hexdigest()[:16]
        else:
            return hashlib.md5(original.encode()).hexdigest()[:8]


class EncryptOperator(BaseOperator):
    """加密操作 - 可逆加密"""
    
    def __init__(self):
        super().__init__("encrypt")
    
    def operate(self, text: str, start: int, end: int, entity_type: str, **params) -> str:
        """
        加密操作（简单实现，生产环境应使用专业加密库）
        
        参数:
            key: 加密密钥
        """
        import base64
        
        original = text[start:end]
        key = params.get("key", "default_key")
        
        # 简单的 XOR 加密 + Base64
        encrypted = bytes([ord(c) ^ ord(key[i % len(key)]) for i, c in enumerate(original)])
        return base64.b64encode(encrypted).decode()[:16] + "..."
