"""
Presidio 兼容服务
项目已切换到纯 Python 新引擎，这里保留同名服务以兼容旧测试和脚本。
"""
from typing import Dict, List, Optional
import logging

from app.engine.desensitization_engine import get_engine

logger = logging.getLogger(__name__)


class PresidioService:
    """
    兼容层。
    对外仍保留 PresidioService 名称，但内部已经改为无 Presidio 依赖的 RegexService。
    """

    def __init__(self):
        logger.info("初始化 Presidio 兼容服务（内部使用新引擎的正则链路）...")
        self.engine = get_engine()

    async def analyze(self, text: str, language: str = "zh") -> List[Dict]:
        return await self.engine.analyze(text, use_llm=False, use_custom=False)

    async def anonymize(
        self,
        text: str,
        entities: List[Dict],
        operators: Optional[Dict] = None,
    ) -> str:
        return await self.engine.anonymize(text, entities, operators)
