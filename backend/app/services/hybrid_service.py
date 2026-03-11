"""
混合识别引擎兼容层
旧代码仍引用 HybridNERService，这里统一转发到新引擎。
"""
from typing import Dict, List, Optional
import logging

from app.engine.desensitization_engine import get_engine

logger = logging.getLogger(__name__)


class HybridNERService:
    """
    兼容旧接口：
    - extract_entities -> 新引擎 analyze
    - anonymize -> 新引擎 anonymize
    """

    def __init__(self):
        logger.info("初始化混合识别兼容层...")
        self.engine = get_engine()

    async def extract_entities(
        self,
        text: str,
        use_llm: bool = True,
        use_custom: bool = True,
    ) -> List[Dict]:
        return await self.engine.analyze(text, use_llm=use_llm, use_custom=use_custom)

    async def anonymize(
        self,
        text: str,
        entities: List[Dict],
        config: Optional[Dict] = None,
    ) -> str:
        return await self.engine.anonymize(text, entities, config)

    def get_entity_statistics(self, entities: List[Dict]) -> Dict:
        return self.engine.get_entity_statistics(entities)
