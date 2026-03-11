"""
自定义规则识别器
将配置文件中的关键词和正则规则接入到新引擎的识别器体系中。
"""
from typing import List, Optional

from app.core.recognizer_base import BaseRecognizer, RecognizerResult
from app.services.custom_recognizer_service import CustomRecognizerService


class CustomRecognizer(BaseRecognizer):
    """基于 CustomRecognizerService 的新引擎识别器"""

    def __init__(self, service: Optional[CustomRecognizerService] = None):
        self.service = service or CustomRecognizerService()
        super().__init__(
            name="custom",
            supported_entities=self.service.get_supported_entities(),
            supported_language="zh",
            version="1.0.0",
        )

    async def analyze(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        **kwargs,
    ) -> List[RecognizerResult]:
        if not self.enabled:
            return []

        self.service.load_config()
        self.supported_entities = self.service.get_supported_entities()

        results = []
        for entity in self.service.match_all(text):
            if entities and entity["type"] not in entities:
                continue

            results.append(
                RecognizerResult(
                    entity_type=entity["type"],
                    start=entity["start"],
                    end=entity["end"],
                    score=entity.get("score", 0.95),
                    text=entity["text"],
                    source=self.name,
                    metadata=entity.get("metadata", {}),
                )
            )

        return results
