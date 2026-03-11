"""Main desensitization engine."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from app.core.config import settings
from app.core.operator_registry import OperatorRegistry
from app.core.pipeline_manager import PipelineManager
from app.core.recognizer_registry import RecognizerRegistry
from app.recognizers.contract_recognizer import ContractFieldRecognizer
from app.recognizers.custom_recognizer import CustomRecognizer
from app.recognizers.llm_recognizer import LLMRecognizer
from app.recognizers.pattern_recognizer import ChinesePatternRecognizer

logger = logging.getLogger(__name__)


class DesensitizationEngine:
    """Singleton engine that coordinates recognition and anonymization."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        logger.info("Initializing desensitization engine.")
        self.recognizer_registry = RecognizerRegistry()
        self.operator_registry = OperatorRegistry()
        self.pipeline_manager = PipelineManager(
            self.recognizer_registry,
            self.operator_registry,
        )

        self._register_default_recognizers()
        self._initialized = True
        logger.info("Desensitization engine initialized.")

    def _register_default_recognizers(self) -> None:
        default_recognizers = [
            LLMRecognizer(),
            ContractFieldRecognizer(),
            ChinesePatternRecognizer(),
            CustomRecognizer(),
        ]
        for recognizer in default_recognizers:
            self.recognizer_registry.add_recognizer(recognizer)

        logger.info(
            "Registered default recognizers: %s",
            [recognizer.name for recognizer in default_recognizers],
        )

    async def analyze(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        use_llm: bool = True,
        use_custom: bool = True,
        llm_model: Optional[str] = None,
    ) -> List[Dict]:
        logger.info(
            "Analyze requested, text_length=%s, use_llm=%s, use_custom=%s",
            len(text),
            use_llm,
            use_custom,
        )

        recognizer_names: List[str] = ["contract", "regex"]
        if use_llm:
            recognizer_names.insert(0, "llm")
        if use_custom:
            recognizer_names.append("custom")

        return await self.pipeline_manager.analyze(
            text=text,
            entities=entities,
            recognizer_names=recognizer_names,
            llm_model=llm_model,
        )

    async def anonymize(
        self,
        text: str,
        entities: List[Dict],
        operator_config: Optional[Dict[str, Dict]] = None,
    ) -> str:
        logger.info("Anonymize requested, entities=%s", len(entities))
        return await self.pipeline_manager.anonymize(
            text=text,
            entities=entities,
            operator_config=operator_config,
        )

    async def prepare_entities_for_anonymization(
        self,
        *,
        text: str,
        entities: List[Dict],
        use_llm: bool,
        operator_config: Optional[Dict[str, Dict]] = None,
        llm_model: Optional[str] = None,
        anonymization_strategy: Optional[str] = None,
    ) -> List[Dict]:
        return await self.pipeline_manager.prepare_entities_for_anonymization(
            text=text,
            entities=entities,
            use_llm=use_llm,
            operator_config=operator_config,
            llm_model=llm_model,
            anonymization_strategy=anonymization_strategy,
        )

    async def analyze_and_anonymize(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        use_llm: bool = True,
        use_custom: bool = True,
        operator_config: Optional[Dict[str, Dict]] = None,
        llm_model: Optional[str] = None,
        anonymization_strategy: Optional[str] = None,
    ) -> Dict:
        detected_entities = await self.analyze(
            text=text,
            entities=entities,
            use_llm=use_llm,
            use_custom=use_custom,
            llm_model=llm_model,
        )
        detected_entities = await self.prepare_entities_for_anonymization(
            text=text,
            entities=detected_entities,
            use_llm=use_llm,
            operator_config=operator_config,
            llm_model=llm_model,
            anonymization_strategy=anonymization_strategy,
        )
        anonymized_text = await self.anonymize(
            text=text,
            entities=detected_entities,
            operator_config=operator_config,
        )

        return {
            "original_text": text,
            "anonymized_text": anonymized_text,
            "entities": detected_entities,
            "statistics": self._get_statistics(detected_entities),
        }

    def _get_statistics(self, entities: List[Dict]) -> Dict:
        stats: Dict[str, Dict[str, object]] = {}
        for entity in entities:
            entity_type = entity["type"]
            bucket = stats.setdefault(entity_type, {"count": 0, "examples": []})
            bucket["count"] += 1
            if len(bucket["examples"]) < 3:
                bucket["examples"].append(entity["text"])
        return stats

    def get_supported_entities(self) -> List[str]:
        return self.recognizer_registry.get_supported_entities()

    def get_supported_operators(self) -> List[str]:
        return self.operator_registry.get_operator_names()

    def get_engine_info(self) -> Dict:
        return {
            "version": "2.1.0",
            "architecture": "registry_pipeline",
            "llm_backend": settings.LLM_BACKEND,
            "llm_model": (
                settings.OLLAMA_MODEL
                if settings.LLM_BACKEND.lower() == "ollama"
                else settings.LLM_MODEL_NAME
            ),
            "supported_entities": self.get_supported_entities(),
            "supported_operators": self.get_supported_operators(),
            "statistics": self.pipeline_manager.get_statistics(),
        }

    def get_entity_statistics(self, entities: List[Dict]) -> Dict:
        return self._get_statistics(entities)

    def get_last_quality_metadata(self) -> Dict:
        return self.pipeline_manager.get_last_quality_metadata()

    def add_recognizer(self, recognizer) -> None:
        self.recognizer_registry.add_recognizer(recognizer)

    def add_operator(self, operator) -> None:
        self.operator_registry.add_operator(operator)


_engine: Optional[DesensitizationEngine] = None


def get_engine() -> DesensitizationEngine:
    global _engine
    if _engine is None:
        _engine = DesensitizationEngine()
    return _engine
