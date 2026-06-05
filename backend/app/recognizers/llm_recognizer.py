"""LLM recognizer for semantic contract entities."""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional

from app.core.config import settings
from app.core.recognizer_base import BaseRecognizer, RecognizerResult

logger = logging.getLogger(__name__)


class LLMRecognizer(BaseRecognizer):
    """Use the configured LLM backend to extract semantic entities."""

    SUPPORTED_ENTITIES = [
        "PERSON",
        "ORGANIZATION",
        "LOCATION",
        "POSITION",
        "PROJECT",
        "CONTRACT_NO",
        "BANK_NAME",
        "ACCOUNT_NAME",
    ]

    def __init__(self) -> None:
        super().__init__(
            name="llm",
            supported_entities=self.SUPPORTED_ENTITIES,
            supported_language="zh",
            version="1.1.0",
        )
        self.llm_services: Dict[str, object] = {}
        self.last_run_metadata: Dict[str, Dict[str, object]] = {}

    def _get_llm_service(self, llm_model: Optional[str] = None):
        if settings.LLM_BACKEND.lower() == "ollama":
            from app.services.ollama_service import OllamaLLMService

            model_name = llm_model or settings.OLLAMA_MODEL
            if model_name not in self.llm_services:
                self.llm_services[model_name] = OllamaLLMService(
                    base_url=settings.OLLAMA_BASE_URL,
                    model=model_name,
                    timeout=settings.OLLAMA_TIMEOUT,
                    num_ctx=settings.OLLAMA_NUM_CTX,
                )
            return self.llm_services[model_name]

        if "vllm" not in self.llm_services:
            from app.services.llm_service import LLMNERService

            self.llm_services["vllm"] = LLMNERService()
        return self.llm_services["vllm"]

    def get_last_run_metadata(self, llm_model: Optional[str] = None) -> Dict[str, object]:
        model_key = llm_model or settings.OLLAMA_MODEL
        return dict(self.last_run_metadata.get(model_key, {}))

    async def analyze(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        llm_model: Optional[str] = None,
        **kwargs,
    ) -> List[RecognizerResult]:
        if not self.enabled:
            return []

        progress_callback = kwargs.get("progress_callback")
        try:
            llm_service = self._get_llm_service(llm_model=llm_model)
            if hasattr(llm_service, "set_progress_callback"):
                llm_service.set_progress_callback(progress_callback)
            if hasattr(llm_service, "extract_entities_async"):
                llm_entities = await llm_service.extract_entities_async(text)
            else:
                llm_entities = await asyncio.to_thread(llm_service.extract_entities, text)
            model_key = (
                llm_model or settings.OLLAMA_MODEL
                if settings.LLM_BACKEND.lower() == "ollama"
                else settings.LLM_MODEL_NAME
            )
            if hasattr(llm_service, "get_last_extract_metadata"):
                self.last_run_metadata[model_key] = dict(llm_service.get_last_extract_metadata())
            else:
                self.last_run_metadata[model_key] = {}
        except Exception as exc:
            logger.error("LLM recognition failed: %s", exc)
            return []
        finally:
            if "llm_service" in locals() and hasattr(llm_service, "set_progress_callback"):
                llm_service.set_progress_callback(None)

        results: List[RecognizerResult] = []
        for entity in llm_entities:
            entity_type = entity.get("type")
            if entity_type not in self.SUPPORTED_ENTITIES:
                continue
            if entities and entity_type not in entities:
                continue

            results.append(
                RecognizerResult(
                    entity_type=entity_type,
                    start=entity["start"],
                    end=entity["end"],
                    score=entity.get("score", 0.88),
                    text=entity["text"],
                    source=entity.get("source", self.name),
                    metadata={
                        **(entity.get("metadata") or {}),
                        "llm_backend": settings.LLM_BACKEND,
                        "llm_model": (
                            llm_model or settings.OLLAMA_MODEL
                            if settings.LLM_BACKEND.lower() == "ollama"
                            else settings.LLM_MODEL_NAME
                        ),
                    },
                )
            )

        return results
