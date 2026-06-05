"""Minimal Hugging Face token-classification inference for packaged lowmem mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class _OpenEntity:
    label: str
    start: int
    end: int
    scores: list[float]


class HFTokenClassificationPipeline:
    """Small replacement for transformers.pipeline("token-classification").

    The generic Hugging Face pipeline imports all pipeline families, including
    optional image, audio, trainer, and dataset paths. The packaged app only
    needs local BERT-style token classification, so this class keeps imports and
    PyInstaller collection focused on AutoTokenizer, AutoModelForTokenClassification,
    torch, tokenizers, and safetensors.
    """

    def __init__(self, model_path: str) -> None:
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(f"hf_token_classifier_runtime_unavailable:{type(exc).__name__}: {exc}") from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            use_fast=True,
        )
        self._model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            local_files_only=True,
        )
        self._model.eval()
        self._id2label = {
            int(key): str(value)
            for key, value in getattr(self._model.config, "id2label", {}).items()
        }

    def __call__(self, text: str) -> List[Dict[str, Any]]:
        if not text:
            return []

        encoded = self._tokenizer(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()

        with self._torch.no_grad():
            logits = self._model(**encoded).logits[0]
            probabilities = self._torch.softmax(logits, dim=-1)
            scores, label_ids = probabilities.max(dim=-1)

        return self._aggregate_predictions(
            text,
            offsets=offsets,
            label_ids=[int(item) for item in label_ids.tolist()],
            scores=[float(item) for item in scores.tolist()],
        )

    def _aggregate_predictions(
        self,
        text: str,
        *,
        offsets: list[list[int]],
        label_ids: list[int],
        scores: list[float],
    ) -> List[Dict[str, Any]]:
        entities: list[Dict[str, Any]] = []
        current: Optional[_OpenEntity] = None

        def flush() -> None:
            nonlocal current
            if current is None:
                return
            if current.start < current.end:
                entities.append(
                    {
                        "entity_group": current.label,
                        "word": text[current.start : current.end],
                        "start": current.start,
                        "end": current.end,
                        "score": sum(current.scores) / max(len(current.scores), 1),
                    }
                )
            current = None

        for offset, label_id, score in zip(offsets, label_ids, scores):
            if not isinstance(offset, list) or len(offset) != 2:
                continue
            start, end = int(offset[0]), int(offset[1])
            if start == end:
                continue

            raw_label = self._id2label.get(label_id, "O")
            prefix, label = self._split_bio_label(raw_label)
            if not label:
                flush()
                continue

            if prefix in {"B", "S"} or current is None or current.label != label:
                flush()
                current = _OpenEntity(label=label, start=start, end=end, scores=[score])
            else:
                current.end = max(current.end, end)
                current.scores.append(score)

            if prefix == "S":
                flush()

        flush()
        return entities

    @staticmethod
    def _split_bio_label(raw_label: str) -> tuple[str, str]:
        label = str(raw_label or "").strip()
        if not label or label.upper() in {"O", "PAD", "[PAD]"}:
            return "O", ""
        if "-" not in label:
            return "B", label.upper()

        prefix, value = label.split("-", maxsplit=1)
        normalized_prefix = prefix.upper()
        if normalized_prefix not in {"B", "I", "S", "E"}:
            normalized_prefix = "B"
        return normalized_prefix, value.upper()
