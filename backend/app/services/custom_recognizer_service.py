"""Custom recognizer configuration service."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from app.core.config import settings


logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PAYLOAD = {
    "patterns": [
        {
            "name": "PROJECT_CODE",
            "description": "项目代号",
            "regex": "PROJ-\\d{4}-\\d{3}",
            "score": 0.9,
            "context": ["项目", "代号", "编号"],
        }
    ],
    "keywords": {
        "COMPANY_NAME": {
            "description": "特定公司名称",
            "keywords": ["XX科技有限公司", "XX贸易有限公司"],
            "score": 0.95,
        }
    },
}


class CustomRecognizerService:
    """Load, persist, and execute custom keyword/regex rules."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else self._get_default_config_path()
        self.custom_keywords: Dict[str, Dict] = {}
        self.custom_patterns: List[Dict] = []
        self._compiled_patterns: List[Dict] = []
        self.load_config()

    def _get_default_config_path(self) -> Path:
        return Path(settings.CUSTOM_CONFIG_PATH)

    def _get_seed_config_path(self) -> Path:
        return Path(settings.DEFAULT_CUSTOM_CONFIG_PATH)

    def _load_seed_config(self) -> Dict:
        seed_path = self._get_seed_config_path()
        if seed_path.exists():
            try:
                with open(seed_path, "r", encoding="utf-8") as file:
                    return json.load(file)
            except Exception as exc:
                logger.warning("Failed to load bundled custom config %s: %s", seed_path, exc)
        return DEFAULT_CONFIG_PAYLOAD

    def _create_default_config(self) -> None:
        default_config = self._load_seed_config()
        os.makedirs(self.config_path.parent, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as file:
            json.dump(default_config, file, ensure_ascii=False, indent=2)
        logger.info("Created custom config file: %s", self.config_path)

    def _normalize_keywords(self, keywords: List[str]) -> List[str]:
        normalized: List[str] = []
        seen = set()
        for keyword in keywords:
            value = keyword.strip()
            if not value or value in seen:
                continue
            normalized.append(value)
            seen.add(value)
        return normalized

    def _compile_patterns(self) -> None:
        self._compiled_patterns = []
        for pattern_config in self.custom_patterns:
            try:
                compiled = re.compile(pattern_config["regex"])
                self._compiled_patterns.append(
                    {
                        **pattern_config,
                        "compiled_regex": compiled,
                    }
                )
            except re.error as exc:
                logger.error(
                    "Failed to compile custom regex %s: %s",
                    pattern_config.get("name", "UNKNOWN"),
                    exc,
                )

    def load_config(self) -> None:
        if not self.config_path.exists():
            logger.warning("Custom config file does not exist: %s", self.config_path)
            self._create_default_config()

        try:
            with open(self.config_path, "r", encoding="utf-8") as file:
                config = json.load(file)

            self.custom_keywords = config.get("keywords", {})
            self.custom_patterns = config.get("patterns", [])

            for entity_type, keyword_config in list(self.custom_keywords.items()):
                keyword_config["keywords"] = self._normalize_keywords(
                    keyword_config.get("keywords", [])
                )
                if not keyword_config["keywords"]:
                    del self.custom_keywords[entity_type]

            self._compile_patterns()

            logger.info(
                "Loaded custom rules: %s regex groups, %s keyword groups",
                len(self.custom_patterns),
                len(self.custom_keywords),
            )
        except Exception as exc:
            logger.error("Failed to load custom config: %s", exc)
            self.custom_keywords = {}
            self.custom_patterns = []
            self._compiled_patterns = []

    def get_supported_entities(self) -> List[str]:
        keyword_entities = set(self.custom_keywords.keys())
        pattern_entities = {
            pattern["name"] for pattern in self.custom_patterns if pattern.get("name")
        }
        return sorted(keyword_entities | pattern_entities)

    def add_keyword_rule(
        self,
        entity_type: str,
        keywords: List[str],
        score: float = 0.95,
        description: str = "",
    ) -> None:
        normalized_keywords = self._normalize_keywords(keywords)
        if not normalized_keywords:
            raise ValueError("Keyword list cannot be empty.")

        self.custom_keywords[entity_type] = {
            "description": description
            or self.custom_keywords.get(entity_type, {}).get("description", ""),
            "keywords": normalized_keywords,
            "score": score,
        }
        logger.info("Saved keyword rule %s with %s keywords", entity_type, len(normalized_keywords))

    def add_pattern_rule(
        self,
        entity_type: str,
        regex: str,
        context: Optional[List[str]] = None,
        score: float = 0.9,
        description: str = "",
    ) -> None:
        pattern_config = {
            "name": entity_type,
            "description": description,
            "regex": regex,
            "score": score,
            "context": context or [],
        }

        updated = False
        for index, existing in enumerate(self.custom_patterns):
            if existing.get("name") == entity_type:
                self.custom_patterns[index] = pattern_config
                updated = True
                break

        if not updated:
            self.custom_patterns.append(pattern_config)

        self._compile_patterns()
        logger.info("Saved regex rule %s", entity_type)

    def delete_keyword_rule(self, entity_type: str) -> bool:
        if entity_type in self.custom_keywords:
            del self.custom_keywords[entity_type]
            return True
        return False

    def delete_pattern_rule(self, entity_type: str) -> bool:
        original_count = len(self.custom_patterns)
        self.custom_patterns = [
            pattern for pattern in self.custom_patterns if pattern.get("name") != entity_type
        ]
        self._compile_patterns()
        return len(self.custom_patterns) != original_count

    def match_keywords(self, text: str) -> List[Dict]:
        entities: List[Dict] = []

        for entity_type, config in self.custom_keywords.items():
            keywords = config.get("keywords", [])
            score = config.get("score", 0.95)

            for keyword in keywords:
                start = 0
                while True:
                    position = text.find(keyword, start)
                    if position == -1:
                        break

                    entities.append(
                        {
                            "type": entity_type,
                            "text": keyword,
                            "start": position,
                            "end": position + len(keyword),
                            "score": score,
                            "source": "custom",
                            "metadata": {"match_type": "keyword"},
                        }
                    )
                    start = position + len(keyword)

        return entities

    def match_patterns(self, text: str) -> List[Dict]:
        entities: List[Dict] = []
        for pattern_config in self._compiled_patterns:
            regex = pattern_config["compiled_regex"]
            for match in regex.finditer(text):
                entities.append(
                    {
                        "type": pattern_config["name"],
                        "text": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "score": pattern_config.get("score", 0.9),
                        "source": "custom",
                        "metadata": {
                            "match_type": "pattern",
                            "regex": pattern_config["regex"],
                            "context": pattern_config.get("context", []),
                        },
                    }
                )
        return entities

    def match_all(self, text: str) -> List[Dict]:
        entities = self.match_keywords(text) + self.match_patterns(text)
        entities.sort(key=lambda item: (item["start"], item["end"]))
        return entities

    def save_config(self) -> None:
        config = {
            "patterns": self.custom_patterns,
            "keywords": self.custom_keywords,
        }
        os.makedirs(self.config_path.parent, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as file:
            json.dump(config, file, ensure_ascii=False, indent=2)
        logger.info("Saved custom config: %s", self.config_path)
