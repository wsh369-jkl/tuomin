"""Risk snippet scheduling for fragment-level review."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

from app.core.config import settings
from app.core.recognizer_base import RecognizerResult


@dataclass(frozen=True)
class RiskSnippet:
    snippet_type: str
    risk_reason: str
    start: int
    end: int
    text: str


class RiskSnippetScheduler:
    """Build high-risk fragments so the review LLM never sees the full document."""

    KEYWORDS = {
        "legal_party_block": ["上诉人", "被上诉人", "原审原告", "原审被告", "一审原告", "一审被告", "申请人", "被申请人", "第三人"],
        "address_block": ["身份证住址", "经常居住地", "户籍地址", "户籍地", "现住址", "住址", "住所", "地址", "送达地址", "注册地址", "通讯地址"],
        "account_block": ["开户行", "开户银行", "户名", "账户", "账号", "收款单位", "付款单位"],
        "definition_block": ["以下简称", "下称", "简称", "又称"],
        "signature_block": ["签字", "签章", "盖章", "落款"],
        "narrative_hotspot": [
            "股东",
            "付款",
            "转账",
            "事实与理由",
            "合同纠纷",
            "往来款",
            "收款",
            "个人银行账户",
            "个人账户",
            "商行",
            "工作室",
            "合作社",
            "经营部",
            "门市部",
            "营业部",
            "办事处",
            "基金会",
            "联合会",
            "事务所",
            "研究院",
            "研究所",
        ],
        "ocr_anomaly_block": ["□", "�", "　　"],
    }

    def build_snippets(
        self,
        text: str,
        entities: Iterable[RecognizerResult],
        *,
        max_snippets: int | None = None,
        max_chars_per_snippet: int | None = None,
    ) -> List[RiskSnippet]:
        max_count = max_snippets or settings.REVIEW_MAX_SNIPPETS
        max_chars = max_chars_per_snippet or settings.REVIEW_MAX_CHARS_PER_SNIPPET
        snippets: list[RiskSnippet] = []

        if not text:
            return snippets

        snippets.append(self._make_snippet(text, 0, min(len(text), max_chars), "header_party_block", "document_header"))

        for snippet_type, keywords in self.KEYWORDS.items():
            added_for_type = 0
            for keyword in keywords:
                for index in self._iter_keyword_indexes(text, keyword):
                    snippets.append(
                        self._make_window_snippet(
                            text,
                            index,
                            snippet_type=snippet_type,
                            risk_reason=f"keyword:{keyword}",
                            max_chars=max_chars,
                        )
                    )
                    added_for_type += 1
                    if added_for_type >= self._max_snippets_for_type(snippet_type):
                        break
                if added_for_type >= self._max_snippets_for_type(snippet_type):
                    break

        for start, reason in self._iter_complex_entity_hotspots(text):
            snippets.append(
                self._make_window_snippet(
                    text,
                    start,
                    snippet_type="narrative_hotspot",
                    risk_reason=reason,
                    max_chars=max_chars,
                )
            )

        if self._entity_density_low(text, list(entities)):
            snippets.append(
                self._make_snippet(
                    text,
                    0,
                    min(len(text), max_chars),
                    "ocr_anomaly_block",
                    "long_document_low_entity_density",
                )
            )

        conflict_span = self._find_model_conflict_span(list(entities))
        if conflict_span is not None:
            snippets.append(
                self._make_window_snippet(
                    text,
                    conflict_span,
                    snippet_type="conflict_block",
                    risk_reason="uie_ner_overlap_conflict",
                    max_chars=max_chars,
                )
            )

        return self._dedupe(snippets, max_count=max_count)

    def _make_window_snippet(
        self,
        text: str,
        center: int,
        *,
        snippet_type: str,
        risk_reason: str,
        max_chars: int,
    ) -> RiskSnippet:
        half = max(80, max_chars // 2)
        start = max(0, center - half)
        end = min(len(text), start + max_chars)
        start = max(0, end - max_chars)
        return self._make_snippet(text, start, end, snippet_type, risk_reason)

    @staticmethod
    def _make_snippet(text: str, start: int, end: int, snippet_type: str, risk_reason: str) -> RiskSnippet:
        return RiskSnippet(
            snippet_type=snippet_type,
            risk_reason=risk_reason,
            start=start,
            end=end,
            text=text[start:end],
        )

    @staticmethod
    def _entity_density_low(text: str, entities: list[RecognizerResult]) -> bool:
        chinese_chars = len(re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]", text))
        if chinese_chars < 1500:
            return False
        sensitive_count = len([item for item in entities if item.entity_type not in {"DATE", "AMOUNT"}])
        return sensitive_count <= 2

    @staticmethod
    def _iter_keyword_indexes(text: str, keyword: str):
        start = 0
        while True:
            index = text.find(keyword, start)
            if index < 0:
                break
            yield index
            start = index + max(1, len(keyword))

    @staticmethod
    def _max_snippets_for_type(snippet_type: str) -> int:
        if snippet_type in {"legal_party_block", "address_block"}:
            return 3
        if snippet_type == "narrative_hotspot":
            return 5
        return 2

    @staticmethod
    def _iter_complex_entity_hotspots(text: str):
        patterns = [
            (r"[\u4e00-\u9fa5·]{2,8}(?:个人银行账户|个人账户|本人账户|银行账户)", "person_account_cue"),
            (r"(?:股东|实际控制人|法定代表人|委托诉讼代理人|诉讼代理人|委托代理人|代理人)\s*[：:为系是]?\s*[\u4e00-\u9fa5·、和及与]{2,30}", "role_person_cue"),
            (r"(?:上诉人|被上诉人|原审原告|原审被告|申请人|被申请人|原告|被告|第三人)[^。\n\r]{2,80}", "legal_party_cue"),
            (r"(?:现住|户籍地|户籍地址|身份证住址|经常居住地)[^。\n\r]{4,100}", "residence_address_cue"),
            (
                r"(?:由|与|同|和|向|对)?[\u4e00-\u9fa5A-Za-z0-9·]{2,16}"
                r"(?:继续履约|负责(?:履约|结算|交付|供货|施工|收款|付款|签约|执行)|"
                r"签订(?:补充)?协议|签订(?:补充)?合同|提供(?:技术)?服务|承担(?:付款|结算|供货|施工|交付)?责任|"
                r"继续结算|继续供货|继续施工|办理结算|履行(?:付款|交付|供货|施工)?义务)",
                "organization_action_cue",
            ),
        ]
        for pattern, reason in patterns:
            for match in re.finditer(pattern, text):
                yield match.start(), reason

    @staticmethod
    def _find_model_conflict_span(entities: list[RecognizerResult]) -> int | None:
        uie_entities = [item for item in entities if item.source == "uie"]
        ner_entities = [item for item in entities if item.source in {"ner", "secondary_ner"}]
        for left in uie_entities:
            for right in ner_entities:
                if left.start < right.end and right.start < left.end and left.entity_type != right.entity_type:
                    return min(left.start, right.start)
        return None

    @staticmethod
    def _dedupe(snippets: list[RiskSnippet], *, max_count: int) -> list[RiskSnippet]:
        seen: set[tuple[int, int, str]] = set()
        deduped: list[RiskSnippet] = []
        for snippet in snippets:
            key = (snippet.start, snippet.end, snippet.snippet_type)
            if key in seen or not snippet.text.strip():
                continue
            seen.add(key)
            deduped.append(snippet)
            if len(deduped) >= max_count:
                break
        return deduped
