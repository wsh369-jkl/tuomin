"""Risk snippet scheduling for fragment-level review."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, List

from app.core.config import settings
from app.core.recognizer_base import RecognizerResult
from app.rules.default_subject_policy import DEFAULT_SUBJECT_TYPES, projected_default_subject_type


@dataclass(frozen=True)
class RiskSnippet:
    snippet_type: str
    risk_reason: str
    start: int
    end: int
    text: str
    target_entity: dict[str, Any] | None = None


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
        source_structure: dict[str, Any] | None = None,
        max_snippets: int | None = None,
        max_chars_per_snippet: int | None = None,
    ) -> List[RiskSnippet]:
        max_count = max_snippets or settings.REVIEW_MAX_SNIPPETS
        max_chars = max_chars_per_snippet or settings.REVIEW_MAX_CHARS_PER_SNIPPET
        snippets: list[RiskSnippet] = []

        if not text:
            return snippets

        entity_list = list(entities)
        snippets.append(self._make_snippet(text, 0, min(len(text), max_chars), "header_party_block", "document_header"))
        snippets.extend(
            self._build_rule_first_review_snippets(
                text=text,
                entities=entity_list,
                max_chars=max_chars,
            )
        )
        snippets.extend(
            self._build_structure_snippets(
                text=text,
                source_structure=source_structure,
                max_chars=max_chars,
            )
        )

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

        if self._entity_density_low(text, entity_list):
            snippets.append(
                self._make_snippet(
                    text,
                    0,
                    min(len(text), max_chars),
                    "ocr_anomaly_block",
                    "long_document_low_entity_density",
                )
            )

        conflict_span = self._find_model_conflict_span(entity_list)
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

    def _build_rule_first_review_snippets(
        self,
        *,
        text: str,
        entities: list[RecognizerResult],
        max_chars: int,
    ) -> List[RiskSnippet]:
        snippets: list[RiskSnippet] = []
        for entity in entities:
            metadata = dict(entity.metadata or {})
            rule_first = metadata.get("rule_first")
            ledger_status = str(metadata.get("subject_ledger_status") or "").strip()
            ledger_subject_status = str(metadata.get("subject_ledger_subject_status") or "").strip()
            ledger_review_required = ledger_status in {
                "ambiguous_short_subject",
                "unresolved_alias",
                "alias_without_anchor",
                "weak_reference",
                "weak_identity_edge",
                "hard_conflict",
            } or ledger_subject_status in {
                "ambiguous_short_subject",
                "unresolved_alias",
                "weak_reference",
                "weak_identity_edge",
                "hard_conflict",
            }
            final_subject_review_required = self._is_final_subject_review_target(entity)
            if not isinstance(rule_first, dict) and not ledger_review_required and not final_subject_review_required:
                continue
            action = str((rule_first or {}).get("action") or "")
            risk_level = str((rule_first or {}).get("risk_level") or "")
            if action != "review" and risk_level != "high" and not ledger_review_required and not final_subject_review_required:
                continue
            start = max(0, int(entity.start))
            end = min(len(text), int(entity.end))
            if end <= start:
                continue
            snippet_type = "ledger_conflict_adjudication" if ledger_review_required else "rule_first_review_block"
            risk_reason = (
                f"subject_ledger:{ledger_status or ledger_subject_status}"
                if ledger_review_required
                else (
                    "rule_first:final_subject_adjudication"
                    if final_subject_review_required
                    else f"rule_first:{action or risk_level}"
                )
            )
            snippet = self._make_window_snippet(
                text,
                start,
                snippet_type=snippet_type,
                risk_reason=risk_reason,
                max_chars=max_chars,
            )
            snippets.append(
                RiskSnippet(
                    snippet_type=snippet.snippet_type,
                    risk_reason=snippet.risk_reason,
                    start=snippet.start,
                    end=snippet.end,
                    text=snippet.text,
                    target_entity={
                        "type": entity.entity_type,
                        "text": entity.text,
                        "start": entity.start,
                        "end": entity.end,
                        "source": entity.source,
                        "rule_first": rule_first,
                        "canonical_key": metadata.get("canonical_key"),
                        "metadata": {
                            **metadata,
                            "subject_ledger_occurrence_id": metadata.get("subject_ledger_occurrence_id"),
                            "subject_ledger_subject_id": metadata.get("subject_ledger_subject_id"),
                            "subject_ledger_status": metadata.get("subject_ledger_status"),
                            "subject_ledger_subject_status": metadata.get("subject_ledger_subject_status"),
                            "subject_ledger_family": metadata.get("subject_ledger_family"),
                            "subject_ledger_canonical_text": metadata.get("subject_ledger_canonical_text"),
                            "subject_ledger_canonical_key": metadata.get("subject_ledger_canonical_key"),
                            "subject_ledger_edge_id": metadata.get("subject_ledger_edge_id"),
                            "subject_ledger_edge_relation": metadata.get("subject_ledger_edge_relation"),
                            "subject_ledger_edge_status": metadata.get("subject_ledger_edge_status"),
                            "subject_ledger_edge_target_subject_id": metadata.get("subject_ledger_edge_target_subject_id"),
                            "subject_ledger_edge_target_canonical_text": metadata.get("subject_ledger_edge_target_canonical_text"),
                            "subject_ledger_edge_target_canonical_key": metadata.get("subject_ledger_edge_target_canonical_key"),
                            "subject_ledger_edge_evidence": metadata.get("subject_ledger_edge_evidence"),
                        },
                    },
                )
            )
        return snippets

    @staticmethod
    def _is_final_subject_review_target(entity: RecognizerResult) -> bool:
        if projected_default_subject_type(entity) not in DEFAULT_SUBJECT_TYPES:
            return False
        metadata = dict(entity.metadata or {})
        rule_first = metadata.get("rule_first")
        if isinstance(rule_first, dict):
            return True
        source = str(entity.source or "").strip()
        return source.startswith("rule_") or str(metadata.get("source_layer") or "") in {"structure", "rule"}

    def _build_structure_snippets(
        self,
        *,
        text: str,
        source_structure: dict[str, Any] | None,
        max_chars: int,
    ) -> List[RiskSnippet]:
        if not isinstance(source_structure, dict):
            return []
        snippets: List[RiskSnippet] = []
        for unit in self._iter_structure_units(source_structure):
            unit_text = str(unit.get("text") or "").strip()
            if not unit_text:
                continue
            container_type = str(unit.get("container_type") or unit.get("unit_type") or "").strip()
            if container_type not in {"table_cell", "textbox", "header", "footer", "footnote", "endnote"}:
                continue
            if not self._structure_unit_has_sensitive_cue(unit_text, container_type):
                continue
            start = self._coerce_int(unit.get("start"), -1)
            if start < 0 or text[start : start + len(unit_text)] != unit_text:
                start = text.find(unit_text)
            if start >= 0:
                unit_end = min(len(text), start + min(len(unit_text), max_chars))
                snippets.append(
                    RiskSnippet(
                        snippet_type=f"docx_{container_type}_block",
                        risk_reason=f"docx_structure:{container_type}",
                        start=start,
                        end=unit_end,
                        text=text[start:unit_end],
                    )
                )
                continue
            snippets.append(
                RiskSnippet(
                    snippet_type=f"docx_{container_type}_block",
                    risk_reason=f"docx_structure:{container_type}",
                    start=0,
                    end=0,
                    text=unit_text[:max_chars],
                )
            )
        return snippets

    def _iter_structure_units(self, source_structure: dict[str, Any]):
        seen_unit_ids: set[str] = set()
        raw_units = source_structure.get("docx_text_units")
        if isinstance(raw_units, list):
            for unit in raw_units:
                if not isinstance(unit, dict):
                    continue
                unit_id = str(unit.get("unit_id") or "")
                if unit_id:
                    seen_unit_ids.add(unit_id)
                yield unit

        pages = source_structure.get("pages")
        if not isinstance(pages, list):
            return
        for page in pages:
            if not isinstance(page, dict):
                continue
            units = page.get("units")
            if not isinstance(units, list):
                continue
            for unit in units:
                if isinstance(unit, dict):
                    unit_id = str(unit.get("unit_id") or "")
                    if unit_id and unit_id in seen_unit_ids:
                        continue
                    if unit_id:
                        seen_unit_ids.add(unit_id)
                    yield unit

    def _coerce_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _structure_unit_has_sensitive_cue(self, unit_text: str, container_type: str) -> bool:
        compact = re.sub(r"\s+", "", unit_text)
        if not compact:
            return False
        cue_keywords = {
            "甲方",
            "乙方",
            "丙方",
            "委托方",
            "受托方",
            "发包人",
            "承包人",
            "供应商",
            "联系人",
            "联系电话",
            "法定代表人",
            "负责人",
            "地址",
            "住所",
            "开户行",
            "账户",
            "账号",
            "户名",
            "项目",
            "合同",
            "签章",
            "盖章",
        }
        if any(keyword in compact for keyword in cue_keywords):
            return True
        if container_type in {"table_cell", "textbox"}:
            return bool(re.search(r"[\u4e00-\u9fa5A-Za-z0-9·]{2,}(?:公司|集团|银行|法院|项目|工程|中心|研究院|事务所)", compact))
        return False

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
        seen: set[tuple[object, ...]] = set()
        ledger_snippets: list[RiskSnippet] = []
        final_subject_snippets: list[RiskSnippet] = []
        structure_snippets: list[RiskSnippet] = []
        ordinary_snippets: list[RiskSnippet] = []
        for snippet in sorted(enumerate(snippets), key=lambda item: RiskSnippetScheduler._snippet_priority(item[1], item[0])):
            snippet = snippet[1]
            key = RiskSnippetScheduler._dedupe_key(snippet)
            if key in seen or not snippet.text.strip():
                continue
            seen.add(key)
            if RiskSnippetScheduler._is_ledger_review_snippet(snippet):
                ledger_snippets.append(snippet)
            elif RiskSnippetScheduler._is_rule_first_review_snippet(snippet):
                final_subject_snippets.append(snippet)
            elif RiskSnippetScheduler._is_docx_structure_snippet(snippet):
                structure_snippets.append(snippet)
            else:
                ordinary_snippets.append(snippet)
        # Ledger conflicts are mandatory adjudication work, but they must not
        # consume the ordinary review budget. Otherwise a conflict-heavy run
        # leaves the final review with no chance to reject obvious bad spans.
        ordinary_limit = max(1, int(max_count or 1))
        structure_limit = max(2, min(len(structure_snippets), max(ordinary_limit // 2, 6)))
        remaining_ordinary_limit = max(1, ordinary_limit - min(structure_limit, len(structure_snippets)))
        return [
            *ledger_snippets,
            *final_subject_snippets,
            *structure_snippets[:structure_limit],
            *ordinary_snippets[:remaining_ordinary_limit],
        ]

    @staticmethod
    def _dedupe_key(snippet: RiskSnippet) -> tuple[object, ...]:
        if RiskSnippetScheduler._is_ledger_review_snippet(snippet):
            target = snippet.target_entity if isinstance(snippet.target_entity, dict) else {}
            metadata = target.get("metadata") if isinstance(target.get("metadata"), dict) else {}
            occurrence_id = str(metadata.get("subject_ledger_occurrence_id") or "").strip()
            edge_id = str(metadata.get("subject_ledger_edge_id") or "").strip()
            subject_id = str(metadata.get("subject_ledger_subject_id") or "").strip()
            target_text = str(target.get("text") or "").strip()
            target_start = target.get("start")
            target_end = target.get("end")
            return (
                "ledger",
                occurrence_id or f"{target_start}:{target_end}:{target_text}",
                edge_id,
                subject_id,
                snippet.snippet_type,
            )
        if RiskSnippetScheduler._is_rule_first_review_snippet(snippet):
            target = snippet.target_entity if isinstance(snippet.target_entity, dict) else {}
            target_text = str(target.get("text") or "").strip()
            target_type = str(target.get("type") or target.get("entity_type") or "").strip()
            target_start = target.get("start")
            target_end = target.get("end")
            return ("rule_first_review", target_type, target_start, target_end, target_text)
        return (snippet.start, snippet.end, snippet.snippet_type)

    @staticmethod
    def _is_ledger_review_snippet(snippet: RiskSnippet) -> bool:
        return (
            snippet.snippet_type == "ledger_conflict_adjudication"
            or str(snippet.risk_reason or "").startswith("subject_ledger:")
        )

    @staticmethod
    def _is_docx_structure_snippet(snippet: RiskSnippet) -> bool:
        return str(snippet.risk_reason or "").startswith("docx_structure:")

    @staticmethod
    def _is_rule_first_review_snippet(snippet: RiskSnippet) -> bool:
        return (
            snippet.snippet_type == "rule_first_review_block"
            or str(snippet.risk_reason or "").startswith("rule_first:")
        )

    @staticmethod
    def _snippet_priority(snippet: RiskSnippet, index: int) -> tuple[int, int]:
        if RiskSnippetScheduler._is_ledger_review_snippet(snippet):
            return (0, index)
        if snippet.snippet_type == "rule_first_review_block":
            return (1, index)
        if str(snippet.risk_reason or "").startswith("docx_structure:"):
            return (2, index)
        if snippet.snippet_type in {"conflict_block", "legal_party_block", "header_party_block"}:
            return (3, index)
        if snippet.snippet_type in {"account_block", "address_block"}:
            return (4, index)
        return (5, index)
