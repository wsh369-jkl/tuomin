from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence

from .common import normalize_text, table_text_artifact_replacement
from .models import (
    ContentCoverageReview,
    ConversionPreflightResult,
    DocxEvidenceUnit,
    TableCellEvidenceReview,
    TableGridEvidence,
)


class TableGridEvidenceBuilder:
    """Build page/table/row/column evidence from DOCX cells and OCR-backed reviews.

    This is a structural evidence layer. It does not create comments by itself;
    it makes table pages inspectable as grids so later gates can reason about
    whole columns and rows instead of isolated OCR snippets.
    """

    MAX_ANOMALIES_PER_GRID = 80
    MAX_COLUMN_SAMPLES = 8

    def build(self, *, preflight_result: ConversionPreflightResult) -> List[TableGridEvidence]:
        docx_cells = [
            unit
            for unit in preflight_result.docx_units
            if unit.container_type == "table_cell"
            and int(unit.estimated_page_no or 0) > 0
            and int(unit.table_index or 0) > 0
        ]
        if not docx_cells:
            return []
        coverage_by_unit = {
            review.unit_id: review
            for review in preflight_result.content_coverage_reviews
            if review.side == "docx" and review.unit_id
        }
        evidence_by_unit = {
            review.docx_unit_id: review
            for review in preflight_result.table_cell_evidence_reviews
            if review.docx_unit_id
        }
        no_issue_units = self._table_vl_no_issue_units(preflight_result=preflight_result)
        groups: Dict[tuple[int, int], List[DocxEvidenceUnit]] = defaultdict(list)
        for unit in docx_cells:
            groups[(int(unit.estimated_page_no or 0), int(unit.table_index or 0))].append(unit)

        table_parse_by_page = {int(item.page_no or 0): item for item in preflight_result.table_reviews}
        rows: List[TableGridEvidence] = []
        for (page_no, table_index), units in sorted(groups.items()):
            units = sorted(units, key=lambda item: (int(item.row_index or 0), int(item.col_index or 0), int(item.order_index or 0)))
            grid = self._build_grid(
                page_no=page_no,
                table_index=table_index,
                units=units,
                coverage_by_unit=coverage_by_unit,
                evidence_by_unit=evidence_by_unit,
                no_issue_units=no_issue_units,
                parse_status=str(getattr(table_parse_by_page.get(page_no), "status", "") or ""),
                parse_confidence=float(getattr(table_parse_by_page.get(page_no), "confidence", 0.0) or 0.0),
            )
            rows.append(grid)
        for index, grid in enumerate(rows, start=1):
            grid.grid_id = f"table_grid_{index:04d}"
        return rows

    def _build_grid(
        self,
        *,
        page_no: int,
        table_index: int,
        units: Sequence[DocxEvidenceUnit],
        coverage_by_unit: Dict[str, ContentCoverageReview],
        evidence_by_unit: Dict[str, TableCellEvidenceReview],
        no_issue_units: set[str],
        parse_status: str,
        parse_confidence: float,
    ) -> TableGridEvidence:
        row_count = max((int(unit.row_index or 0) for unit in units), default=0)
        col_count = max((int(unit.col_index or 0) for unit in units), default=0)
        nonempty = [unit for unit in units if normalize_text(unit.text)]
        coverage_status_counts: Counter[str] = Counter()
        evidence_decision_counts: Counter[str] = Counter()
        columns: Dict[int, List[DocxEvidenceUnit]] = defaultdict(list)
        for unit in units:
            columns[int(unit.col_index or 0)].append(unit)
            coverage = coverage_by_unit.get(unit.unit_id)
            if coverage:
                coverage_status_counts[coverage.status or "unknown"] += 1
            evidence = evidence_by_unit.get(unit.unit_id)
            if evidence:
                evidence_decision_counts[evidence.decision or "unknown"] += 1

        column_profiles = [
            self._column_profile(
                col_index=col_index,
                units=col_units,
                coverage_by_unit=coverage_by_unit,
                evidence_by_unit=evidence_by_unit,
            )
            for col_index, col_units in sorted(columns.items())
            if col_index > 0
        ]
        decimal_context_cols = {
            int(profile["col_index"])
            for profile in column_profiles
            if int(profile.get("decimal_evidence_count") or 0) >= 2
        }
        anomaly_cells = self._anomaly_cells(
            units=units,
            coverage_by_unit=coverage_by_unit,
            evidence_by_unit=evidence_by_unit,
            decimal_context_cols=decimal_context_cols,
        )
        confirmed = sum(1 for item in anomaly_cells if item.get("decision") == "confirmed_error")
        suspected = sum(1 for item in anomaly_cells if item.get("decision") == "suspected_error")
        unresolved = sum(
            1
            for unit in units
            for coverage in [coverage_by_unit.get(unit.unit_id)]
            if coverage is not None and coverage.decision != "covered"
        )
        no_issue_sample_count = sum(1 for unit in units if unit.unit_id in no_issue_units)
        status = self._grid_status(
            confirmed=confirmed,
            suspected=suspected,
            anomaly_count=len(anomaly_cells),
            unresolved=unresolved,
            no_issue_sample_count=no_issue_sample_count,
        )
        flags = [
            "table_grid_evidence",
            f"parse_status={parse_status or 'unknown'}",
        ]
        if decimal_context_cols:
            flags.append("has_decimal_context_columns")
        if no_issue_sample_count:
            flags.append("has_table_page_vl_no_issue_samples")
        route_hints = self._route_hints(
            anomaly_count=len(anomaly_cells),
            confirmed=confirmed,
            suspected=suspected,
            unresolved=unresolved,
            decimal_context_cols=decimal_context_cols,
        )
        confidence = max(float(parse_confidence or 0.0), 0.76 if confirmed else 0.64 if anomaly_cells else 0.52)
        return TableGridEvidence(
            grid_id="",
            page_no=page_no,
            table_index=table_index,
            status=status,
            confidence=confidence,
            row_count=row_count,
            col_count=col_count,
            cell_count=len(units),
            nonempty_cell_count=len(nonempty),
            unresolved_cell_count=unresolved,
            confirmed_error_count=confirmed,
            suspected_error_count=suspected,
            no_issue_sample_count=no_issue_sample_count,
            column_profiles=column_profiles,
            anomaly_cells=anomaly_cells[: self.MAX_ANOMALIES_PER_GRID],
            route_hints=route_hints,
            coverage_status_counts=dict(sorted(coverage_status_counts.items())),
            evidence_decision_counts=dict(sorted(evidence_decision_counts.items())),
            flags=flags,
        )

    def _column_profile(
        self,
        *,
        col_index: int,
        units: Sequence[DocxEvidenceUnit],
        coverage_by_unit: Dict[str, ContentCoverageReview],
        evidence_by_unit: Dict[str, TableCellEvidenceReview],
    ) -> Dict[str, Any]:
        nonempty = [unit for unit in units if normalize_text(unit.text)]
        type_counts = Counter(self._value_type(unit.text) for unit in nonempty)
        evidence_rows = [evidence_by_unit[unit.unit_id] for unit in units if unit.unit_id in evidence_by_unit]
        unresolved_count = sum(
            1
            for unit in units
            for coverage in [coverage_by_unit.get(unit.unit_id)]
            if coverage is not None and coverage.decision != "covered"
        )
        header_samples = [self._clip(unit.text, 32) for unit in units if int(unit.row_index or 0) <= 3 and normalize_text(unit.text)]
        value_samples = [self._clip(unit.text, 32) for unit in nonempty if int(unit.row_index or 0) > 3]
        return {
            "col_index": int(col_index),
            "cell_count": len(units),
            "nonempty_cell_count": len(nonempty),
            "dominant_type": type_counts.most_common(1)[0][0] if type_counts else "empty",
            "type_counts": dict(sorted(type_counts.items())),
            "unresolved_count": unresolved_count,
            "evidence_count": len(evidence_rows),
            "confirmed_evidence_count": sum(1 for item in evidence_rows if item.decision == "confirmed_error"),
            "suspected_evidence_count": sum(1 for item in evidence_rows if item.decision == "suspected_error"),
            "decimal_evidence_count": sum(1 for item in evidence_rows if item.issue_type in {"decimal_point_missing", "decimal_or_punctuation_pollution"}),
            "header_samples": header_samples[: self.MAX_COLUMN_SAMPLES],
            "value_samples": value_samples[: self.MAX_COLUMN_SAMPLES],
        }

    def _anomaly_cells(
        self,
        *,
        units: Sequence[DocxEvidenceUnit],
        coverage_by_unit: Dict[str, ContentCoverageReview],
        evidence_by_unit: Dict[str, TableCellEvidenceReview],
        decimal_context_cols: set[int],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for unit in units:
            evidence = evidence_by_unit.get(unit.unit_id)
            if evidence is not None:
                rows.append(self._evidence_anomaly(unit=unit, evidence=evidence))
                continue
            text = " ".join(str(unit.text or "").split())
            if not text:
                continue
            replacement = table_text_artifact_replacement(text)
            if replacement:
                rows.append(
                    self._rule_anomaly(
                        unit=unit,
                        coverage=coverage_by_unit.get(unit.unit_id),
                        anomaly_type="table_header_text_artifact",
                        expected_text=replacement,
                        confidence=0.58,
                        reason="该表格短文本符合通用表头误识别形态，需结合 PDF 表头确认。",
                        flags=["table_grid_rule_candidate", "table_header_artifact_candidate"],
                    )
                )
                continue
            if self._confusable_numeric(text):
                rows.append(
                    self._rule_anomaly(
                        unit=unit,
                        coverage=coverage_by_unit.get(unit.unit_id),
                        anomaly_type="confusable_numeric_text",
                        expected_text="",
                        confidence=0.55,
                        reason="该单元为数字主体但混入易混字母，需结合 PDF 表格值确认。",
                        flags=["table_grid_rule_candidate", "confusable_numeric_candidate"],
                    )
                )
                continue
            if int(unit.col_index or 0) in decimal_context_cols and self._plain_integer_decimal_candidate(text):
                rows.append(
                    self._rule_anomaly(
                        unit=unit,
                        coverage=coverage_by_unit.get(unit.unit_id),
                        anomaly_type="column_decimal_missing_candidate",
                        expected_text="",
                        confidence=0.54,
                        reason="同列已有多个小数点缺失证据，该纯数字单元需按列级模式继续核查。",
                        flags=["table_grid_rule_candidate", "column_decimal_context_candidate"],
                    )
                )
        rows.sort(key=lambda item: (int(item.get("row_index") or 0), int(item.get("col_index") or 0), item.get("unit_id", "")))
        for index, row in enumerate(rows, start=1):
            row["anomaly_id"] = f"grid_anomaly_{index:04d}"
        return rows

    def _evidence_anomaly(self, *, unit: DocxEvidenceUnit, evidence: TableCellEvidenceReview) -> Dict[str, Any]:
        return {
            "anomaly_id": "",
            "unit_id": unit.unit_id,
            "row_index": unit.row_index,
            "col_index": unit.col_index,
            "text": self._clip(unit.text, 120),
            "normalized_text": normalize_text(unit.text),
            "anomaly_type": evidence.issue_type or "table_cell_evidence",
            "decision": evidence.decision,
            "expected_text": evidence.visible_text,
            "confidence": round(float(evidence.confidence or 0.0), 4),
            "coverage_review_id": evidence.coverage_review_id,
            "evidence_review_id": evidence.review_id,
            "reason": evidence.reason,
            "flags": list(dict.fromkeys(["table_cell_evidence_linked", *evidence.flags])),
        }

    def _rule_anomaly(
        self,
        *,
        unit: DocxEvidenceUnit,
        coverage: ContentCoverageReview | None,
        anomaly_type: str,
        expected_text: str,
        confidence: float,
        reason: str,
        flags: Sequence[str],
    ) -> Dict[str, Any]:
        return {
            "anomaly_id": "",
            "unit_id": unit.unit_id,
            "row_index": unit.row_index,
            "col_index": unit.col_index,
            "text": self._clip(unit.text, 120),
            "normalized_text": normalize_text(unit.text),
            "anomaly_type": anomaly_type,
            "decision": "suspected_error",
            "expected_text": expected_text,
            "confidence": round(float(confidence or 0.0), 4),
            "coverage_review_id": coverage.review_id if coverage else "",
            "evidence_review_id": "",
            "reason": reason,
            "flags": list(dict.fromkeys(flags)),
        }

    def _table_vl_no_issue_units(self, *, preflight_result: ConversionPreflightResult) -> set[str]:
        rows: set[str] = set()
        for review in preflight_result.table_page_vl_reviews:
            if not review.available or review.verdict != "no_obvious_issue" or review.suspicious_values:
                continue
            for sample in review.docx_samples:
                unit_id = str(sample.get("unit_id") or "").strip()
                if unit_id:
                    rows.add(unit_id)
        return rows

    def _grid_status(
        self,
        *,
        confirmed: int,
        suspected: int,
        anomaly_count: int,
        unresolved: int,
        no_issue_sample_count: int,
    ) -> str:
        if confirmed:
            return "grid_with_confirmed_errors"
        if suspected or anomaly_count:
            return "grid_with_suspected_anomalies"
        if no_issue_sample_count and not unresolved:
            return "grid_sampled_no_issue"
        if unresolved:
            return "grid_pending_review"
        return "grid_covered_or_low_risk"

    def _route_hints(
        self,
        *,
        anomaly_count: int,
        confirmed: int,
        suspected: int,
        unresolved: int,
        decimal_context_cols: set[int],
    ) -> List[str]:
        hints: List[str] = []
        if confirmed:
            hints.append("comment_confirmed_table_cells")
        if suspected:
            hints.append("needs_table_column_pattern_review")
        if decimal_context_cols:
            hints.append("needs_decimal_column_sweep")
        if anomaly_count and not confirmed:
            hints.append("needs_qwen_vl_table_cell_review")
        if unresolved:
            hints.append("needs_table_parser")
        return list(dict.fromkeys(hints))

    def _value_type(self, value: Any) -> str:
        text = normalize_text(str(value or ""))
        if not text:
            return "empty"
        if self._date_like(text):
            return "date"
        if re.fullmatch(r"[¥￥]?\d+(?:\.\d+)?", text.replace(",", "")):
            return "decimal_number" if "." in text else "integer_number"
        if self._confusable_numeric(text):
            return "confusable_numeric"
        if re.search(r"\d", text) and any("\u4e00" <= ch <= "\u9fff" for ch in text):
            return "mixed_text_number"
        if re.search(r"\d", text):
            return "numeric_text"
        return "text"

    def _date_like(self, value: str) -> bool:
        return bool(re.search(r"\d{4}年\d{1,2}月|\d{1,2}月\d{1,2}日|\d{4}[./-]\d{1,2}", value))

    def _confusable_numeric(self, value: str) -> bool:
        compact = normalize_text(value).upper()
        if len(compact) < 3 or not any(ch.isdigit() for ch in compact):
            return False
        if any("\u4e00" <= ch <= "\u9fff" for ch in compact):
            return False
        return any(ch in set("BDEGILOQSZ") for ch in compact)

    def _plain_integer_decimal_candidate(self, value: str) -> bool:
        compact = normalize_text(value)
        if not re.fullmatch(r"\d{4,8}", compact):
            return False
        return True

    def _clip(self, value: Any, limit: int) -> str:
        return " ".join(str(value or "").split())[:limit]
