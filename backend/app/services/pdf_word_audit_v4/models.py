from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PdfEvidenceUnit:
    unit_id: str
    page_no: int
    unit_type: str
    text: str = ""
    normalized_text: str = ""
    bbox: List[float] = field(default_factory=list)
    source: str = "pdf_native_text"
    confidence: float = 0.0
    order_index: int = 0
    region_type: str = ""
    flags: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "page_no": int(self.page_no),
            "unit_type": self.unit_type,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "text_chars": len(self.normalized_text),
            "bbox": list(self.bbox),
            "source": self.source,
            "confidence": round(float(self.confidence or 0.0), 4),
            "order_index": int(self.order_index or 0),
            "region_type": self.region_type,
            "flags": list(self.flags),
            "metrics": dict(self.metrics),
        }


@dataclass
class DocxEvidenceUnit:
    unit_id: str
    part_name: str
    unit_type: str
    container_type: str
    text: str
    normalized_text: str
    order_index: int
    estimated_page_no: Optional[int] = None
    table_index: Optional[int] = None
    row_index: Optional[int] = None
    col_index: Optional[int] = None
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "part_name": self.part_name,
            "unit_type": self.unit_type,
            "container_type": self.container_type,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "text_chars": len(self.normalized_text),
            "order_index": int(self.order_index or 0),
            "estimated_page_no": self.estimated_page_no,
            "table_index": self.table_index,
            "row_index": self.row_index,
            "col_index": self.col_index,
            "flags": list(self.flags),
        }


@dataclass
class AlignmentLink:
    link_id: str
    pdf_unit_id: str
    docx_unit_id: str
    pdf_page_no: int
    docx_estimated_page_no: Optional[int]
    alignment_type: str
    confidence: float
    status: str
    anchors: List[Dict[str, str]] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "link_id": self.link_id,
            "pdf_unit_id": self.pdf_unit_id,
            "docx_unit_id": self.docx_unit_id,
            "pdf_page_no": int(self.pdf_page_no),
            "docx_estimated_page_no": self.docx_estimated_page_no,
            "alignment_type": self.alignment_type,
            "confidence": round(float(self.confidence or 0.0), 4),
            "status": self.status,
            "anchors": [dict(item) for item in self.anchors],
            "reasons": list(self.reasons),
        }


@dataclass
class ConversionDiffCandidate:
    diff_id: str
    category: str
    risk: str
    pdf_unit_id: str = ""
    docx_unit_id: str = ""
    pdf_page_no: Optional[int] = None
    docx_estimated_page_no: Optional[int] = None
    pdf_text: str = ""
    docx_text: str = ""
    field_type: str = ""
    field_role: str = ""
    pdf_value: str = ""
    docx_value: str = ""
    alignment_confidence: float = 0.0
    confidence: float = 0.0
    reason: str = ""
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diff_id": self.diff_id,
            "category": self.category,
            "risk": self.risk,
            "pdf_unit_id": self.pdf_unit_id,
            "docx_unit_id": self.docx_unit_id,
            "pdf_page_no": self.pdf_page_no,
            "docx_estimated_page_no": self.docx_estimated_page_no,
            "pdf_text": self.pdf_text,
            "docx_text": self.docx_text,
            "field_type": self.field_type,
            "field_role": self.field_role,
            "pdf_value": self.pdf_value,
            "docx_value": self.docx_value,
            "alignment_confidence": round(float(self.alignment_confidence or 0.0), 4),
            "confidence": round(float(self.confidence or 0.0), 4),
            "reason": self.reason,
            "flags": list(self.flags),
        }


@dataclass
class ReviewRoute:
    route_id: str
    route: str
    reason: str
    page_no: Optional[int] = None
    pdf_unit_id: str = ""
    docx_unit_id: str = ""
    diff_id: str = ""
    priority: int = 0
    status: str = "planned"
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route_id": self.route_id,
            "route": self.route,
            "reason": self.reason,
            "page_no": self.page_no,
            "pdf_unit_id": self.pdf_unit_id,
            "docx_unit_id": self.docx_unit_id,
            "diff_id": self.diff_id,
            "priority": int(self.priority or 0),
            "status": self.status,
            "flags": list(self.flags),
        }


@dataclass
class FocusedCandidateReview:
    review_id: str
    diff_id: str
    category: str
    page_no: Optional[int]
    status: str
    decision: str
    confidence: float
    reason: str
    next_route: str = ""
    crop_path: str = ""
    crop_ocr_text: str = ""
    crop_ocr_quality: str = ""
    pdf_text: str = ""
    docx_text: str = ""
    flags: List[str] = field(default_factory=list)
    table_cell: Dict[str, Any] = field(default_factory=dict)
    visual_text: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "review_id": self.review_id,
            "diff_id": self.diff_id,
            "category": self.category,
            "page_no": self.page_no,
            "status": self.status,
            "decision": self.decision,
            "confidence": round(float(self.confidence or 0.0), 4),
            "reason": self.reason,
            "next_route": self.next_route,
            "crop_path": self.crop_path,
            "crop_ocr_text": self.crop_ocr_text,
            "crop_ocr_quality": self.crop_ocr_quality,
            "pdf_text": self.pdf_text,
            "docx_text": self.docx_text,
            "flags": list(self.flags),
            "table_cell": dict(self.table_cell),
            "visual_text": dict(self.visual_text),
        }


@dataclass
class QwenGateReview:
    gate_id: str
    diff_id: str
    attempted: bool
    available: bool
    model: str = ""
    verdict: str = ""
    decision: str = ""
    confidence: float = 0.0
    reason: str = ""
    preferred_text: str = ""
    next_route: str = ""
    error: str = ""
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "diff_id": self.diff_id,
            "attempted": bool(self.attempted),
            "available": bool(self.available),
            "model": self.model,
            "verdict": self.verdict,
            "decision": self.decision,
            "confidence": round(float(self.confidence or 0.0), 4),
            "reason": self.reason,
            "preferred_text": self.preferred_text,
            "next_route": self.next_route,
            "error": self.error,
            "flags": list(self.flags),
        }


@dataclass
class QwenVlGateReview:
    vl_id: str
    diff_id: str
    attempted: bool
    available: bool
    page_no: int = 0
    model: str = ""
    verdict: str = ""
    decision: str = ""
    confidence: float = 0.0
    reason: str = ""
    visible_text: str = ""
    preferred_text: str = ""
    next_route: str = ""
    crop_path: str = ""
    error: str = ""
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vl_id": self.vl_id,
            "diff_id": self.diff_id,
            "page_no": int(self.page_no or 0),
            "attempted": bool(self.attempted),
            "available": bool(self.available),
            "model": self.model,
            "verdict": self.verdict,
            "decision": self.decision,
            "confidence": round(float(self.confidence or 0.0), 4),
            "reason": self.reason,
            "visible_text": self.visible_text,
            "preferred_text": self.preferred_text,
            "next_route": self.next_route,
            "crop_path": self.crop_path,
            "error": self.error,
            "flags": list(self.flags),
        }


@dataclass
class ContentCoverageReview:
    review_id: str
    side: str
    unit_id: str
    page_no: Optional[int]
    status: str
    decision: str
    confidence: float
    reason: str
    text: str = ""
    next_route: str = ""
    related_link_id: str = ""
    related_diff_id: str = ""
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "review_id": self.review_id,
            "side": self.side,
            "unit_id": self.unit_id,
            "page_no": self.page_no,
            "status": self.status,
            "decision": self.decision,
            "confidence": round(float(self.confidence or 0.0), 4),
            "reason": self.reason,
            "text": self.text,
            "next_route": self.next_route,
            "related_link_id": self.related_link_id,
            "related_diff_id": self.related_diff_id,
            "flags": list(self.flags),
        }


@dataclass
class ContentCoverageBackfillReview:
    backfill_id: str
    coverage_review_id: str
    side: str
    unit_id: str
    page_no: Optional[int]
    attempted: bool
    available: bool
    method: str = ""
    status: str = ""
    confidence: float = 0.0
    quality: str = ""
    readability: str = ""
    extracted_text: str = ""
    normalized_text: str = ""
    crop_path: str = ""
    reason: str = ""
    next_route: str = ""
    error: str = ""
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backfill_id": self.backfill_id,
            "coverage_review_id": self.coverage_review_id,
            "side": self.side,
            "unit_id": self.unit_id,
            "page_no": self.page_no,
            "attempted": bool(self.attempted),
            "available": bool(self.available),
            "method": self.method,
            "status": self.status,
            "confidence": round(float(self.confidence or 0.0), 4),
            "quality": self.quality,
            "readability": self.readability,
            "extracted_text": self.extracted_text,
            "normalized_text": self.normalized_text,
            "text_chars": len(self.normalized_text),
            "crop_path": self.crop_path,
            "reason": self.reason,
            "next_route": self.next_route,
            "error": self.error,
            "flags": list(self.flags),
        }


@dataclass
class TableParseReview:
    table_id: str
    page_no: int
    status: str
    confidence: float
    bbox: List[float] = field(default_factory=list)
    row_count: int = 0
    col_count: int = 0
    horizontal_line_count: int = 0
    vertical_line_count: int = 0
    related_diff_ids: List[str] = field(default_factory=list)
    docx_table_candidates: List[Dict[str, Any]] = field(default_factory=list)
    cell_evidence: List[Dict[str, Any]] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_id": self.table_id,
            "page_no": int(self.page_no),
            "status": self.status,
            "confidence": round(float(self.confidence or 0.0), 4),
            "bbox": list(self.bbox),
            "row_count": int(self.row_count or 0),
            "col_count": int(self.col_count or 0),
            "horizontal_line_count": int(self.horizontal_line_count or 0),
            "vertical_line_count": int(self.vertical_line_count or 0),
            "related_diff_ids": list(self.related_diff_ids),
            "docx_table_candidates": [dict(item) for item in self.docx_table_candidates],
            "cell_evidence": [dict(item) for item in self.cell_evidence],
            "flags": list(self.flags),
        }


@dataclass
class TablePageVlReview:
    review_id: str
    page_no: int
    attempted: bool
    available: bool
    model: str = ""
    verdict: str = ""
    confidence: float = 0.0
    reason: str = ""
    page_image_path: str = ""
    review_image_path: str = ""
    orientation_degrees: int = 0
    orientation_confidence: float = 0.0
    orientation_reason: str = ""
    visible_text_excerpt: str = ""
    suspicious_values: List[Dict[str, Any]] = field(default_factory=list)
    docx_samples: List[Dict[str, Any]] = field(default_factory=list)
    next_route: str = ""
    error: str = ""
    model_parse_error: str = ""
    model_raw_content_excerpt: str = ""
    model_parsed_keys: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "review_id": self.review_id,
            "page_no": int(self.page_no),
            "attempted": bool(self.attempted),
            "available": bool(self.available),
            "model": self.model,
            "verdict": self.verdict,
            "confidence": round(float(self.confidence or 0.0), 4),
            "reason": self.reason,
            "page_image_path": self.page_image_path,
            "review_image_path": self.review_image_path,
            "orientation_degrees": int(self.orientation_degrees or 0),
            "orientation_confidence": round(float(self.orientation_confidence or 0.0), 4),
            "orientation_reason": self.orientation_reason,
            "visible_text_excerpt": self.visible_text_excerpt,
            "suspicious_values": [dict(item) for item in self.suspicious_values],
            "docx_samples": [dict(item) for item in self.docx_samples],
            "next_route": self.next_route,
            "error": self.error,
            "model_parse_error": self.model_parse_error,
            "model_raw_content_excerpt": self.model_raw_content_excerpt,
            "model_parsed_keys": list(self.model_parsed_keys),
            "flags": list(self.flags),
        }


@dataclass
class TableCellEvidenceReview:
    review_id: str
    page_no: int
    docx_unit_id: str
    table_index: Optional[int]
    row_index: Optional[int]
    col_index: Optional[int]
    docx_text: str
    status: str
    decision: str
    confidence: float
    visible_text: str = ""
    issue_type: str = ""
    evidence_source: str = "pdf_page_ocr"
    reason: str = ""
    normalized_docx: str = ""
    normalized_visible: str = ""
    matched_token: str = ""
    coverage_review_id: str = ""
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "review_id": self.review_id,
            "page_no": int(self.page_no),
            "docx_unit_id": self.docx_unit_id,
            "table_index": self.table_index,
            "row_index": self.row_index,
            "col_index": self.col_index,
            "docx_text": self.docx_text,
            "visible_text": self.visible_text,
            "status": self.status,
            "decision": self.decision,
            "confidence": round(float(self.confidence or 0.0), 4),
            "issue_type": self.issue_type,
            "evidence_source": self.evidence_source,
            "reason": self.reason,
            "normalized_docx": self.normalized_docx,
            "normalized_visible": self.normalized_visible,
            "matched_token": self.matched_token,
            "coverage_review_id": self.coverage_review_id,
            "flags": list(self.flags),
        }


@dataclass
class TableGridEvidence:
    grid_id: str
    page_no: int
    table_index: int
    status: str
    confidence: float
    row_count: int = 0
    col_count: int = 0
    cell_count: int = 0
    nonempty_cell_count: int = 0
    unresolved_cell_count: int = 0
    confirmed_error_count: int = 0
    suspected_error_count: int = 0
    no_issue_sample_count: int = 0
    column_profiles: List[Dict[str, Any]] = field(default_factory=list)
    anomaly_cells: List[Dict[str, Any]] = field(default_factory=list)
    route_hints: List[str] = field(default_factory=list)
    coverage_status_counts: Dict[str, int] = field(default_factory=dict)
    evidence_decision_counts: Dict[str, int] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grid_id": self.grid_id,
            "page_no": int(self.page_no),
            "table_index": int(self.table_index),
            "status": self.status,
            "confidence": round(float(self.confidence or 0.0), 4),
            "row_count": int(self.row_count or 0),
            "col_count": int(self.col_count or 0),
            "cell_count": int(self.cell_count or 0),
            "nonempty_cell_count": int(self.nonempty_cell_count or 0),
            "unresolved_cell_count": int(self.unresolved_cell_count or 0),
            "confirmed_error_count": int(self.confirmed_error_count or 0),
            "suspected_error_count": int(self.suspected_error_count or 0),
            "no_issue_sample_count": int(self.no_issue_sample_count or 0),
            "column_profiles": [dict(item) for item in self.column_profiles],
            "anomaly_cells": [dict(item) for item in self.anomaly_cells],
            "route_hints": list(self.route_hints),
            "coverage_status_counts": dict(self.coverage_status_counts),
            "evidence_decision_counts": dict(self.evidence_decision_counts),
            "flags": list(self.flags),
        }


@dataclass
class FragmentAnomalyReview:
    review_id: str
    page_no: int
    anchor_unit_id: str
    verdict: str
    confidence: float
    reason: str = ""
    repeated_terms: List[Dict[str, Any]] = field(default_factory=list)
    docx_examples: List[Dict[str, Any]] = field(default_factory=list)
    pdf_anchor_excerpt: str = ""
    fragment_count: int = 0
    repeated_fragment_count: int = 0
    short_fragment_count: int = 0
    unique_fragment_count: int = 0
    next_route: str = "needs_human_page_review"
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "review_id": self.review_id,
            "page_no": int(self.page_no),
            "anchor_unit_id": self.anchor_unit_id,
            "verdict": self.verdict,
            "confidence": round(float(self.confidence or 0.0), 4),
            "reason": self.reason,
            "repeated_terms": [dict(item) for item in self.repeated_terms],
            "docx_examples": [dict(item) for item in self.docx_examples],
            "pdf_anchor_excerpt": self.pdf_anchor_excerpt,
            "fragment_count": int(self.fragment_count or 0),
            "repeated_fragment_count": int(self.repeated_fragment_count or 0),
            "short_fragment_count": int(self.short_fragment_count or 0),
            "unique_fragment_count": int(self.unique_fragment_count or 0),
            "next_route": self.next_route,
            "flags": list(self.flags),
        }


@dataclass
class ImagePdfPageReview:
    review_id: str
    page_no: int
    page_kind: str
    verdict: str
    risk_level: str
    confidence: float
    anchor_unit_id: str = ""
    reason: str = ""
    labels: List[str] = field(default_factory=list)
    native_text_reliable: bool = False
    image_area_ratio: float = 0.0
    dark_pixel_ratio: float = 0.0
    ocr_quality: str = ""
    ocr_confidence: float = 0.0
    ocr_text_chars: int = 0
    ocr_line_count: int = 0
    ocr_anchor_count: int = 0
    docx_unit_count: int = 0
    docx_text_chars: int = 0
    unresolved_count: int = 0
    coverage_status_counts: Dict[str, int] = field(default_factory=dict)
    fragment_anomaly_count: int = 0
    table_like: bool = False
    docx_samples: List[Dict[str, Any]] = field(default_factory=list)
    pdf_ocr_excerpt: str = ""
    next_route: str = "needs_full_page_review"
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "review_id": self.review_id,
            "page_no": int(self.page_no),
            "page_kind": self.page_kind,
            "verdict": self.verdict,
            "risk_level": self.risk_level,
            "confidence": round(float(self.confidence or 0.0), 4),
            "anchor_unit_id": self.anchor_unit_id,
            "reason": self.reason,
            "labels": list(self.labels),
            "native_text_reliable": bool(self.native_text_reliable),
            "image_area_ratio": round(float(self.image_area_ratio or 0.0), 4),
            "dark_pixel_ratio": round(float(self.dark_pixel_ratio or 0.0), 6),
            "ocr_quality": self.ocr_quality,
            "ocr_confidence": round(float(self.ocr_confidence or 0.0), 4),
            "ocr_text_chars": int(self.ocr_text_chars or 0),
            "ocr_line_count": int(self.ocr_line_count or 0),
            "ocr_anchor_count": int(self.ocr_anchor_count or 0),
            "docx_unit_count": int(self.docx_unit_count or 0),
            "docx_text_chars": int(self.docx_text_chars or 0),
            "unresolved_count": int(self.unresolved_count or 0),
            "coverage_status_counts": dict(self.coverage_status_counts),
            "fragment_anomaly_count": int(self.fragment_anomaly_count or 0),
            "table_like": bool(self.table_like),
            "docx_samples": [dict(item) for item in self.docx_samples],
            "pdf_ocr_excerpt": self.pdf_ocr_excerpt,
            "next_route": self.next_route,
            "flags": list(self.flags),
        }


@dataclass
class ImagePageVlTextReview:
    review_id: str
    page_no: int
    attempted: bool
    available: bool
    model: str
    verdict: str
    decision: str
    confidence: float = 0.0
    reason: str = ""
    page_image_path: str = ""
    visible_text_excerpt: str = ""
    suspicious_values: List[Dict[str, Any]] = field(default_factory=list)
    docx_samples: List[Dict[str, Any]] = field(default_factory=list)
    next_route: str = ""
    error: str = ""
    model_parse_error: str = ""
    model_raw_content_excerpt: str = ""
    model_parsed_keys: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "review_id": self.review_id,
            "page_no": int(self.page_no),
            "attempted": bool(self.attempted),
            "available": bool(self.available),
            "model": self.model,
            "verdict": self.verdict,
            "decision": self.decision,
            "confidence": round(float(self.confidence or 0.0), 4),
            "reason": self.reason,
            "page_image_path": self.page_image_path,
            "visible_text_excerpt": self.visible_text_excerpt,
            "suspicious_values": [dict(item) for item in self.suspicious_values],
            "docx_samples": [dict(item) for item in self.docx_samples],
            "next_route": self.next_route,
            "error": self.error,
            "model_parse_error": self.model_parse_error,
            "model_raw_content_excerpt": self.model_raw_content_excerpt,
            "model_parsed_keys": list(self.model_parsed_keys),
            "flags": list(self.flags),
        }


@dataclass
class PageOcrTextEvidenceReview:
    review_id: str
    page_no: int
    docx_unit_id: str
    coverage_review_id: str
    docx_text: str
    suggested_text: str
    status: str
    decision: str
    confidence: float = 0.0
    issue_type: str = ""
    evidence_source: str = "anchor_ocr"
    reason: str = ""
    ocr_text_excerpt: str = ""
    normalized_docx: str = ""
    normalized_suggestion: str = ""
    ocr_support_score: float = 0.0
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "review_id": self.review_id,
            "page_no": int(self.page_no),
            "docx_unit_id": self.docx_unit_id,
            "coverage_review_id": self.coverage_review_id,
            "docx_text": self.docx_text,
            "suggested_text": self.suggested_text,
            "status": self.status,
            "decision": self.decision,
            "confidence": round(float(self.confidence or 0.0), 4),
            "issue_type": self.issue_type,
            "evidence_source": self.evidence_source,
            "reason": self.reason,
            "ocr_text_excerpt": self.ocr_text_excerpt,
            "normalized_docx": self.normalized_docx,
            "normalized_suggestion": self.normalized_suggestion,
            "ocr_support_score": round(float(self.ocr_support_score or 0.0), 4),
            "flags": list(self.flags),
        }


@dataclass
class PageTextQwenReview:
    review_id: str
    page_no: int
    attempted: bool
    available: bool
    model: str
    status: str
    decision: str
    verdict: str = ""
    confidence: float = 0.0
    reason: str = ""
    pdf_text_excerpt: str = ""
    docx_samples: List[Dict[str, Any]] = field(default_factory=list)
    suspicious_values: List[Dict[str, Any]] = field(default_factory=list)
    next_route: str = ""
    error: str = ""
    model_parse_error: str = ""
    model_raw_content_excerpt: str = ""
    model_parsed_keys: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "review_id": self.review_id,
            "page_no": int(self.page_no),
            "attempted": bool(self.attempted),
            "available": bool(self.available),
            "model": self.model,
            "status": self.status,
            "verdict": self.verdict,
            "decision": self.decision,
            "confidence": round(float(self.confidence or 0.0), 4),
            "reason": self.reason,
            "pdf_text_excerpt": self.pdf_text_excerpt,
            "docx_samples": [dict(item) for item in self.docx_samples],
            "suspicious_values": [dict(item) for item in self.suspicious_values],
            "next_route": self.next_route,
            "error": self.error,
            "model_parse_error": self.model_parse_error,
            "model_raw_content_excerpt": self.model_raw_content_excerpt,
            "model_parsed_keys": list(self.model_parsed_keys),
            "flags": list(self.flags),
        }


@dataclass
class PageTextCoverageProfile:
    page_no: int
    status: str
    risk_level: str
    docx_text_chars: int = 0
    pdf_text_chars: int = 0
    docx_unit_count: int = 0
    pdf_text_source_count: int = 0
    docx_token_count: int = 0
    pdf_token_count: int = 0
    docx_token_coverage_ratio: float = 0.0
    pdf_token_coverage_ratio: float = 0.0
    page_text_similarity: float = 0.0
    table_cell_ratio: float = 0.0
    reason: str = ""
    docx_gap_samples: List[str] = field(default_factory=list)
    pdf_gap_samples: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_no": int(self.page_no),
            "status": self.status,
            "risk_level": self.risk_level,
            "docx_text_chars": int(self.docx_text_chars or 0),
            "pdf_text_chars": int(self.pdf_text_chars or 0),
            "docx_unit_count": int(self.docx_unit_count or 0),
            "pdf_text_source_count": int(self.pdf_text_source_count or 0),
            "docx_token_count": int(self.docx_token_count or 0),
            "pdf_token_count": int(self.pdf_token_count or 0),
            "docx_token_coverage_ratio": round(float(self.docx_token_coverage_ratio or 0.0), 4),
            "pdf_token_coverage_ratio": round(float(self.pdf_token_coverage_ratio or 0.0), 4),
            "page_text_similarity": round(float(self.page_text_similarity or 0.0), 4),
            "table_cell_ratio": round(float(self.table_cell_ratio or 0.0), 4),
            "reason": self.reason,
            "docx_gap_samples": list(self.docx_gap_samples),
            "pdf_gap_samples": list(self.pdf_gap_samples),
            "flags": list(self.flags),
        }


@dataclass
class HighRiskPageCoverageReview:
    review_id: str
    page_no: int
    status: str
    decision: str
    risk_level: str
    priority: int
    unresolved_count: int
    docx_unresolved_count: int = 0
    pdf_unresolved_count: int = 0
    table_unresolved_count: int = 0
    mapping_uncertain_count: int = 0
    visual_unresolved_count: int = 0
    needs_ocr_count: int = 0
    backfilled_count: int = 0
    comment_anchor_unit_id: str = ""
    comment_anchor_text: str = ""
    reason: str = ""
    next_route: str = "needs_full_page_review"
    coverage_review_ids: List[str] = field(default_factory=list)
    docx_samples: List[Dict[str, Any]] = field(default_factory=list)
    pdf_samples: List[Dict[str, Any]] = field(default_factory=list)
    status_counts: Dict[str, int] = field(default_factory=dict)
    route_counts: Dict[str, int] = field(default_factory=dict)
    page_text_coverage: Dict[str, Any] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "review_id": self.review_id,
            "page_no": int(self.page_no),
            "status": self.status,
            "decision": self.decision,
            "risk_level": self.risk_level,
            "priority": int(self.priority or 0),
            "unresolved_count": int(self.unresolved_count or 0),
            "docx_unresolved_count": int(self.docx_unresolved_count or 0),
            "pdf_unresolved_count": int(self.pdf_unresolved_count or 0),
            "table_unresolved_count": int(self.table_unresolved_count or 0),
            "mapping_uncertain_count": int(self.mapping_uncertain_count or 0),
            "visual_unresolved_count": int(self.visual_unresolved_count or 0),
            "needs_ocr_count": int(self.needs_ocr_count or 0),
            "backfilled_count": int(self.backfilled_count or 0),
            "comment_anchor_unit_id": self.comment_anchor_unit_id,
            "comment_anchor_text": self.comment_anchor_text,
            "reason": self.reason,
            "next_route": self.next_route,
            "coverage_review_ids": list(self.coverage_review_ids),
            "docx_samples": [dict(item) for item in self.docx_samples],
            "pdf_samples": [dict(item) for item in self.pdf_samples],
            "status_counts": dict(self.status_counts),
            "route_counts": dict(self.route_counts),
            "page_text_coverage": dict(self.page_text_coverage),
            "flags": list(self.flags),
        }


@dataclass
class ReviewTask:
    task_id: str
    task_type: str
    page_no: Optional[int]
    priority: int
    status: str
    reason: str
    source_gap_id: str = ""
    source_payload_ref: str = ""
    target_unit_id: str = ""
    next_engine: str = ""
    planned_engine: str = ""
    budget: Dict[str, Any] = field(default_factory=dict)
    fallback: str = "human_review_required"
    evidence_refs: List[Dict[str, Any]] = field(default_factory=list)
    candidate_ids: List[str] = field(default_factory=list)
    coverage_review_ids: List[str] = field(default_factory=list)
    open_coverage_review_ids: List[str] = field(default_factory=list)
    closed_coverage_review_ids: List[str] = field(default_factory=list)
    open_coverage_count: int = 0
    resolved_coverage_count: int = 0
    executor: str = ""
    execution_outcome: str = ""
    execution_reason: str = ""
    execution_refs: List[Dict[str, Any]] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    def is_resolved(self) -> bool:
        return str(self.status or "") == "resolved"

    def is_open(self) -> bool:
        return not self.is_resolved()

    def requires_human_review(self) -> bool:
        return self.is_open() and (
            str(self.status or "") == "human_required"
            or str(self.next_engine or "") == "human_review"
            or str(self.fallback or "") == "human_review_required"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "page_no": self.page_no,
            "priority": int(self.priority or 0),
            "status": self.status,
            "reason": self.reason,
            "source_gap_id": self.source_gap_id,
            "source_payload_ref": self.source_payload_ref,
            "target_unit_id": self.target_unit_id,
            "next_engine": self.next_engine,
            "planned_engine": self.planned_engine,
            "budget": dict(self.budget),
            "fallback": self.fallback,
            "evidence_refs": [dict(item) for item in self.evidence_refs],
            "candidate_ids": list(self.candidate_ids),
            "coverage_review_ids": list(self.coverage_review_ids),
            "open_coverage_review_ids": list(self.open_coverage_review_ids),
            "closed_coverage_review_ids": list(self.closed_coverage_review_ids),
            "open_coverage_count": int(self.open_coverage_count or 0),
            "resolved_coverage_count": int(self.resolved_coverage_count or 0),
            "executor": self.executor,
            "execution_outcome": self.execution_outcome,
            "execution_reason": self.execution_reason,
            "execution_refs": [dict(item) for item in self.execution_refs],
            "flags": list(self.flags),
        }


@dataclass
class SpecialistReviewTask:
    task_id: str
    task_type: str
    page_no: Optional[int]
    priority: int
    status: str
    reason: str
    next_route: str
    model_hint: str = ""
    comment_policy: str = "report_only_until_confirmed"
    candidate_ids: List[str] = field(default_factory=list)
    coverage_review_ids: List[str] = field(default_factory=list)
    backfill_ids: List[str] = field(default_factory=list)
    review_ids: List[str] = field(default_factory=list)
    evidence_refs: List[Dict[str, Any]] = field(default_factory=list)
    source_counts: Dict[str, int] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "page_no": self.page_no,
            "priority": int(self.priority or 0),
            "status": self.status,
            "reason": self.reason,
            "next_route": self.next_route,
            "model_hint": self.model_hint,
            "comment_policy": self.comment_policy,
            "candidate_ids": list(self.candidate_ids),
            "coverage_review_ids": list(self.coverage_review_ids),
            "backfill_ids": list(self.backfill_ids),
            "review_ids": list(self.review_ids),
            "evidence_refs": [dict(item) for item in self.evidence_refs],
            "source_counts": dict(self.source_counts),
            "flags": list(self.flags),
        }


@dataclass
class SpecialistReviewResult:
    result_id: str
    task_id: str
    task_type: str
    page_no: Optional[int]
    status: str
    decision: str
    confidence: float
    reason: str
    issue_type: str = ""
    wps_unit_id: str = ""
    old_text: str = ""
    new_text: str = ""
    next_route: str = ""
    model: str = ""
    comment_policy: str = "report_only_until_confirmed"
    evidence_refs: List[Dict[str, Any]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result_id": self.result_id,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "page_no": self.page_no,
            "status": self.status,
            "decision": self.decision,
            "confidence": round(float(self.confidence or 0.0), 4),
            "reason": self.reason,
            "issue_type": self.issue_type,
            "wps_unit_id": self.wps_unit_id,
            "old_text": self.old_text,
            "new_text": self.new_text,
            "next_route": self.next_route,
            "model": self.model,
            "comment_policy": self.comment_policy,
            "evidence_refs": [dict(item) for item in self.evidence_refs],
            "context": dict(self.context),
            "flags": list(self.flags),
        }


@dataclass
class ConversionPreflightResult:
    pdf_page_count: int
    pdf_units: List[PdfEvidenceUnit]
    docx_units: List[DocxEvidenceUnit]
    alignment_links: List[AlignmentLink]
    diff_candidates: List[ConversionDiffCandidate]
    review_routes: List[ReviewRoute]
    page_profiles: Dict[str, Dict[str, Any]]
    warnings: List[str] = field(default_factory=list)
    page_orientation: Dict[str, Any] = field(default_factory=dict)
    anchor_ocr: Dict[str, Any] = field(default_factory=dict)
    docx_page_remap: Dict[str, Any] = field(default_factory=dict)
    focused_reviews: List[FocusedCandidateReview] = field(default_factory=list)
    table_reviews: List[TableParseReview] = field(default_factory=list)
    content_coverage_reviews: List[ContentCoverageReview] = field(default_factory=list)
    content_coverage_backfills: List[ContentCoverageBackfillReview] = field(default_factory=list)
    table_page_vl_reviews: List[TablePageVlReview] = field(default_factory=list)
    table_cell_evidence_reviews: List[TableCellEvidenceReview] = field(default_factory=list)
    table_grid_evidence: List[TableGridEvidence] = field(default_factory=list)
    fragment_anomaly_reviews: List[FragmentAnomalyReview] = field(default_factory=list)
    image_page_reviews: List[ImagePdfPageReview] = field(default_factory=list)
    image_text_vl_reviews: List[ImagePageVlTextReview] = field(default_factory=list)
    page_ocr_text_evidence_reviews: List[PageOcrTextEvidenceReview] = field(default_factory=list)
    page_text_qwen_reviews: List[PageTextQwenReview] = field(default_factory=list)
    page_text_coverage_profiles: List[PageTextCoverageProfile] = field(default_factory=list)
    high_risk_page_coverage_reviews: List[HighRiskPageCoverageReview] = field(default_factory=list)
    coverage_review_tasks: List[ReviewTask] = field(default_factory=list)
    qwen_vl_reviews: List[QwenVlGateReview] = field(default_factory=list)
    qwen_gate_reviews: List[QwenGateReview] = field(default_factory=list)
    specialist_review_tasks: List[SpecialistReviewTask] = field(default_factory=list)
    specialist_review_results: List[SpecialistReviewResult] = field(default_factory=list)
    table_audit_summary: Dict[str, Any] = field(default_factory=dict)
    quality_inspection: Dict[str, Any] = field(default_factory=dict)
    audit_route_plan: Dict[str, Any] = field(default_factory=dict)
    content_coverage_reconciliation: Dict[str, Any] = field(default_factory=dict)
    mapping_stabilization: Dict[str, Any] = field(default_factory=dict)
    model_output_guard: Dict[str, Any] = field(default_factory=dict)

    def qwen_vl_candidate_count(self) -> int:
        return sum(
            1
            for item in self.focused_reviews
            if item.crop_path
            and (
                item.next_route == "needs_qwen_vl"
                or item.decision == "needs_visual_review"
                or item.status == "recall_guard"
            )
            and "crop_ocr_supports_docx" not in set(item.flags)
        )

    def summary(self) -> Dict[str, Any]:
        matched = sum(1 for item in self.alignment_links if item.status == "matched")
        uncertain = sum(1 for item in self.alignment_links if item.status == "mapping_uncertain")
        unmatched_pdf = sum(1 for item in self.diff_candidates if item.category == "missing_content")
        unmatched_docx = sum(1 for item in self.diff_candidates if item.category == "extra_content")
        actionable_categories = {
            "critical_field_changed",
            "text_substitution",
            "table_cell_mismatch_suspect",
            "missing_content",
            "extra_content",
            "duplicate_content",
            "reading_order_changed",
            "unlocated_hard_field",
            "page_coverage_gap",
        }
        category_counts: Dict[str, int] = {}
        for diff in self.diff_candidates:
            category_counts[diff.category] = category_counts.get(diff.category, 0) + 1
        route_counts: Dict[str, int] = {}
        for route in self.review_routes:
            route_counts[route.route] = route_counts.get(route.route, 0) + 1
        focused_status_counts: Dict[str, int] = {}
        focused_decision_counts: Dict[str, int] = {}
        focused_next_route_counts: Dict[str, int] = {}
        no_confirmed_error_count = 0
        for review in self.focused_reviews:
            focused_status_counts[review.status] = focused_status_counts.get(review.status, 0) + 1
            focused_decision_counts[review.decision] = focused_decision_counts.get(review.decision, 0) + 1
            if review.decision == "no_confirmed_error":
                no_confirmed_error_count += 1
            elif review.next_route:
                focused_next_route_counts[review.next_route] = focused_next_route_counts.get(review.next_route, 0) + 1
        qwen_verdict_counts: Dict[str, int] = {}
        qwen_decision_counts: Dict[str, int] = {}
        qwen_next_route_counts: Dict[str, int] = {}
        for review in self.qwen_gate_reviews:
            verdict = review.verdict or "not_available"
            decision = review.decision or "not_available"
            qwen_verdict_counts[verdict] = qwen_verdict_counts.get(verdict, 0) + 1
            qwen_decision_counts[decision] = qwen_decision_counts.get(decision, 0) + 1
            if review.next_route:
                qwen_next_route_counts[review.next_route] = qwen_next_route_counts.get(review.next_route, 0) + 1
        qwen_vl_candidate_count = self.qwen_vl_candidate_count()
        qwen_vl_verdict_counts: Dict[str, int] = {}
        qwen_vl_decision_counts: Dict[str, int] = {}
        qwen_vl_next_route_counts: Dict[str, int] = {}
        for review in self.qwen_vl_reviews:
            verdict = review.verdict or "not_available"
            decision = review.decision or "not_available"
            qwen_vl_verdict_counts[verdict] = qwen_vl_verdict_counts.get(verdict, 0) + 1
            qwen_vl_decision_counts[decision] = qwen_vl_decision_counts.get(decision, 0) + 1
            if review.next_route:
                qwen_vl_next_route_counts[review.next_route] = qwen_vl_next_route_counts.get(review.next_route, 0) + 1
        coverage_status_counts: Dict[str, int] = {}
        coverage_decision_counts: Dict[str, int] = {}
        coverage_next_route_counts: Dict[str, int] = {}
        for review in self.content_coverage_reviews:
            coverage_status_counts[review.status] = coverage_status_counts.get(review.status, 0) + 1
            coverage_decision_counts[review.decision] = coverage_decision_counts.get(review.decision, 0) + 1
            if review.next_route:
                coverage_next_route_counts[review.next_route] = coverage_next_route_counts.get(review.next_route, 0) + 1
        backfill_status_counts: Dict[str, int] = {}
        backfill_method_counts: Dict[str, int] = {}
        backfill_next_route_counts: Dict[str, int] = {}
        for review in self.content_coverage_backfills:
            status = review.status or "unknown"
            method = review.method or "unknown"
            backfill_status_counts[status] = backfill_status_counts.get(status, 0) + 1
            backfill_method_counts[method] = backfill_method_counts.get(method, 0) + 1
            if review.next_route:
                backfill_next_route_counts[review.next_route] = backfill_next_route_counts.get(review.next_route, 0) + 1
        table_vl_verdict_counts: Dict[str, int] = {}
        for review in self.table_page_vl_reviews:
            verdict = review.verdict or "not_available"
            table_vl_verdict_counts[verdict] = table_vl_verdict_counts.get(verdict, 0) + 1
        table_cell_status_counts: Dict[str, int] = {}
        table_cell_decision_counts: Dict[str, int] = {}
        table_cell_issue_counts: Dict[str, int] = {}
        for review in self.table_cell_evidence_reviews:
            table_cell_status_counts[review.status] = table_cell_status_counts.get(review.status, 0) + 1
            table_cell_decision_counts[review.decision] = table_cell_decision_counts.get(review.decision, 0) + 1
            issue = review.issue_type or "unknown"
            table_cell_issue_counts[issue] = table_cell_issue_counts.get(issue, 0) + 1
        table_grid_status_counts: Dict[str, int] = {}
        table_grid_anomaly_count = 0
        for grid in self.table_grid_evidence:
            table_grid_status_counts[grid.status] = table_grid_status_counts.get(grid.status, 0) + 1
            table_grid_anomaly_count += len(grid.anomaly_cells)
        fragment_verdict_counts: Dict[str, int] = {}
        fragment_pages: set[int] = set()
        for review in self.fragment_anomaly_reviews:
            verdict = review.verdict or "not_available"
            fragment_verdict_counts[verdict] = fragment_verdict_counts.get(verdict, 0) + 1
            fragment_pages.add(int(review.page_no or 0))
        image_page_verdict_counts: Dict[str, int] = {}
        image_page_risk_counts: Dict[str, int] = {}
        for review in self.image_page_reviews:
            verdict = review.verdict or "not_available"
            risk = review.risk_level or "unknown"
            image_page_verdict_counts[verdict] = image_page_verdict_counts.get(verdict, 0) + 1
            image_page_risk_counts[risk] = image_page_risk_counts.get(risk, 0) + 1
        image_text_vl_verdict_counts: Dict[str, int] = {}
        image_text_vl_decision_counts: Dict[str, int] = {}
        for review in self.image_text_vl_reviews:
            verdict = review.verdict or "not_available"
            decision = review.decision or "not_available"
            image_text_vl_verdict_counts[verdict] = image_text_vl_verdict_counts.get(verdict, 0) + 1
            image_text_vl_decision_counts[decision] = image_text_vl_decision_counts.get(decision, 0) + 1
        page_ocr_text_status_counts: Dict[str, int] = {}
        page_ocr_text_decision_counts: Dict[str, int] = {}
        page_ocr_text_issue_counts: Dict[str, int] = {}
        for review in self.page_ocr_text_evidence_reviews:
            page_ocr_text_status_counts[review.status] = page_ocr_text_status_counts.get(review.status, 0) + 1
            page_ocr_text_decision_counts[review.decision] = page_ocr_text_decision_counts.get(review.decision, 0) + 1
            issue = review.issue_type or "unknown"
            page_ocr_text_issue_counts[issue] = page_ocr_text_issue_counts.get(issue, 0) + 1
        page_text_qwen_status_counts: Dict[str, int] = {}
        page_text_qwen_decision_counts: Dict[str, int] = {}
        page_text_qwen_verdict_counts: Dict[str, int] = {}
        for review in self.page_text_qwen_reviews:
            page_text_qwen_status_counts[review.status or "unknown"] = page_text_qwen_status_counts.get(review.status or "unknown", 0) + 1
            page_text_qwen_decision_counts[review.decision or "unknown"] = page_text_qwen_decision_counts.get(review.decision or "unknown", 0) + 1
            page_text_qwen_verdict_counts[review.verdict or "unknown"] = page_text_qwen_verdict_counts.get(review.verdict or "unknown", 0) + 1
        page_text_coverage_status_counts: Dict[str, int] = {}
        page_text_coverage_risk_counts: Dict[str, int] = {}
        for profile in self.page_text_coverage_profiles:
            page_text_coverage_status_counts[profile.status] = page_text_coverage_status_counts.get(profile.status, 0) + 1
            page_text_coverage_risk_counts[profile.risk_level] = page_text_coverage_risk_counts.get(profile.risk_level, 0) + 1
        high_risk_page_status_counts: Dict[str, int] = {}
        high_risk_page_decision_counts: Dict[str, int] = {}
        high_risk_page_route_counts: Dict[str, int] = {}
        for review in self.high_risk_page_coverage_reviews:
            high_risk_page_status_counts[review.status] = high_risk_page_status_counts.get(review.status, 0) + 1
            high_risk_page_decision_counts[review.decision] = high_risk_page_decision_counts.get(review.decision, 0) + 1
            if review.next_route:
                high_risk_page_route_counts[review.next_route] = high_risk_page_route_counts.get(review.next_route, 0) + 1
        review_task_type_counts: Dict[str, int] = {}
        review_task_status_counts: Dict[str, int] = {}
        review_task_engine_counts: Dict[str, int] = {}
        review_task_planned_engine_counts: Dict[str, int] = {}
        review_task_executor_counts: Dict[str, int] = {}
        review_task_execution_outcome_counts: Dict[str, int] = {}
        review_task_pages: set[int] = set()
        open_review_task_pages: set[int] = set()
        resolved_review_task_pages: set[int] = set()
        open_review_task_count = 0
        resolved_review_task_count = 0
        for task in self.coverage_review_tasks:
            review_task_type_counts[task.task_type] = review_task_type_counts.get(task.task_type, 0) + 1
            review_task_status_counts[task.status] = review_task_status_counts.get(task.status, 0) + 1
            if task.next_engine:
                review_task_engine_counts[task.next_engine] = review_task_engine_counts.get(task.next_engine, 0) + 1
            planned_engine = str(task.planned_engine or "")
            if planned_engine:
                review_task_planned_engine_counts[planned_engine] = review_task_planned_engine_counts.get(planned_engine, 0) + 1
            executor = str(task.executor or "")
            if executor:
                review_task_executor_counts[executor] = review_task_executor_counts.get(executor, 0) + 1
            execution_outcome = str(task.execution_outcome or "")
            if execution_outcome:
                review_task_execution_outcome_counts[execution_outcome] = (
                    review_task_execution_outcome_counts.get(execution_outcome, 0) + 1
                )
            if task.page_no:
                review_task_pages.add(int(task.page_no))
                if task.is_resolved():
                    resolved_review_task_pages.add(int(task.page_no))
                elif task.requires_human_review():
                    open_review_task_pages.add(int(task.page_no))
            if task.is_resolved():
                resolved_review_task_count += 1
            elif task.requires_human_review():
                open_review_task_count += 1
        table_status_counts: Dict[str, int] = {}
        table_related_diff_ids: set[str] = set()
        for review in self.table_reviews:
            table_status_counts[review.status] = table_status_counts.get(review.status, 0) + 1
            table_related_diff_ids.update(review.related_diff_ids)
        specialist_type_counts: Dict[str, int] = {}
        specialist_status_counts: Dict[str, int] = {}
        specialist_route_counts: Dict[str, int] = {}
        specialist_pages: set[int] = set()
        for task in self.specialist_review_tasks:
            specialist_type_counts[task.task_type] = specialist_type_counts.get(task.task_type, 0) + 1
            specialist_status_counts[task.status] = specialist_status_counts.get(task.status, 0) + 1
            if task.next_route:
                specialist_route_counts[task.next_route] = specialist_route_counts.get(task.next_route, 0) + 1
            if task.page_no:
                specialist_pages.add(int(task.page_no))
        specialist_result_status_counts: Dict[str, int] = {}
        specialist_result_decision_counts: Dict[str, int] = {}
        specialist_result_type_counts: Dict[str, int] = {}
        for result in self.specialist_review_results:
            specialist_result_status_counts[result.status] = specialist_result_status_counts.get(result.status, 0) + 1
            specialist_result_decision_counts[result.decision] = specialist_result_decision_counts.get(result.decision, 0) + 1
            specialist_result_type_counts[result.task_type] = specialist_result_type_counts.get(result.task_type, 0) + 1
        return {
            "enabled": True,
            "version": "conversion_preflight_v1",
            "mode": "report_only",
            "pdf_page_count": int(self.pdf_page_count),
            "docx_unit_count": len(self.docx_units),
            "pdf_region_count": sum(1 for item in self.pdf_units if "region" in item.unit_type),
            "pdf_unit_count": len(self.pdf_units),
            "alignment": {
                "matched_count": matched,
                "uncertain_count": uncertain,
                "unmatched_pdf_count": unmatched_pdf,
                "unmatched_docx_count": unmatched_docx,
            },
            "diff_candidate_count": len(self.diff_candidates),
            "diff_category_counts": dict(sorted(category_counts.items())),
            "actionable_diff_count": sum(1 for item in self.diff_candidates if item.category in actionable_categories),
            "mapping_uncertain_page_count": sum(1 for item in self.diff_candidates if item.category == "mapping_uncertain"),
            "table_suspect_page_count": sum(1 for item in self.diff_candidates if item.category == "table_structure_suspect"),
            "recall_guard_candidate_count": sum(1 for item in self.diff_candidates if "recall_guard" in set(item.flags)),
            "unlocated_hard_field_count": sum(1 for item in self.diff_candidates if item.category == "unlocated_hard_field"),
            "page_coverage_gap_count": sum(1 for item in self.diff_candidates if item.category == "page_coverage_gap"),
            "route_counts": dict(sorted(route_counts.items())),
            "safe_to_generate_findings": False,
            "page_profiles": self.page_profiles,
            "audit_route_plan": dict((self.audit_route_plan or {}).get("summary") or {}),
            "page_orientation": dict((self.page_orientation or {}).get("summary") or self.page_orientation or {}),
            "anchor_ocr": dict(self.anchor_ocr.get("summary") or self.anchor_ocr or {}),
            "docx_page_remap": dict((self.docx_page_remap or {}).get("summary") or self.docx_page_remap or {}),
            "mapping_stabilization": dict(
                self.mapping_stabilization
                or {
                    "enabled": False,
                    "version": "mapping_stabilizer_v1",
                }
            ),
            "model_output_guard": dict(
                self.model_output_guard
                or {
                    "enabled": False,
                    "version": "model_output_guard_v1",
                }
            ),
            "focused_review": {
                "enabled": bool(self.focused_reviews),
                "version": "focused_review_v1",
                "reviewed_candidate_count": len(self.focused_reviews),
                "status_counts": dict(sorted(focused_status_counts.items())),
                "decision_counts": dict(sorted(focused_decision_counts.items())),
                "post_gate_route_counts": dict(sorted(focused_next_route_counts.items())),
                "no_confirmed_error_count": no_confirmed_error_count,
                "confirmable_candidate_count": sum(1 for item in self.focused_reviews if item.decision == "possible_conversion_error"),
                "blocked_or_deferred_count": sum(1 for item in self.focused_reviews if item.decision != "possible_conversion_error"),
            },
            "table_review": {
                "enabled": bool(self.table_reviews),
                "version": "table_parse_review_v1",
                "reviewed_page_count": len(self.table_reviews),
                "parseable_table_count": sum(1 for item in self.table_reviews if item.status == "parseable_grid"),
                "related_candidate_count": len(table_related_diff_ids),
                "status_counts": dict(sorted(table_status_counts.items())),
                "ready_cell_candidate_count": sum(1 for item in self.focused_reviews if item.status == "ready_for_table_gate"),
                "stable_cell_evidence_count": sum(
                    1
                    for item in self.focused_reviews
                    if bool((item.table_cell or {}).get("stable_text_match"))
                ),
            },
            "full_content_coverage": {
                "enabled": bool(self.content_coverage_reviews),
                "version": "full_content_coverage_v1",
                "reviewed_unit_count": len(self.content_coverage_reviews),
                "docx_reviewed_unit_count": sum(1 for item in self.content_coverage_reviews if item.side == "docx"),
                "pdf_reviewed_unit_count": sum(1 for item in self.content_coverage_reviews if item.side == "pdf"),
                "covered_count": sum(1 for item in self.content_coverage_reviews if item.decision == "covered"),
                "unresolved_count": sum(1 for item in self.content_coverage_reviews if item.decision != "covered"),
                "backfilled_count": sum(1 for item in self.content_coverage_backfills if item.available),
                "status_counts": dict(sorted(coverage_status_counts.items())),
                "decision_counts": dict(sorted(coverage_decision_counts.items())),
                "next_route_counts": dict(sorted(coverage_next_route_counts.items())),
                "reconciliation": dict(self.content_coverage_reconciliation or {}),
            },
            "full_content_backfill": {
                "enabled": bool(self.content_coverage_backfills),
                "version": "full_content_backfill_v1",
                "reviewed_unit_count": len(self.content_coverage_backfills),
                "attempted_count": sum(1 for item in self.content_coverage_backfills if item.attempted),
                "available_count": sum(1 for item in self.content_coverage_backfills if item.available),
                "crop_count": sum(1 for item in self.content_coverage_backfills if item.crop_path),
                "macos_vision_count": sum(1 for item in self.content_coverage_backfills if item.method == "macos_vision"),
                "qwen_vl_count": sum(1 for item in self.content_coverage_backfills if item.method == "qwen_vl"),
                "text_chars": sum(len(item.normalized_text or "") for item in self.content_coverage_backfills),
                "status_counts": dict(sorted(backfill_status_counts.items())),
                "method_counts": dict(sorted(backfill_method_counts.items())),
                "next_route_counts": dict(sorted(backfill_next_route_counts.items())),
            },
            "qwen_vl_gate": {
                "enabled": bool(self.qwen_vl_reviews or qwen_vl_candidate_count),
                "version": "qwen_vl_gate_v1",
                "candidate_count": qwen_vl_candidate_count,
                "reviewed_candidate_count": len(self.qwen_vl_reviews),
                "skipped_candidate_count": max(0, qwen_vl_candidate_count - len(self.qwen_vl_reviews)),
                "available_count": sum(1 for item in self.qwen_vl_reviews if item.available),
                "verdict_counts": dict(sorted(qwen_vl_verdict_counts.items())),
                "decision_counts": dict(sorted(qwen_vl_decision_counts.items())),
                "next_route_counts": dict(sorted(qwen_vl_next_route_counts.items())),
                "allow_report_candidate_count": sum(1 for item in self.qwen_vl_reviews if item.decision == "allow_report_candidate"),
                "blocked_or_deferred_count": sum(1 for item in self.qwen_vl_reviews if item.decision != "allow_report_candidate"),
            },
            "table_page_vl": {
                "enabled": bool(self.table_page_vl_reviews),
                "version": "table_page_vl_v1",
                "reviewed_page_count": len(self.table_page_vl_reviews),
                "available_count": sum(1 for item in self.table_page_vl_reviews if item.available),
                "verdict_counts": dict(sorted(table_vl_verdict_counts.items())),
                "suspicious_value_count": sum(len(item.suspicious_values) for item in self.table_page_vl_reviews),
            },
            "table_cell_evidence": {
                "enabled": bool(self.table_cell_evidence_reviews),
                "version": "table_cell_evidence_v1",
                "reviewed_cell_count": len(self.table_cell_evidence_reviews),
                "confirmable_cell_count": sum(1 for item in self.table_cell_evidence_reviews if item.decision == "confirmed_error"),
                "status_counts": dict(sorted(table_cell_status_counts.items())),
                "decision_counts": dict(sorted(table_cell_decision_counts.items())),
                "issue_counts": dict(sorted(table_cell_issue_counts.items())),
            },
            "table_grid_evidence": {
                "enabled": bool(self.table_grid_evidence),
                "version": "table_grid_evidence_v1",
                "grid_count": len(self.table_grid_evidence),
                "anomaly_cell_count": table_grid_anomaly_count,
                "confirmed_error_count": sum(int(item.confirmed_error_count or 0) for item in self.table_grid_evidence),
                "suspected_error_count": sum(int(item.suspected_error_count or 0) for item in self.table_grid_evidence),
                "unresolved_cell_count": sum(int(item.unresolved_cell_count or 0) for item in self.table_grid_evidence),
                "status_counts": dict(sorted(table_grid_status_counts.items())),
            },
            "fragment_anomaly": {
                "enabled": bool(self.fragment_anomaly_reviews),
                "version": "fragment_anomaly_v1",
                "reviewed_page_count": len(self.fragment_anomaly_reviews),
                "commentable_count": sum(1 for item in self.fragment_anomaly_reviews if item.anchor_unit_id),
                "risk_page_count": len({page for page in fragment_pages if page > 0}),
                "fragment_count": sum(int(item.fragment_count or 0) for item in self.fragment_anomaly_reviews),
                "repeated_fragment_count": sum(int(item.repeated_fragment_count or 0) for item in self.fragment_anomaly_reviews),
                "short_fragment_count": sum(int(item.short_fragment_count or 0) for item in self.fragment_anomaly_reviews),
                "verdict_counts": dict(sorted(fragment_verdict_counts.items())),
            },
            "image_pdf_page_review": {
                "enabled": bool(self.image_page_reviews),
                "version": "image_pdf_page_review_v1",
                "reviewed_page_count": len(self.image_page_reviews),
                "high_risk_page_count": sum(1 for item in self.image_page_reviews if item.risk_level == "high"),
                "medium_risk_page_count": sum(1 for item in self.image_page_reviews if item.risk_level == "medium"),
                "commentable_count": sum(1 for item in self.image_page_reviews if item.anchor_unit_id and item.risk_level in {"high", "medium"}),
                "unresolved_count": sum(int(item.unresolved_count or 0) for item in self.image_page_reviews),
                "low_ocr_page_count": sum(1 for item in self.image_page_reviews if item.ocr_quality in {"low", "failed", ""}),
                "verdict_counts": dict(sorted(image_page_verdict_counts.items())),
                "risk_counts": dict(sorted(image_page_risk_counts.items())),
            },
            "image_text_vl": {
                "enabled": bool(self.image_text_vl_reviews),
                "version": "image_text_vl_v1",
                "reviewed_page_count": len(self.image_text_vl_reviews),
                "available_count": sum(1 for item in self.image_text_vl_reviews if item.available),
                "suspicious_value_count": sum(len(item.suspicious_values) for item in self.image_text_vl_reviews),
                "verdict_counts": dict(sorted(image_text_vl_verdict_counts.items())),
                "decision_counts": dict(sorted(image_text_vl_decision_counts.items())),
            },
            "page_ocr_text_evidence": {
                "enabled": bool(self.page_ocr_text_evidence_reviews),
                "version": "page_ocr_text_evidence_v1",
                "reviewed_unit_count": len(self.page_ocr_text_evidence_reviews),
                "confirmable_unit_count": sum(1 for item in self.page_ocr_text_evidence_reviews if item.decision == "confirmed_error"),
                "status_counts": dict(sorted(page_ocr_text_status_counts.items())),
                "decision_counts": dict(sorted(page_ocr_text_decision_counts.items())),
                "issue_counts": dict(sorted(page_ocr_text_issue_counts.items())),
            },
            "page_text_qwen_review": {
                "enabled": bool(self.page_text_qwen_reviews),
                "version": "page_text_qwen_review_v1",
                "reviewed_page_count": len(self.page_text_qwen_reviews),
                "available_page_count": sum(1 for item in self.page_text_qwen_reviews if item.available),
                "exact_replacement_count": sum(len(item.suspicious_values) for item in self.page_text_qwen_reviews if item.decision == "allow_exact_replacements"),
                "status_counts": dict(sorted(page_text_qwen_status_counts.items())),
                "decision_counts": dict(sorted(page_text_qwen_decision_counts.items())),
                "verdict_counts": dict(sorted(page_text_qwen_verdict_counts.items())),
            },
            "page_text_coverage": {
                "enabled": bool(self.page_text_coverage_profiles),
                "version": "page_text_coverage_v1",
                "reviewed_page_count": len(self.page_text_coverage_profiles),
                "high_risk_page_count": sum(1 for item in self.page_text_coverage_profiles if item.risk_level == "high"),
                "medium_risk_page_count": sum(1 for item in self.page_text_coverage_profiles if item.risk_level == "medium"),
                "low_docx_coverage_page_count": sum(1 for item in self.page_text_coverage_profiles if float(item.docx_token_coverage_ratio or 0.0) < 0.72),
                "low_pdf_coverage_page_count": sum(1 for item in self.page_text_coverage_profiles if float(item.pdf_token_coverage_ratio or 0.0) < 0.72),
                "avg_docx_token_coverage_ratio": round(
                    sum(float(item.docx_token_coverage_ratio or 0.0) for item in self.page_text_coverage_profiles)
                    / max(1, len(self.page_text_coverage_profiles)),
                    4,
                ),
                "avg_pdf_token_coverage_ratio": round(
                    sum(float(item.pdf_token_coverage_ratio or 0.0) for item in self.page_text_coverage_profiles)
                    / max(1, len(self.page_text_coverage_profiles)),
                    4,
                ),
                "status_counts": dict(sorted(page_text_coverage_status_counts.items())),
                "risk_counts": dict(sorted(page_text_coverage_risk_counts.items())),
            },
            "high_risk_page_coverage": {
                "enabled": bool(self.high_risk_page_coverage_reviews),
                "version": "high_risk_page_coverage_v1",
                "reviewed_page_count": len(self.high_risk_page_coverage_reviews),
                "high_risk_page_count": sum(1 for item in self.high_risk_page_coverage_reviews if item.risk_level == "high"),
                "total_unresolved_count": sum(int(item.unresolved_count or 0) for item in self.high_risk_page_coverage_reviews),
                "commentable_page_count": sum(1 for item in self.high_risk_page_coverage_reviews if item.comment_anchor_unit_id),
                "status_counts": dict(sorted(high_risk_page_status_counts.items())),
                "decision_counts": dict(sorted(high_risk_page_decision_counts.items())),
                "next_route_counts": dict(sorted(high_risk_page_route_counts.items())),
            },
            "coverage_review_tasks": {
                "enabled": bool(self.coverage_review_tasks),
                "version": "coverage_gap_scheduler_v1",
                "task_count": len(self.coverage_review_tasks),
                "page_count": len(review_task_pages),
                "open_task_count": open_review_task_count,
                "resolved_task_count": resolved_review_task_count,
                "open_page_count": len(open_review_task_pages),
                "resolved_page_count": len(resolved_review_task_pages),
                "high_priority_task_count": sum(1 for item in self.coverage_review_tasks if int(item.priority or 0) >= 85),
                "human_required_count": sum(1 for item in self.coverage_review_tasks if item.requires_human_review()),
                "type_counts": dict(sorted(review_task_type_counts.items())),
                "status_counts": dict(sorted(review_task_status_counts.items())),
                "next_engine_counts": dict(sorted(review_task_engine_counts.items())),
                "planned_engine_counts": dict(sorted(review_task_planned_engine_counts.items())),
                "executor_counts": dict(sorted(review_task_executor_counts.items())),
                "execution_outcome_counts": dict(sorted(review_task_execution_outcome_counts.items())),
            },
            "qwen_gate": {
                "enabled": bool(self.qwen_gate_reviews),
                "version": "focused_qwen_gate_v1",
                "reviewed_candidate_count": len(self.qwen_gate_reviews),
                "available_count": sum(1 for item in self.qwen_gate_reviews if item.available),
                "verdict_counts": dict(sorted(qwen_verdict_counts.items())),
                "decision_counts": dict(sorted(qwen_decision_counts.items())),
                "next_route_counts": dict(sorted(qwen_next_route_counts.items())),
                "allow_report_candidate_count": sum(1 for item in self.qwen_gate_reviews if item.decision == "allow_report_candidate"),
                "blocked_or_deferred_count": sum(1 for item in self.qwen_gate_reviews if item.decision != "allow_report_candidate"),
            },
            "specialist_review_plan": {
                "enabled": bool(self.specialist_review_tasks),
                "version": "specialist_review_plan_v1",
                "task_count": len(self.specialist_review_tasks),
                "page_count": len(specialist_pages),
                "high_priority_task_count": sum(1 for item in self.specialist_review_tasks if int(item.priority or 0) >= 85),
                "type_counts": dict(sorted(specialist_type_counts.items())),
                "status_counts": dict(sorted(specialist_status_counts.items())),
                "next_route_counts": dict(sorted(specialist_route_counts.items())),
            },
            "specialist_review_results": {
                "enabled": bool(self.specialist_review_results),
                "version": "specialist_review_results_v1",
                "executed_task_count": len({item.task_id for item in self.specialist_review_results if item.task_id}),
                "result_count": len(self.specialist_review_results),
                "confirmed_error_count": sum(1 for item in self.specialist_review_results if item.decision == "confirmed_error"),
                "comment_promoted_count": sum(
                    1
                    for item in self.specialist_review_results
                    if item.decision == "confirmed_error"
                    and item.comment_policy == "comment_if_exact_replacement"
                    and bool(item.wps_unit_id and item.old_text and item.new_text)
                ),
                "deferred_count": sum(1 for item in self.specialist_review_results if item.decision != "confirmed_error"),
                "type_counts": dict(sorted(specialist_result_type_counts.items())),
                "status_counts": dict(sorted(specialist_result_status_counts.items())),
                "decision_counts": dict(sorted(specialist_result_decision_counts.items())),
            },
            "table_audit_summary": dict(
                self.table_audit_summary
                or {
                    "enabled": False,
                    "version": "table_audit_engine_v1",
                }
            ),
            "quality_inspection": dict(self.quality_inspection.get("summary") or self.quality_inspection or {}),
            "warnings": list(self.warnings),
        }

    def raw_payload(self) -> Dict[str, Any]:
        qwen_vl_candidate_count = self.qwen_vl_candidate_count()
        return {
            "enabled": True,
            "version": "conversion_preflight_v1",
            "mode": "report_only",
            "summary": self.summary(),
            "page_profiles": dict(self.page_profiles),
            "audit_route_plan": dict(self.audit_route_plan or {}),
            "page_orientation": dict(self.page_orientation or {}),
            "anchor_ocr": dict(self.anchor_ocr),
            "docx_page_remap": dict(self.docx_page_remap or {}),
            "mapping_stabilization": dict(
                self.mapping_stabilization
                or {
                    "enabled": False,
                    "version": "mapping_stabilizer_v1",
                }
            ),
            "focused_review": {
                "enabled": bool(self.focused_reviews),
                "version": "focused_review_v1",
                "reviews": [item.to_dict() for item in self.focused_reviews],
            },
            "table_review": {
                "enabled": bool(self.table_reviews),
                "version": "table_parse_review_v1",
                "reviews": [item.to_dict() for item in self.table_reviews],
            },
            "full_content_coverage": {
                "enabled": bool(self.content_coverage_reviews),
                "version": "full_content_coverage_v1",
                "reviews": [item.to_dict() for item in self.content_coverage_reviews],
            },
            "full_content_backfill": {
                "enabled": bool(self.content_coverage_backfills),
                "version": "full_content_backfill_v1",
                "reviews": [item.to_dict() for item in self.content_coverage_backfills],
            },
            "full_content_reconciliation": dict(
                self.content_coverage_reconciliation
                or {
                    "enabled": False,
                    "version": "content_coverage_backfill_reconciliation_v1",
                }
            ),
            "qwen_vl_gate": {
                "enabled": bool(self.qwen_vl_reviews or qwen_vl_candidate_count),
                "version": "qwen_vl_gate_v1",
                "candidate_count": qwen_vl_candidate_count,
                "skipped_candidate_count": max(0, qwen_vl_candidate_count - len(self.qwen_vl_reviews)),
                "reviews": [item.to_dict() for item in self.qwen_vl_reviews],
            },
            "table_page_vl": {
                "enabled": bool(self.table_page_vl_reviews),
                "version": "table_page_vl_v1",
                "reviews": [item.to_dict() for item in self.table_page_vl_reviews],
            },
            "table_cell_evidence": {
                "enabled": bool(self.table_cell_evidence_reviews),
                "version": "table_cell_evidence_v1",
                "reviews": [item.to_dict() for item in self.table_cell_evidence_reviews],
            },
            "table_grid_evidence": {
                "enabled": bool(self.table_grid_evidence),
                "version": "table_grid_evidence_v1",
                "grids": [item.to_dict() for item in self.table_grid_evidence],
            },
            "fragment_anomaly": {
                "enabled": bool(self.fragment_anomaly_reviews),
                "version": "fragment_anomaly_v1",
                "reviews": [item.to_dict() for item in self.fragment_anomaly_reviews],
            },
            "image_pdf_page_review": {
                "enabled": bool(self.image_page_reviews),
                "version": "image_pdf_page_review_v1",
                "reviews": [item.to_dict() for item in self.image_page_reviews],
            },
            "image_text_vl": {
                "enabled": bool(self.image_text_vl_reviews),
                "version": "image_text_vl_v1",
                "reviews": [item.to_dict() for item in self.image_text_vl_reviews],
            },
            "page_ocr_text_evidence": {
                "enabled": bool(self.page_ocr_text_evidence_reviews),
                "version": "page_ocr_text_evidence_v1",
                "reviews": [item.to_dict() for item in self.page_ocr_text_evidence_reviews],
            },
            "page_text_qwen_review": {
                "enabled": bool(self.page_text_qwen_reviews),
                "version": "page_text_qwen_review_v1",
                "reviews": [item.to_dict() for item in self.page_text_qwen_reviews],
            },
            "page_text_coverage": {
                "enabled": bool(self.page_text_coverage_profiles),
                "version": "page_text_coverage_v1",
                "profiles": [item.to_dict() for item in self.page_text_coverage_profiles],
            },
            "high_risk_page_coverage": {
                "enabled": bool(self.high_risk_page_coverage_reviews),
                "version": "high_risk_page_coverage_v1",
                "reviews": [item.to_dict() for item in self.high_risk_page_coverage_reviews],
            },
            "coverage_review_tasks": {
                "enabled": bool(self.coverage_review_tasks),
                "version": "coverage_gap_scheduler_v1",
                "tasks": [item.to_dict() for item in self.coverage_review_tasks],
            },
            "qwen_gate": {
                "enabled": bool(self.qwen_gate_reviews),
                "version": "focused_qwen_gate_v1",
                "reviews": [item.to_dict() for item in self.qwen_gate_reviews],
            },
            "specialist_review_plan": {
                "enabled": bool(self.specialist_review_tasks),
                "version": "specialist_review_plan_v1",
                "tasks": [item.to_dict() for item in self.specialist_review_tasks],
            },
            "specialist_review_results": {
                "enabled": bool(self.specialist_review_results),
                "version": "specialist_review_results_v1",
                "results": [item.to_dict() for item in self.specialist_review_results],
            },
            "table_audit_summary": dict(
                self.table_audit_summary
                or {
                    "enabled": False,
                    "version": "table_audit_engine_v1",
                }
            ),
            "quality_inspection": dict(self.quality_inspection),
        }
