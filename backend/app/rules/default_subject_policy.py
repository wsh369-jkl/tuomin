"""Default-line public subject policy.

The default workflow has two different type layers:

* Internal recognizers may keep fine-grained labels such as ``ALIAS``,
  ``BANK_NAME``, ``ACCOUNT_NAME`` and stable numeric/format identifiers because
  those labels are useful evidence for recall, boundary repair and identity
  merging.
* The public subject ledger/replacement code system projects identity-bearing
  rows into only four families: ``PERSON``, ``ORGANIZATION``, ``LOCATION`` and
  ``GOVERNMENT``.

This module owns that boundary. It must not be used as an early recognizer
drop-gate for useful internal evidence.
"""

from __future__ import annotations

from app.core.recognizer_base import RecognizerResult
from app.services.lowmem_entity_utils import normalize_entity_text
from app.services.lowmem_entity_utils import is_official_institution_text


DEFAULT_SUBJECT_TYPES = {"PERSON", "ORGANIZATION", "LOCATION", "GOVERNMENT"}
DEFAULT_INTERNAL_ENTITY_TYPES = {
    *DEFAULT_SUBJECT_TYPES,
    "PERSON_NAME",
    "LEGAL_REPRESENTATIVE",
    "CONTACT_PERSON",
    "SIGNATORY",
    "ORGANIZATION",
    "COMPANY_NAME",
    "ACCOUNT_NAME",
    "BANK_NAME",
    "ALIAS",
    "LOCATION",
    "ADDRESS",
    "GOVERNMENT",
    "GOVERNMENT_AGENCY",
    "COURT",
    "PROJECT",
    "CONTRACT_NO",
    "CASE_NO",
    "CN_ID_CARD",
    "CN_PHONE",
    "LANDLINE_PHONE",
    "CN_BANK_CARD",
    "CN_CREDIT_CODE",
    "EMAIL_ADDRESS",
    "DATE",
    "AMOUNT",
}
_PERSON_ALIASES = {"PERSON", "PERSON_NAME", "LEGAL_REPRESENTATIVE", "CONTACT_PERSON", "SIGNATORY"}
_ORGANIZATION_ALIASES = {"ORGANIZATION", "COMPANY_NAME", "ACCOUNT_NAME"}
_FINANCIAL_ALIASES = {"BANK_NAME"}
_LOCATION_ALIASES = {"LOCATION", "ADDRESS"}
_GOVERNMENT_ALIASES = {"GOVERNMENT", "GOVERNMENT_AGENCY", "COURT"}

GOVERNMENT_MARKERS = (
    "人民法院",
    "中级人民法院",
    "高级人民法院",
    "法院",
    "人民检察院",
    "检察院",
    "公安局",
    "派出所",
    "仲裁委员会",
    "人民政府",
    "政府",
    "委员会",
    "管理委员会",
)
FINANCIAL_MARKERS = ("银行", "支行", "分行", "营业部", "分理处", "信用社", "联社")


def canonical_default_entity_type(entity_type: str, text: str = "") -> str:
    raw_type = str(entity_type or "").strip().upper()
    value = str(text or "")
    normalized = normalize_entity_text(value)
    if raw_type in _PERSON_ALIASES:
        return "PERSON"
    if raw_type in _GOVERNMENT_ALIASES:
        return "GOVERNMENT" if is_official_institution_text(value) else ""
    if raw_type in _LOCATION_ALIASES:
        return "LOCATION"
    if raw_type in _FINANCIAL_ALIASES:
        return "GOVERNMENT" if is_official_institution_text(value) else ""
    if raw_type in _ORGANIZATION_ALIASES:
        if is_official_institution_text(value):
            return "GOVERNMENT"
        if any(marker in normalized for marker in (*GOVERNMENT_MARKERS, *FINANCIAL_MARKERS)):
            return ""
        return "ORGANIZATION"
    return ""


def projected_default_subject_type(result: RecognizerResult) -> str:
    """Return the public four-family subject type for an internal result."""

    raw_type = str(result.entity_type or "").strip().upper()
    metadata = dict(result.metadata or {})
    if raw_type == "ALIAS":
        alias_text = str(result.text or "")
        if not (
            metadata.get("canonical")
            or metadata.get("definition_full_text")
            or metadata.get("definition_alias")
        ):
            return ""
        # Role aliases such as 甲方/乙方 are relationship references, not public
        # subject names. They can still help replacement through metadata.
        if alias_text in {"甲方", "乙方", "丙方", "委托方", "受托方", "发包人", "承包人", "采购人", "供应商"}:
            return ""
        canonical = str(metadata.get("canonical") or "")
        canonical_normalized = normalize_entity_text(canonical)
        if is_official_institution_text(canonical):
            return "GOVERNMENT"
        if any(marker in canonical_normalized for marker in (*GOVERNMENT_MARKERS, *FINANCIAL_MARKERS)):
            return ""
        return "ORGANIZATION"
    return canonical_default_entity_type(raw_type, result.text)


def is_default_subject_type(entity_type: str, text: str = "") -> bool:
    return canonical_default_entity_type(entity_type, text) in DEFAULT_SUBJECT_TYPES


def is_default_allowed_entity_type(entity_type: str, text: str = "") -> bool:
    return str(entity_type or "").strip().upper() in DEFAULT_INTERNAL_ENTITY_TYPES


def canonicalize_default_result(result: RecognizerResult) -> RecognizerResult | None:
    raw_type = str(result.entity_type or "").strip().upper()
    if raw_type not in DEFAULT_INTERNAL_ENTITY_TYPES:
        return None
    if raw_type == result.entity_type:
        return result
    metadata = dict(result.metadata or {})
    metadata.setdefault("default_internal_type_from", result.entity_type)
    return RecognizerResult(
        entity_type=raw_type,
        start=result.start,
        end=result.end,
        score=result.score,
        text=result.text,
        source=result.source,
        metadata=metadata,
    )
