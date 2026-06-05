from __future__ import annotations

import json
import re
import signal
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import requests

from app.core.config import settings

from .common import normalize_text
from .models import ConversionDiffCandidate, ConversionPreflightResult, FocusedCandidateReview, QwenGateReview


@dataclass
class QwenStructuredResult:
    ok: bool
    model: str
    endpoint: str
    parsed: Dict[str, Any] = field(default_factory=dict)
    raw_content: str = ""
    error: str = ""
    attempts: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "model": self.model,
            "endpoint": self.endpoint,
            "parsed": dict(self.parsed),
            "raw_content_chars": len(self.raw_content),
            "error": self.error,
            "attempts": int(self.attempts or 0),
            "metadata": dict(self.metadata),
        }


class OllamaWallTimeout(TimeoutError):
    pass


@contextmanager
def ollama_wall_timeout(seconds: int):
    """Bound a blocking Ollama HTTP call by wall-clock time in worker processes."""

    limit = max(1, int(seconds or 1))
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)

    def _handle_timeout(signum, frame):  # noqa: ARG001
        raise OllamaWallTimeout()

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, limit)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer and previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def parse_json_object(value: Any) -> tuple[Dict[str, Any], str]:
    if isinstance(value, dict):
        return value, ""
    text = str(value or "").strip()
    if not text:
        return {}, "empty_response"
    candidates = _json_object_candidates(text)
    last_error = "json_object_not_found"
    best: tuple[int, int, Dict[str, Any]] | None = None
    for index, candidate in enumerate(candidates):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = f"JSONDecodeError:{exc.msg}:{exc.pos}"
            continue
        if isinstance(parsed, dict):
            score = _json_object_score(parsed)
            if best is None or score > best[0] or (score == best[0] and index > best[1]):
                best = (score, index, parsed)
            continue
        last_error = f"response_not_object:{type(parsed).__name__}"
    if best is not None:
        score, _, parsed = best
        if score >= 3:
            return parsed, ""
        return {}, f"json_object_not_final:{score}"
    return {}, last_error


def _json_object_candidates(text: str) -> List[str]:
    candidates: List[str] = []

    def add(candidate: str) -> None:
        value = str(candidate or "").strip()
        if value and value not in candidates:
            candidates.append(value)

    add(text)
    cleaned = re.sub(r"^```(?:json)?\s*", "", str(text or "").strip(), flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    add(cleaned)
    without_think = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.I | re.S).strip()
    if without_think and without_think != cleaned:
        add(without_think)
    repaired = _balanced_json_object_candidate(cleaned)
    if repaired:
        add(repaired)
    repaired_without_think = _balanced_json_object_candidate(without_think)
    if repaired_without_think:
        add(repaired_without_think)
    for match in re.finditer(r"\{", cleaned):
        start = match.start()
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(cleaned)):
            char = cleaned[index]
            if escaped:
                escaped = False
                continue
            if in_string and char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    add(cleaned[start : index + 1])
                    break
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first >= 0 and last > first:
        add(cleaned[first : last + 1])
    return candidates


def _json_object_score(value: Dict[str, Any]) -> int:
    keys = {str(key) for key in value.keys()}
    score = 0
    if "verdict" in keys:
        score += 14
    if "decision" in keys:
        score += 8
    if "suspicious_values" in keys:
        score += 12
        if isinstance(value.get("suspicious_values"), list) and value.get("suspicious_values"):
            score += 4
    if "reason" in keys:
        score += 5
    if "confidence" in keys:
        score += 4
    if "next_route" in keys:
        score += 3
    if "visible_text_excerpt" in keys:
        score += 3
    if "preferred_text" in keys or "visible_text" in keys:
        score += 3
    if "docx_text" in keys and "visible_text" in keys:
        score += 12
    if "issue_type" in keys and "severity" in keys:
        score += 4

    output_limit_keys = {"max_suspicious_values", "row_col_type"}
    sample_keys = {"id", "sample_id", "unit_id", "text", "cat", "status", "row", "col", "row_text"}
    prompt_payload_keys = {
        "page_no",
        "unresolved_table_cell_count",
        "pdf_table_signal",
        "image_orientation",
        "docx_samples",
        "output_limits",
        "rules",
    }
    if keys and keys <= output_limit_keys:
        score -= 30
    if keys and keys <= sample_keys and not {"docx_text", "visible_text"}.issubset(keys):
        score -= 24
    if keys and keys <= prompt_payload_keys:
        score -= 24
    if "docx_samples" in keys or "rules" in keys or "output_limits" in keys:
        score -= 12
    return score


def _balanced_json_object_candidate(text: str) -> str:
    """Best-effort recovery for model JSON truncated after a valid prefix."""

    source = str(text or "").strip()
    start = source.find("{")
    if start < 0:
        return ""
    value = source[start:]
    stack: List[str] = []
    in_string = False
    escaped = False
    last_meaningful = ""
    for char in value:
        if escaped:
            escaped = False
            continue
        if in_string and char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            last_meaningful = char
            continue
        if in_string:
            continue
        if char in "{[":
            stack.append(char)
            last_meaningful = char
        elif char == "}":
            if stack and stack[-1] == "{":
                stack.pop()
            last_meaningful = char
        elif char == "]":
            if stack and stack[-1] == "[":
                stack.pop()
            last_meaningful = char
        elif not char.isspace():
            last_meaningful = char
    if not stack and not in_string:
        return ""
    if len(stack) > 20:
        return ""
    repaired = value.rstrip()
    if in_string:
        repaired += '"'
    if last_meaningful == ",":
        repaired = repaired.rstrip().rstrip(",")
    closers = {"{": "}", "[": "]"}
    repaired += "".join(closers[item] for item in reversed(stack))
    return repaired


def qwen_gate_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["confirmed_error", "suspected_error", "model_conflict", "no_error", "coverage_gap"],
            },
            "decision": {
                "type": "string",
                "enum": ["allow_report_candidate", "block_candidate", "defer"],
            },
            "reason": {"type": "string"},
            "preferred_text": {"type": "string"},
            "confidence": {"type": "number"},
            "next_route": {"type": "string"},
        },
        "required": ["verdict", "decision", "reason", "preferred_text", "confidence", "next_route"],
    }


def _audit_review_model() -> str:
    configured = str(getattr(settings, "PDF_WORD_AUDIT_REVIEW_MODEL", "") or "").strip()
    if configured:
        return configured
    fallback = getattr(settings, "get_default_review_llm_model", lambda: None)()
    return str(fallback or "qwen3.5:9b").strip()


class FocusedQwenGateBuilder:
    """Run Qwen as a final conservative gate for focused text candidates."""

    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        max_candidates: Optional[int] = None,
        client: Any = None,
    ) -> None:
        self.enabled = bool(getattr(settings, "PDF_WORD_AUDIT_V4_QWEN_GATE_ENABLED", True) if enabled is None else enabled)
        self.model = str(model or _audit_review_model() or "qwen3.5:9b").strip()
        self.timeout = max(1, int(timeout or getattr(settings, "PDF_WORD_AUDIT_V4_QWEN_GATE_TIMEOUT", 120) or 120))
        self.max_candidates = max(0, int(max_candidates if max_candidates is not None else getattr(settings, "PDF_WORD_AUDIT_V4_QWEN_GATE_MAX_CANDIDATES", 9999) or 9999))
        self.client = client or OllamaQwenGateClient(model=self.model)

    def build(self, *, preflight_result: ConversionPreflightResult) -> List[QwenGateReview]:
        if not self.enabled or self.max_candidates <= 0:
            return []
        selected = self._selected_reviews(preflight_result=preflight_result)
        if not selected:
            return []
        available, preflight_error = self._preflight()
        if not available:
            unload = getattr(self.client, "unload", None)
            if callable(unload):
                unload()
            return [
                QwenGateReview(
                    gate_id=f"qwen_gate_{index:04d}",
                    diff_id=review.diff_id,
                    attempted=False,
                    available=False,
                    model=self.model,
                    verdict="coverage_gap",
                    decision="defer",
                    next_route=review.next_route or "needs_human_mapping_review",
                    error=preflight_error or "qwen_gate_unavailable",
                    flags=["qwen_preflight_failed"],
                )
                for index, review in enumerate(selected, start=1)
            ]

        diff_by_id = {item.diff_id: item for item in preflight_result.diff_candidates}
        rows: List[QwenGateReview] = []
        for index, review in enumerate(selected, start=1):
            diff = diff_by_id.get(review.diff_id)
            rows.append(self._review(index=index, review=review, diff=diff))
        unload = getattr(self.client, "unload", None)
        if callable(unload):
            unload()
        return rows

    def _selected_reviews(self, *, preflight_result: ConversionPreflightResult) -> List[FocusedCandidateReview]:
        priority = {"possible_conversion_error": 0}
        selected = [
            item
            for item in preflight_result.focused_reviews
            if item.decision == "possible_conversion_error"
        ]
        selected.sort(key=lambda item: (priority.get(item.decision, 99), item.page_no or 9999, item.diff_id))
        return selected[: self.max_candidates]

    def _preflight(self) -> tuple[bool, str]:
        preflight = getattr(self.client, "preflight", None)
        if not callable(preflight):
            return True, ""
        result = preflight(timeout=min(30, self.timeout))
        if bool(result.get("available")):
            return True, ""
        return False, str(result.get("error") or result.get("reason") or "qwen_preflight_failed")

    def _review(
        self,
        *,
        index: int,
        review: FocusedCandidateReview,
        diff: ConversionDiffCandidate | None,
    ) -> QwenGateReview:
        result = self.client.structured_chat(
            system_prompt=self._system_prompt(),
            user_prompt=self._user_prompt(review=review, diff=diff),
            schema=qwen_gate_schema(),
            timeout=self.timeout,
            num_predict=220,
            temperature=0.0,
        )
        if not getattr(result, "ok", False):
            return QwenGateReview(
                gate_id=f"qwen_gate_{index:04d}",
                diff_id=review.diff_id,
                attempted=True,
                available=False,
                model=self.model,
                verdict="coverage_gap",
                decision="defer",
                next_route=review.next_route or "needs_human_mapping_review",
                error=str(getattr(result, "error", "") or "qwen_gate_failed"),
                flags=["qwen_gate_failed"],
            )
        parsed = dict(getattr(result, "parsed", {}) or {})
        verdict = self._choice(parsed.get("verdict"), {"confirmed_error", "suspected_error", "model_conflict", "no_error", "coverage_gap"}, "coverage_gap")
        decision = self._choice(parsed.get("decision"), {"allow_report_candidate", "block_candidate", "defer"}, "defer")
        confidence = self._confidence(parsed.get("confidence"))
        preferred_text = str(parsed.get("preferred_text") or "")[:180]
        next_route = str(parsed.get("next_route") or "").strip()
        flags: List[str] = ["qwen_final_gate"]

        if review.decision != "possible_conversion_error" and decision == "allow_report_candidate":
            decision = "defer"
            next_route = next_route or review.next_route or "needs_qwen_vl"
            flags.append("qwen_allow_blocked_by_visual_gate")
        if (
            decision == "defer"
            and verdict != "no_error"
            and self._stable_table_cell_should_allow(review=review, preferred_text=preferred_text, confidence=confidence)
        ):
            verdict = "suspected_error"
            decision = "allow_report_candidate"
            next_route = ""
            flags.append("qwen_defer_promoted_by_stable_table_cell_evidence")
        if decision == "allow_report_candidate" and (verdict not in {"confirmed_error", "suspected_error"} or not preferred_text):
            decision = "defer"
            next_route = next_route or "needs_human_mapping_review"
            flags.append("qwen_allow_failed_safety_requirements")
        if verdict == "no_error":
            decision = "block_candidate"
            next_route = ""
        if review.status == "ready_for_table_gate" and decision == "defer" and next_route in {"", "needs_table_parser"}:
            next_route = "needs_human_table_review"
            flags.append("table_gate_deferred_after_partial_parse")

        return QwenGateReview(
            gate_id=f"qwen_gate_{index:04d}",
            diff_id=review.diff_id,
            attempted=True,
            available=True,
            model=self.model,
            verdict=verdict,
            decision=decision,
            confidence=confidence,
            reason=self._clip_reason(parsed.get("reason")),
            preferred_text=preferred_text,
            next_route=next_route,
            flags=flags,
        )

    def _stable_table_cell_should_allow(
        self,
        *,
        review: FocusedCandidateReview,
        preferred_text: str,
        confidence: float,
    ) -> bool:
        table_cell = dict(getattr(review, "table_cell", {}) or {})
        if not table_cell.get("stable_text_match"):
            return False
        if "crop_ocr_supports_pdf" not in set(review.flags):
            return False
        if confidence < 0.72:
            return False
        pdf_norm = self._semantic_compact(table_cell.get("pdf_text") or review.pdf_text)
        preferred_norm = self._semantic_compact(preferred_text)
        docx_norm = self._semantic_compact(table_cell.get("docx_text") or review.docx_text)
        cell_norm = self._semantic_compact(table_cell.get("cell_ocr_text") or table_cell.get("fallback_ocr_text") or "")
        if not pdf_norm or not docx_norm or pdf_norm == docx_norm:
            return False
        if preferred_norm and preferred_norm != pdf_norm:
            return False
        if cell_norm and pdf_norm not in cell_norm:
            return False
        if table_cell.get("confusable_substitution"):
            return True
        if not self._stable_table_numeric_conflict(
            pdf_text=table_cell.get("pdf_text") or review.pdf_text,
            docx_text=table_cell.get("docx_text") or review.docx_text,
        ):
            return False
        return True

    def _stable_table_numeric_conflict(self, *, pdf_text: Any, docx_text: Any) -> bool:
        pdf_value = self._normalize_table_number(pdf_text)
        docx_value = self._normalize_table_number(docx_text)
        if not pdf_value or not docx_value or pdf_value == docx_value:
            return False
        if abs(len(pdf_value) - len(docx_value)) > 1:
            return False
        return True

    def _normalize_table_number(self, text: Any) -> str:
        value = str(text or "").strip().replace(",", "")
        if not re.fullmatch(r"[¥￥]?\s*\d+(?:\.\d+)?", value):
            return ""
        return re.sub(r"\D", "", value)

    def _semantic_compact(self, text: Any) -> str:
        value = normalize_text(str(text or "")).lower()
        value = value.replace("〇", "0").replace("○", "0")
        value = re.sub(r"\s+", "", value)
        return re.sub(r"[^\w\u4e00-\u9fff]+", "", value)

    def _system_prompt(self) -> str:
        return (
            "你是 WPS PDF 转 DOCX 忠实度审查的 Qwen 最终门槛。"
            "你不能看原图，只能根据预审文本、局部 crop OCR 和候选上下文做保守判断。"
            "你的首要目标是阻止误报，不是尽量多报错。必须只输出 JSON。"
        )

    def _user_prompt(self, *, review: FocusedCandidateReview, diff: ConversionDiffCandidate | None) -> str:
        payload = {
            "diff": diff.to_dict() if diff else {"diff_id": review.diff_id},
            "focused_review": review.to_dict(),
            "rules": [
                "只有 PDF 候选文本与局部 crop OCR 都稳定支持同一个可见值，并且该值与 DOCX 直接冲突，才可 confirmed_error 或 suspected_error。",
                "如果 crop OCR 更支持 DOCX，必须 no_error。",
                "如果 PDF OCR、crop OCR、DOCX 三者互相冲突，必须 model_conflict 或 coverage_gap。",
                "普通文本/案号/编号差异必须查看 focused_review.visual_text：只有 visual_text.stable=true 且 support=pdf，才可 suspected_error；support=docx/conflict/low_quality_ambiguous 时必须 block_candidate 或 defer。",
                "表格结构未解析、页映射不稳定、crop OCR 质量 low 时，通常 defer，不得 allow_report_candidate。",
                "如果 focused_review.table_cell.stable_text_match=true，且 table_cell.confusable_substitution=true，表示 PDF 表格单元局部证据与 PDF 候选一致、DOCX 是明显数字/字符替换；此时即使普通 crop OCR quality=low，也应 suspected_error + allow_report_candidate。",
                "如果 focused_review.flags 包含 table_structure_parseable/table_structure_partial_with_docx/table_cell_evidence_stable，并且同时有 crop_ocr_supports_pdf，且 DOCX 文本是明显数字/字符替换，可 suspected_error。",
                "格式、空格、标点、全半角、换行差异不是转换错误。",
            ],
        }
        return (
            "请审查以下单个候选。输出字段固定为 verdict, decision, reason, preferred_text, confidence, next_route。\n"
            "decision=allow_report_candidate 表示这条候选证据足够写入 reviewed.docx 的疑似错误批注，但仍不自动修改正文。\n"
            "next_route 可为空，或填写 needs_qwen_vl / needs_region_ocr / needs_table_parser / needs_human_mapping_review。\n"
            f"候选 JSON：{json.dumps(payload, ensure_ascii=False)[:6200]}"
        )

    def _choice(self, value: Any, allowed: Sequence[str] | set[str], fallback: str) -> str:
        text = str(value or "").strip()
        return text if text in allowed else fallback

    def _confidence(self, value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return 0.0

    def _clip_reason(self, value: Any, *, limit: int = 80) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        first_sentence = text.split("。", 1)[0].strip()
        if first_sentence and len(first_sentence) <= limit:
            return first_sentence + "。"
        return text[:limit].rstrip() + "..."


class OllamaQwenGateClient:
    def __init__(self, *, model: str, base_url: Optional[str] = None) -> None:
        self.model = str(model or "").strip()
        self.base_url = str(base_url or settings.OLLAMA_BASE_URL or "").rstrip("/")

    @property
    def chat_endpoint(self) -> str:
        return f"{self.base_url}/api/chat"

    @property
    def generate_endpoint(self) -> str:
        return f"{self.base_url}/api/generate"

    def preflight(self, *, timeout: int = 30) -> Dict[str, Any]:
        result = self.structured_chat(
            system_prompt="关闭思考。你是结构化 JSON 连通性探针。",
            user_prompt='返回 {"ok":true,"token":"wps_qwen_preflight","reason":"ready"}',
            schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "ok": {"type": "boolean"},
                    "token": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["ok", "token", "reason"],
            },
            timeout=timeout,
            num_predict=100,
            temperature=0.0,
        )
        payload = result.to_dict()
        parsed = result.parsed or {}
        payload["available"] = bool(result.ok and parsed.get("ok") is True and parsed.get("token") == "wps_qwen_preflight")
        if not payload["available"] and not payload.get("error"):
            payload["error"] = "qwen_preflight_schema_mismatch"
        return payload

    def structured_chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: Dict[str, Any],
        timeout: int,
        num_predict: int = 256,
        temperature: float = 0.0,
    ) -> QwenStructuredResult:
        if not self.base_url:
            return QwenStructuredResult(ok=False, model=self.model, endpoint="", error="ollama_base_url_missing")
        if not self.model:
            return QwenStructuredResult(ok=False, model="", endpoint=self.chat_endpoint, error="qwen_model_missing")
        messages = [
            {"role": "system", "content": "/no_think\n关闭思考。只输出合法 JSON。\n" + str(system_prompt or "")},
            {"role": "user", "content": "/no_think\n" + str(user_prompt or "")},
        ]
        chat_payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": schema,
            "think": False,
            "keep_alive": "1m",
            "options": {
                "temperature": float(temperature),
                "num_predict": max(1, int(num_predict or 256)),
            },
        }
        result = self._post_chat(chat_payload=chat_payload, timeout=timeout)
        if result.ok:
            return result
        generate_payload = {
            "model": self.model,
            "prompt": messages[0]["content"] + "\n" + messages[1]["content"],
            "stream": False,
            "format": "json",
            "think": False,
            "keep_alive": "1m",
            "options": {
                "temperature": float(temperature),
                "num_predict": max(1, int(num_predict or 256)),
            },
        }
        fallback = self._post_generate(generate_payload=generate_payload, timeout=timeout)
        if fallback.ok:
            fallback.attempts = 2
            fallback.metadata["chat_error"] = result.error
        return fallback

    def unload(self) -> None:
        if not self.base_url or not self.model:
            return
        try:
            requests.post(
                self.generate_endpoint,
                json={"model": self.model, "prompt": "", "stream": False, "keep_alive": 0},
                timeout=5,
            )
        except Exception:
            return

    def _post_chat(self, *, chat_payload: Dict[str, Any], timeout: int) -> QwenStructuredResult:
        timeout_seconds = max(1, int(timeout or 1))
        try:
            with ollama_wall_timeout(timeout_seconds):
                response = requests.post(
                    self.chat_endpoint,
                    json=chat_payload,
                    timeout=(min(10, timeout_seconds), timeout_seconds),
                )
            response.raise_for_status()
            raw = response.json()
            content = self._chat_content(raw)
            parsed, error = parse_json_object(content)
            if error:
                return QwenStructuredResult(
                    ok=False,
                    model=self.model,
                    endpoint=self.chat_endpoint,
                    raw_content=content,
                    error=error,
                    metadata={"transport": "chat", "timeout_seconds": timeout_seconds},
                )
            return QwenStructuredResult(
                ok=True,
                model=self.model,
                endpoint=self.chat_endpoint,
                parsed=parsed,
                raw_content=content,
                metadata={"transport": "chat", "timeout_seconds": timeout_seconds},
            )
        except OllamaWallTimeout:
            return QwenStructuredResult(
                ok=False,
                model=self.model,
                endpoint=self.chat_endpoint,
                error="OllamaWallTimeout",
                metadata={"transport": "chat", "timeout_seconds": timeout_seconds},
            )
        except Exception as exc:
            return QwenStructuredResult(
                ok=False,
                model=self.model,
                endpoint=self.chat_endpoint,
                error=type(exc).__name__,
                metadata={"transport": "chat", "timeout_seconds": timeout_seconds},
            )

    def _post_generate(self, *, generate_payload: Dict[str, Any], timeout: int) -> QwenStructuredResult:
        timeout_seconds = max(1, int(timeout or 1))
        try:
            with ollama_wall_timeout(timeout_seconds):
                response = requests.post(
                    self.generate_endpoint,
                    json=generate_payload,
                    timeout=(min(10, timeout_seconds), timeout_seconds),
                )
            response.raise_for_status()
            raw = response.json()
            content = self._generate_content(raw)
            parsed, error = parse_json_object(content)
            if error:
                return QwenStructuredResult(
                    ok=False,
                    model=self.model,
                    endpoint=self.generate_endpoint,
                    raw_content=content,
                    error=error,
                    metadata={"transport": "generate", "timeout_seconds": timeout_seconds},
                )
            return QwenStructuredResult(
                ok=True,
                model=self.model,
                endpoint=self.generate_endpoint,
                parsed=parsed,
                raw_content=content,
                metadata={"transport": "generate", "timeout_seconds": timeout_seconds},
            )
        except OllamaWallTimeout:
            return QwenStructuredResult(
                ok=False,
                model=self.model,
                endpoint=self.generate_endpoint,
                error="OllamaWallTimeout",
                metadata={"transport": "generate", "timeout_seconds": timeout_seconds},
            )
        except Exception as exc:
            return QwenStructuredResult(
                ok=False,
                model=self.model,
                endpoint=self.generate_endpoint,
                error=type(exc).__name__,
                metadata={"transport": "generate", "timeout_seconds": timeout_seconds},
            )

    def _chat_content(self, raw: Any) -> str:
        if isinstance(raw, dict):
            message = raw.get("message")
            if isinstance(message, dict):
                content = str(message.get("content") or "")
                if content.strip():
                    return content
                thinking = str(message.get("thinking") or "")
                if thinking.strip():
                    return thinking
            if isinstance(raw.get("response"), str):
                return str(raw.get("response") or "")
        return str(raw or "")

    def _generate_content(self, raw: Any) -> str:
        if isinstance(raw, dict):
            content = str(raw.get("response") or "")
            if content.strip():
                return content
            thinking = str(raw.get("thinking") or "")
            if thinking.strip():
                return thinking
        return str(raw or "")
