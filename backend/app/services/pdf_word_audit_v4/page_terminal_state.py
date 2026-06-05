from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Sequence


class PageTerminalStateMachine:
    VERSION = "page_terminal_state_v1"
    RESOLVED_STATES = {"covered", "confirmed_error"}
    REVIEW_REQUIRED_STATES = {
        "human_review_required",
        "coverage_gap",
        "model_conflict",
        "suspected_error",
    }
    LABELS = {
        "covered": "已覆盖",
        "confirmed_error": "确认错误",
        "suspected_error": "疑似错误",
        "model_conflict": "模型冲突",
        "coverage_gap": "覆盖缺口",
        "human_review_required": "待人工复核",
    }

    def apply(self, *, pages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for source in pages:
            page = dict(source)
            state = self._terminal_state(page)
            page["terminal_state"] = state
            page["terminal_label"] = self.LABELS[state]
            page["terminal_reason"] = self._terminal_reason(page=page, state=state)
            page["is_terminal"] = True
            page["is_resolved"] = state in self.RESOLVED_STATES
            page["requires_human_review"] = state in self.REVIEW_REQUIRED_STATES
            page["open_issue_count"] = self._open_issue_count(page)
            rows.append(page)
        return rows

    def summary(self, *, pages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        rows = list(pages)
        state_counts = Counter(str(item.get("terminal_state") or "unknown") for item in rows)
        high_risk_rows = [item for item in rows if str(item.get("risk_level") or "") == "high"]
        high_risk_state_counts = Counter(str(item.get("terminal_state") or "unknown") for item in high_risk_rows)
        return {
            "enabled": True,
            "version": self.VERSION,
            "page_count": len(rows),
            "terminal_state_counts": dict(sorted(state_counts.items())),
            "resolved_page_count": sum(1 for item in rows if bool(item.get("is_resolved"))),
            "review_required_page_count": sum(1 for item in rows if bool(item.get("requires_human_review"))),
            "open_issue_total": sum(int(item.get("open_issue_count") or 0) for item in rows),
            "high_risk_page_count": len(high_risk_rows),
            "high_risk_resolved_page_count": sum(1 for item in high_risk_rows if bool(item.get("is_resolved"))),
            "high_risk_review_required_page_count": sum(
                1 for item in high_risk_rows if bool(item.get("requires_human_review"))
            ),
            "high_risk_terminal_state_counts": dict(sorted(high_risk_state_counts.items())),
        }

    def _terminal_state(self, page: Dict[str, Any]) -> str:
        review_task_count = int(page.get("review_task_count") or 0)
        unresolved_coverage_count = int(page.get("unresolved_coverage_count") or 0)
        coverage_gap_count = int(page.get("coverage_gap_count") or 0)
        model_conflict_count = int(page.get("model_conflict_count") or 0)
        suspected_count = int(page.get("suspected_count") or 0)
        confirmed_count = int(page.get("confirmed_count") or 0)

        if review_task_count > 0 or unresolved_coverage_count > 0:
            return "human_review_required"
        if coverage_gap_count > 0:
            return "coverage_gap"
        if model_conflict_count > 0:
            return "model_conflict"
        if suspected_count > 0:
            return "suspected_error"
        if confirmed_count > 0:
            return "confirmed_error"
        return "covered"

    def _open_issue_count(self, page: Dict[str, Any]) -> int:
        return sum(
            max(0, int(page.get(key) or 0))
            for key in (
                "review_task_count",
                "unresolved_coverage_count",
                "coverage_gap_count",
                "model_conflict_count",
                "suspected_count",
            )
        )

    def _terminal_reason(self, *, page: Dict[str, Any], state: str) -> str:
        review_task_count = int(page.get("review_task_count") or 0)
        unresolved_coverage_count = int(page.get("unresolved_coverage_count") or 0)
        coverage_gap_count = int(page.get("coverage_gap_count") or 0)
        model_conflict_count = int(page.get("model_conflict_count") or 0)
        suspected_count = int(page.get("suspected_count") or 0)
        confirmed_count = int(page.get("confirmed_count") or 0)

        if state == "human_review_required":
            parts: List[str] = []
            if review_task_count > 0:
                parts.append(f"仍有 {review_task_count} 个页级复核任务")
            if unresolved_coverage_count > 0:
                parts.append(f"{unresolved_coverage_count} 个内容单元未闭环")
            return "；".join(parts) or "本页仍有未闭环内容，需要进入人工复核终态。"
        if state == "coverage_gap":
            return f"本页存在 {coverage_gap_count} 个覆盖缺口发现，尚不能判定为已覆盖。"
        if state == "model_conflict":
            return f"本页存在 {model_conflict_count} 个模型冲突发现，需要人工裁决。"
        if state == "suspected_error":
            return f"本页存在 {suspected_count} 个疑似错误，但尚未形成确认错误证据。"
        if state == "confirmed_error":
            return f"本页已形成 {confirmed_count} 个确认错误，且没有剩余开放覆盖任务。"
        return "本页未发现错误且没有剩余开放覆盖任务。"
