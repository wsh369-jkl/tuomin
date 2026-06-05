from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .common import normalize_text, similarity


GOLD_EVAL_VERSION = "v4_gold_eval_v1"
GOLD_EVAL_SUITE_VERSION = "v4_gold_eval_suite_v1"


DEFAULT_THRESHOLDS = {
    "min_confirmed_recall": 0.95,
    "max_confirmed_false_positive_rate": 0.10,
    "max_critical_missed": 0,
}
DEFAULT_SUITE_THRESHOLDS = {
    "max_failed_samples": 0,
    "min_review_page_coverage_ratio": 1.0,
}


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def write_gold_eval_summary(summary: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_gold_files(
    *,
    report_path: str | Path,
    annotation_path: str | Path,
    thresholds: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return evaluate_gold_payloads(
        report_payload=load_json(report_path),
        annotation_payload=load_json(annotation_path),
        thresholds=thresholds,
        report_path=Path(report_path).expanduser(),
        annotation_path=Path(annotation_path).expanduser(),
    )


def load_gold_suite_manifest(path: str | Path) -> List[Dict[str, Any]]:
    manifest_path = Path(path).expanduser()
    payload = load_json(manifest_path)
    rows = payload.get("samples") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list):
        return []
    manifest_dir = manifest_path.parent
    pairs: List[Dict[str, Any]] = []
    for index, item in enumerate(rows, start=1):
        if not isinstance(item, Mapping):
            continue
        report_raw = str(item.get("report") or item.get("report_path") or "").strip()
        annotation_raw = str(item.get("gold") or item.get("annotation") or item.get("annotation_path") or "").strip()
        pairs.append(
            {
                "sample_id": str(item.get("sample_id") or f"sample_{index:04d}"),
                "report_path": _resolve_manifest_entry_path(report_raw, manifest_dir),
                "annotation_path": _resolve_manifest_entry_path(annotation_raw, manifest_dir),
            }
        )
    return pairs


def evaluate_gold_suite_manifest(
    *,
    manifest_path: str | Path,
    thresholds: Optional[Mapping[str, Any]] = None,
    suite_thresholds: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    pairs = load_gold_suite_manifest(manifest_path)
    summary = evaluate_gold_suite(
        pairs=pairs,
        thresholds=thresholds,
        suite_thresholds=suite_thresholds,
    )
    summary["paths"] = dict(summary.get("paths") or {})
    summary["paths"]["manifest_path"] = str(Path(manifest_path).expanduser())
    return summary


def evaluate_gold_payloads(
    *,
    report_payload: Mapping[str, Any],
    annotation_payload: Mapping[str, Any],
    thresholds: Optional[Mapping[str, Any]] = None,
    report_path: Optional[Path] = None,
    annotation_path: Optional[Path] = None,
) -> Dict[str, Any]:
    merged_thresholds = dict(DEFAULT_THRESHOLDS)
    merged_thresholds.update({key: value for key, value in dict(thresholds or {}).items() if value is not None})

    expected_errors = _expected_errors(annotation_payload)
    findings = _findings(report_payload)
    confirmed_findings = [item for item in findings if item.get("status") == "confirmed_error"]
    expected_review_pages = {
        int(page)
        for page in annotation_payload.get("expected_review_pages", [])
        if _safe_int(page) > 0
    }

    matches: List[Dict[str, Any]] = []
    matched_finding_ids: set[str] = set()
    for expected in expected_errors:
        match = _best_match(expected=expected, findings=confirmed_findings, used_finding_ids=matched_finding_ids)
        if match:
            matched_finding_ids.add(str(match["finding"].get("id") or ""))
            matches.append(
                {
                    "expected_id": expected.get("id"),
                    "finding_id": match["finding"].get("id"),
                    "page_no": expected.get("page_no"),
                    "score": round(float(match["score"] or 0.0), 4),
                    "critical": bool(expected.get("critical")),
                    "category": expected.get("category") or match["finding"].get("category"),
                }
            )

    matched_expected_ids = {str(item.get("expected_id") or "") for item in matches}
    missed = [item for item in expected_errors if str(item.get("id") or "") not in matched_expected_ids]
    false_positive_confirmed = [
        _finding_issue(item)
        for item in confirmed_findings
        if str(item.get("id") or "") not in matched_finding_ids
    ]
    critical_missed = [item for item in missed if bool(item.get("critical"))]
    table_expected = [item for item in expected_errors if str(item.get("category") or "").startswith("table")]
    table_matched = [item for item in matches if str(item.get("category") or "").startswith("table")]
    review_pages_covered = _review_pages_covered(report_payload=report_payload, findings=findings, expected_pages=expected_review_pages)

    expected_count = len(expected_errors)
    matched_count = len(matches)
    confirmed_count = len(confirmed_findings)
    recall = matched_count / expected_count if expected_count else 1.0
    false_positive_rate = (
        len(false_positive_confirmed) / confirmed_count
        if confirmed_count
        else 0.0
    )
    table_recall = len(table_matched) / len(table_expected) if table_expected else 1.0

    metrics = {
        "expected_error_count": expected_count,
        "matched_confirmed_count": matched_count,
        "missed_count": len(missed),
        "critical_missed_count": len(critical_missed),
        "confirmed_finding_count": confirmed_count,
        "false_positive_confirmed_count": len(false_positive_confirmed),
        "confirmed_recall": round(recall, 4),
        "confirmed_false_positive_rate": round(false_positive_rate, 4),
        "table_expected_count": len(table_expected),
        "table_matched_count": len(table_matched),
        "table_recall": round(table_recall, 4),
        "expected_review_page_count": len(expected_review_pages),
        "review_page_covered_count": len(review_pages_covered),
    }
    checks = _checks(metrics=metrics, thresholds=merged_thresholds)
    return {
        "version": GOLD_EVAL_VERSION,
        "sample_id": annotation_payload.get("sample_id") or report_payload.get("audit_id") or "",
        "status": "passed" if all(item["passed"] for item in checks if item["severity"] == "fail") else "failed",
        "paths": {
            "report_path": str(report_path) if report_path else "",
            "annotation_path": str(annotation_path) if annotation_path else "",
        },
        "thresholds": merged_thresholds,
        "metrics": metrics,
        "matches": matches,
        "missed_expected_errors": [_expected_issue(item) for item in missed],
        "critical_missed_errors": [_expected_issue(item) for item in critical_missed],
        "false_positive_confirmed": false_positive_confirmed[:80],
        "review_pages": {
            "expected_pages": sorted(expected_review_pages),
            "covered_pages": sorted(review_pages_covered),
            "missing_pages": sorted(expected_review_pages - review_pages_covered),
        },
        "checks": checks,
    }


def evaluate_gold_suite(
    *,
    pairs: Sequence[Mapping[str, Any]],
    thresholds: Optional[Mapping[str, Any]] = None,
    suite_thresholds: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    merged_thresholds = dict(DEFAULT_THRESHOLDS)
    merged_thresholds.update({key: value for key, value in dict(thresholds or {}).items() if value is not None})
    merged_suite_thresholds = dict(DEFAULT_SUITE_THRESHOLDS)
    merged_suite_thresholds.update({key: value for key, value in dict(suite_thresholds or {}).items() if value is not None})

    sample_summaries: List[Dict[str, Any]] = []
    for index, pair in enumerate(pairs, start=1):
        if not isinstance(pair, Mapping):
            continue
        summary = _evaluate_gold_suite_pair(pair=pair, thresholds=merged_thresholds, index=index)
        sample_id = str(pair.get("sample_id") or summary.get("sample_id") or f"sample_{index:04d}")
        summary["sample_id"] = sample_id
        sample_summaries.append(summary)

    aggregate = _aggregate_gold_suite_metrics(sample_summaries)
    failed_samples = [
        {
            "sample_id": str(item.get("sample_id") or ""),
            "status": str(item.get("status") or ""),
            "critical_missed_count": int(((item.get("metrics") or {}).get("critical_missed_count")) or 0),
            "confirmed_recall": float(((item.get("metrics") or {}).get("confirmed_recall")) or 0.0),
            "paths": dict(item.get("paths") or {}),
        }
        for item in sample_summaries
        if str(item.get("status") or "") != "passed"
    ]
    checks = _suite_checks(
        metrics=aggregate,
        sample_thresholds=merged_thresholds,
        suite_thresholds=merged_suite_thresholds,
        failed_sample_count=len(failed_samples),
    )
    status = "passed" if sample_summaries and all(item["passed"] for item in checks) else "failed"
    return {
        "version": GOLD_EVAL_SUITE_VERSION,
        "status": status,
        "thresholds": {
            "sample": merged_thresholds,
            "suite": merged_suite_thresholds,
        },
        "metrics": aggregate,
        "checks": checks,
        "sample_count": len(sample_summaries),
        "passed_sample_count": sum(1 for item in sample_summaries if str(item.get("status") or "") == "passed"),
        "failed_sample_count": len(failed_samples),
        "failed_samples": failed_samples[:200],
        "samples": [
            {
                "sample_id": str(item.get("sample_id") or ""),
                "status": str(item.get("status") or ""),
                "metrics": dict(item.get("metrics") or {}),
                "paths": dict(item.get("paths") or {}),
            }
            for item in sample_summaries
        ][:500],
        "paths": {},
    }


def _evaluate_gold_suite_pair(
    *,
    pair: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    index: int,
) -> Dict[str, Any]:
    if isinstance(pair.get("report_payload"), Mapping) and isinstance(pair.get("annotation_payload"), Mapping):
        return evaluate_gold_payloads(
            report_payload=pair["report_payload"],
            annotation_payload=pair["annotation_payload"],
            thresholds=thresholds,
            report_path=Path(str(pair.get("report_path") or "")).expanduser() if pair.get("report_path") else None,
            annotation_path=Path(str(pair.get("annotation_path") or "")).expanduser() if pair.get("annotation_path") else None,
        )
    report_raw = str(pair.get("report_path") or "").strip()
    annotation_raw = str(pair.get("annotation_path") or "").strip()
    report_path = Path(report_raw).expanduser() if report_raw else Path()
    annotation_path = Path(annotation_raw).expanduser() if annotation_raw else Path()
    if not report_raw or not annotation_raw:
        return {
            "version": GOLD_EVAL_VERSION,
            "sample_id": str(pair.get("sample_id") or f"sample_{index:04d}"),
            "status": "failed",
            "paths": {
                "report_path": str(report_path),
                "annotation_path": str(annotation_path),
            },
            "metrics": {
                "expected_error_count": 0,
                "matched_confirmed_count": 0,
                "missed_count": 0,
                "critical_missed_count": 0,
                "confirmed_finding_count": 0,
                "false_positive_confirmed_count": 0,
                "confirmed_recall": 0.0,
                "confirmed_false_positive_rate": 0.0,
                "table_expected_count": 0,
                "table_matched_count": 0,
                "table_recall": 0.0,
                "expected_review_page_count": 0,
                "review_page_covered_count": 0,
            },
            "checks": [
                {
                    "name": "pair_paths_present",
                    "passed": False,
                    "severity": "fail",
                    "actual": {"report_path": str(report_path), "annotation_path": str(annotation_path)},
                    "expected": "valid report_path and annotation_path",
                }
            ],
        }
    return evaluate_gold_files(
        report_path=report_path,
        annotation_path=annotation_path,
        thresholds=thresholds,
    )


def _aggregate_gold_suite_metrics(sample_summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total_expected = 0
    total_matched = 0
    total_missed = 0
    total_critical_missed = 0
    total_confirmed = 0
    total_false_positive = 0
    total_table_expected = 0
    total_table_matched = 0
    total_expected_review_pages = 0
    total_review_pages_covered = 0
    for item in sample_summaries:
        metrics = dict(item.get("metrics") or {})
        total_expected += int(metrics.get("expected_error_count") or 0)
        total_matched += int(metrics.get("matched_confirmed_count") or 0)
        total_missed += int(metrics.get("missed_count") or 0)
        total_critical_missed += int(metrics.get("critical_missed_count") or 0)
        total_confirmed += int(metrics.get("confirmed_finding_count") or 0)
        total_false_positive += int(metrics.get("false_positive_confirmed_count") or 0)
        total_table_expected += int(metrics.get("table_expected_count") or 0)
        total_table_matched += int(metrics.get("table_matched_count") or 0)
        total_expected_review_pages += int(metrics.get("expected_review_page_count") or 0)
        total_review_pages_covered += int(metrics.get("review_page_covered_count") or 0)
    return {
        "expected_error_count": total_expected,
        "matched_confirmed_count": total_matched,
        "missed_count": total_missed,
        "critical_missed_count": total_critical_missed,
        "confirmed_finding_count": total_confirmed,
        "false_positive_confirmed_count": total_false_positive,
        "confirmed_recall": round(total_matched / total_expected, 4) if total_expected else 1.0,
        "confirmed_false_positive_rate": round(total_false_positive / total_confirmed, 4) if total_confirmed else 0.0,
        "table_expected_count": total_table_expected,
        "table_matched_count": total_table_matched,
        "table_recall": round(total_table_matched / total_table_expected, 4) if total_table_expected else 1.0,
        "expected_review_page_count": total_expected_review_pages,
        "review_page_covered_count": total_review_pages_covered,
        "review_page_coverage_ratio": round(total_review_pages_covered / total_expected_review_pages, 4)
        if total_expected_review_pages
        else 1.0,
    }


def _suite_checks(
    *,
    metrics: Mapping[str, Any],
    sample_thresholds: Mapping[str, Any],
    suite_thresholds: Mapping[str, Any],
    failed_sample_count: int,
) -> List[Dict[str, Any]]:
    return [
        {
            "name": "max_failed_samples",
            "passed": int(failed_sample_count) <= int(suite_thresholds["max_failed_samples"]),
            "severity": "fail",
            "actual": failed_sample_count,
            "expected": f"<={suite_thresholds['max_failed_samples']}",
        },
        {
            "name": "suite_confirmed_recall",
            "passed": float(metrics["confirmed_recall"]) >= float(sample_thresholds["min_confirmed_recall"]),
            "severity": "fail",
            "actual": metrics["confirmed_recall"],
            "expected": f">={sample_thresholds['min_confirmed_recall']}",
        },
        {
            "name": "suite_confirmed_false_positive_rate",
            "passed": float(metrics["confirmed_false_positive_rate"]) <= float(sample_thresholds["max_confirmed_false_positive_rate"]),
            "severity": "fail",
            "actual": metrics["confirmed_false_positive_rate"],
            "expected": f"<={sample_thresholds['max_confirmed_false_positive_rate']}",
        },
        {
            "name": "suite_critical_missed_count",
            "passed": int(metrics["critical_missed_count"]) <= int(sample_thresholds["max_critical_missed"]),
            "severity": "fail",
            "actual": metrics["critical_missed_count"],
            "expected": f"<={sample_thresholds['max_critical_missed']}",
        },
        {
            "name": "review_page_coverage_ratio",
            "passed": float(metrics["review_page_coverage_ratio"]) >= float(suite_thresholds["min_review_page_coverage_ratio"]),
            "severity": "fail",
            "actual": metrics["review_page_coverage_ratio"],
            "expected": f">={suite_thresholds['min_review_page_coverage_ratio']}",
        },
    ]


def _resolve_manifest_entry_path(raw: str, manifest_dir: Path) -> Path:
    if not raw:
        return Path()
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return (manifest_dir / path).resolve()


def _expected_errors(annotation_payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = annotation_payload.get("expected_errors")
    if not isinstance(rows, list):
        return []
    result: List[Dict[str, Any]] = []
    for index, item in enumerate(rows, start=1):
        if not isinstance(item, Mapping):
            continue
        row = dict(item)
        row.setdefault("id", f"expected_{index:04d}")
        row["page_no"] = _safe_int(row.get("page_no"))
        result.append(row)
    return result


def _findings(report_payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    findings = report_payload.get("findings")
    if isinstance(findings, list):
        rows = [dict(item) for item in findings if isinstance(item, Mapping)]
        if rows:
            return rows
    corrections = report_payload.get("corrections")
    if not isinstance(corrections, list):
        return []
    rows: List[Dict[str, Any]] = []
    for item in corrections:
        if not isinstance(item, Mapping):
            continue
        text = str(item.get("comment_text") or "")
        confirmed = "确认错误" in text or "必须改为" in text
        rows.append(
            {
                "id": item.get("id"),
                "status": "confirmed_error" if confirmed else "suspected_error",
                "category": "substitution",
                "page_no": item.get("page_no"),
                "wps_text": item.get("old_text"),
                "suggested_text": item.get("new_text"),
                "reason": item.get("reason") or text.splitlines()[0] if text else "",
            }
        )
    return rows


def _best_match(
    *,
    expected: Mapping[str, Any],
    findings: Sequence[Mapping[str, Any]],
    used_finding_ids: set[str],
) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    for finding in findings:
        finding_id = str(finding.get("id") or "")
        if finding_id and finding_id in used_finding_ids:
            continue
        if _safe_int(finding.get("page_no")) != _safe_int(expected.get("page_no")):
            continue
        score = _match_score(expected=expected, finding=finding)
        if score < 0.82:
            continue
        if best is None or score > best["score"]:
            best = {"finding": dict(finding), "score": score}
    return best


def _match_score(*, expected: Mapping[str, Any], finding: Mapping[str, Any]) -> float:
    expected_wps = _compact(expected.get("wps_text"))
    expected_text = _compact(expected.get("expected_text") or expected.get("suggested_text"))
    finding_wps = _compact(finding.get("wps_text"))
    finding_text = _compact(finding.get("suggested_text"))
    old_score = _text_score(expected_wps, finding_wps)
    new_score = _text_score(expected_text, finding_text)
    return (old_score * 0.45) + (new_score * 0.55)


def _text_score(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if left in right or right in left:
        return 0.92
    return float(similarity(left, right))


def _review_pages_covered(
    *,
    report_payload: Mapping[str, Any],
    findings: Sequence[Mapping[str, Any]],
    expected_pages: set[int],
) -> set[int]:
    covered = {
        _safe_int(item.get("page_no"))
        for item in findings
        if _safe_int(item.get("page_no")) in expected_pages
    }
    for item in report_payload.get("coverage_review_tasks", []) or []:
        if isinstance(item, Mapping) and _safe_int(item.get("page_no")) in expected_pages:
            covered.add(_safe_int(item.get("page_no")))
    product = report_payload.get("product_report")
    if isinstance(product, Mapping):
        for item in product.get("human_review_queue", []) or []:
            if isinstance(item, Mapping) and _safe_int(item.get("page_no")) in expected_pages:
                covered.add(_safe_int(item.get("page_no")))
    return {page for page in covered if page > 0}


def _checks(*, metrics: Mapping[str, Any], thresholds: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "name": "confirmed_recall",
            "passed": float(metrics["confirmed_recall"]) >= float(thresholds["min_confirmed_recall"]),
            "severity": "fail",
            "actual": metrics["confirmed_recall"],
            "expected": f">={thresholds['min_confirmed_recall']}",
        },
        {
            "name": "confirmed_false_positive_rate",
            "passed": float(metrics["confirmed_false_positive_rate"]) <= float(thresholds["max_confirmed_false_positive_rate"]),
            "severity": "fail",
            "actual": metrics["confirmed_false_positive_rate"],
            "expected": f"<={thresholds['max_confirmed_false_positive_rate']}",
        },
        {
            "name": "critical_missed_count",
            "passed": int(metrics["critical_missed_count"]) <= int(thresholds["max_critical_missed"]),
            "severity": "fail",
            "actual": metrics["critical_missed_count"],
            "expected": f"<={thresholds['max_critical_missed']}",
        },
    ]


def _expected_issue(item: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "id": item.get("id"),
        "page_no": item.get("page_no"),
        "category": item.get("category"),
        "critical": bool(item.get("critical")),
        "wps_text": item.get("wps_text"),
        "expected_text": item.get("expected_text") or item.get("suggested_text"),
    }


def _finding_issue(item: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "id": item.get("id"),
        "page_no": item.get("page_no"),
        "category": item.get("category"),
        "wps_text": item.get("wps_text"),
        "suggested_text": item.get("suggested_text"),
        "reason": item.get("reason"),
    }


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", normalize_text(str(value or "")))


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0
