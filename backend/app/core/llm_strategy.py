"""Model-aware LLM strategy profiles."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re


@dataclass(frozen=True)
class LLMStrategyProfile:
    key: str
    label: str
    description: str
    chunk_target_size: int
    chunk_overlap: int
    extract_num_predict: int
    recall_num_predict: int
    classify_num_predict: int
    targeted_recall_passes: int
    recall_snippet_size: int
    review_text_limit: int
    resolution_num_predict: int
    focus_line_limit: int
    prefer_llm_first: bool
    high_risk_block_limit: int = 0
    specialized_passes: tuple[str, ...] = ()
    complexity_extra_passes: int = 0
    enable_definition_recall: bool = False
    enable_residual_scan: bool = False


@dataclass(frozen=True)
class LLMRuntimeProfile:
    strategy: LLMStrategyProfile
    budget_tier: str
    budget_label: str


STABLE_4B_PROFILE = LLMStrategyProfile(
    key="precision_4b",
    label="4B 稳定增强策略",
    description=(
        "面向本地弱设备的 4B 稳定方案。保持高召回、多轮专项复查、简称定义回扫和残留复检，"
        "并通过更小的稳态分块来降低单次调用峰值，不用牺牲识别精度换稳定性。"
    ),
    chunk_target_size=1650,
    chunk_overlap=220,
    extract_num_predict=896,
    recall_num_predict=768,
    classify_num_predict=384,
    targeted_recall_passes=4,
    recall_snippet_size=1700,
    review_text_limit=8800,
    resolution_num_predict=2000,
    focus_line_limit=28,
    prefer_llm_first=True,
    high_risk_block_limit=14,
    specialized_passes=(
        "person",
        "organization",
        "location",
        "account_identifier",
        "relation_alias",
        "residual",
    ),
    complexity_extra_passes=3,
    enable_definition_recall=True,
    enable_residual_scan=True,
)

# Keep the old constant names as compatibility aliases for existing imports/tests.
PRECISION_4B_PROFILE = STABLE_4B_PROFILE
FAST_4B_PROFILE = STABLE_4B_PROFILE

def parse_model_size_in_b(model_name: str | None) -> float:
    normalized = str(model_name or "").lower()
    match = re.search(r":(\d+(?:\.\d+)?)b", normalized)
    return float(match.group(1)) if match else 0.0


def get_llm_strategy_profile(model_name: str | None) -> LLMStrategyProfile:
    # The runtime is now intentionally unified around the local 4B engine.
    return STABLE_4B_PROFILE


def get_runtime_llm_strategy_profile(
    model_name: str | None,
    *,
    text_length: int = 0,
    line_count: int = 0,
) -> LLMRuntimeProfile:
    strategy = get_llm_strategy_profile(model_name)
    normalized_length = max(0, int(text_length or 0))
    normalized_lines = max(0, int(line_count or 0))

    if normalized_length >= 18000 or normalized_lines >= 420:
        return LLMRuntimeProfile(
            strategy=replace(
                strategy,
                chunk_target_size=1000,
                chunk_overlap=max(strategy.chunk_overlap, 280),
                targeted_recall_passes=max(strategy.targeted_recall_passes, 6),
                recall_snippet_size=min(strategy.recall_snippet_size, 1200),
                focus_line_limit=max(strategy.focus_line_limit, 34),
                high_risk_block_limit=max(strategy.high_risk_block_limit, 20),
            ),
            budget_tier="very_large",
            budget_label="超长文档稳态分块",
        )

    if normalized_length >= 9000 or normalized_lines >= 220:
        return LLMRuntimeProfile(
            strategy=replace(
                strategy,
                chunk_target_size=1200,
                chunk_overlap=max(strategy.chunk_overlap, 260),
                targeted_recall_passes=max(strategy.targeted_recall_passes, 5),
                recall_snippet_size=min(strategy.recall_snippet_size, 1350),
                focus_line_limit=max(strategy.focus_line_limit, 32),
                high_risk_block_limit=max(strategy.high_risk_block_limit, 18),
            ),
            budget_tier="large",
            budget_label="长文档稳态分块",
        )

    if normalized_length >= 4500 or normalized_lines >= 110:
        return LLMRuntimeProfile(
            strategy=replace(
                strategy,
                chunk_target_size=1400,
                chunk_overlap=max(strategy.chunk_overlap, 240),
                targeted_recall_passes=max(strategy.targeted_recall_passes, 4),
                recall_snippet_size=min(strategy.recall_snippet_size, 1500),
                focus_line_limit=max(strategy.focus_line_limit, 30),
                high_risk_block_limit=max(strategy.high_risk_block_limit, 16),
            ),
            budget_tier="medium",
            budget_label="中长文档稳态分块",
        )

    return LLMRuntimeProfile(
        strategy=strategy,
        budget_tier="standard",
        budget_label="标准精度模式",
    )
