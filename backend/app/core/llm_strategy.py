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
    review_entity_limit: int
    prefer_llm_first: bool
    high_risk_block_limit: int = 0
    specialized_passes: tuple[str, ...] = ()
    specialized_pass_limit: int = 0
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
    label="4B 平衡速度策略",
    description=(
        "面向本地弱设备的 4B 平衡方案。保留必要的分块召回、简称定义回扫和专项复查，"
        "但控制总轮次，避免短文也进入过长的串行分析。"
    ),
    chunk_target_size=1650,
    chunk_overlap=220,
    extract_num_predict=896,
    recall_num_predict=768,
    classify_num_predict=384,
    targeted_recall_passes=2,
    recall_snippet_size=1700,
    review_text_limit=8800,
    resolution_num_predict=960,
    focus_line_limit=24,
    review_entity_limit=24,
    prefer_llm_first=True,
    high_risk_block_limit=6,
    specialized_passes=(
        "person",
        "organization",
        "location",
        "account_identifier",
    ),
    specialized_pass_limit=6,
    complexity_extra_passes=1,
    enable_definition_recall=True,
    enable_residual_scan=True,
)

# Keep the old constant names as compatibility aliases for existing imports/tests.
PRECISION_4B_PROFILE = STABLE_4B_PROFILE
FAST_4B_PROFILE = STABLE_4B_PROFILE

MID_REVIEW_PROFILE = LLMStrategyProfile(
    key="review_mid_14b",
    label="14B 片段补充审查策略",
    description=(
        "面向 qwen3:14b 等中档模型的补充审查路线。"
        "不做全文长轮次推理，只接收规则和低内存主识别后的高风险片段，"
        "用于补漏、拒绝误识别、归并别名和标注角色。"
    ),
    chunk_target_size=1900,
    chunk_overlap=260,
    extract_num_predict=1050,
    recall_num_predict=860,
    classify_num_predict=420,
    targeted_recall_passes=2,
    recall_snippet_size=1900,
    review_text_limit=12000,
    resolution_num_predict=1120,
    focus_line_limit=32,
    review_entity_limit=32,
    prefer_llm_first=False,
    high_risk_block_limit=8,
    specialized_passes=(
        "person",
        "organization",
        "location",
        "account_identifier",
        "litigation_role",
    ),
    specialized_pass_limit=8,
    complexity_extra_passes=1,
    enable_definition_recall=True,
    enable_residual_scan=True,
)

REVIEW_27B_PROFILE = LLMStrategyProfile(
    key="review_27b",
    label="27B 精审策略",
    description=(
        "面向高精度识别的 27B 路线。保持分块和重点片段汇总，但显著提高专项召回、"
        "上下文复审和高风险块覆盖能力，适合手动触发的精查任务。"
    ),
    chunk_target_size=2200,
    chunk_overlap=320,
    extract_num_predict=1400,
    recall_num_predict=1180,
    classify_num_predict=512,
    targeted_recall_passes=4,
    recall_snippet_size=2200,
    review_text_limit=18000,
    resolution_num_predict=1500,
    focus_line_limit=40,
    review_entity_limit=42,
    prefer_llm_first=True,
    high_risk_block_limit=12,
    specialized_passes=(
        "person",
        "organization",
        "location",
        "account_identifier",
        "signature",
        "litigation_role",
    ),
    specialized_pass_limit=12,
    complexity_extra_passes=2,
    enable_definition_recall=True,
    enable_residual_scan=True,
)

def parse_model_size_in_b(model_name: str | None) -> float:
    normalized = str(model_name or "").lower()
    match = re.search(r":(\d+(?:\.\d+)?)b", normalized)
    return float(match.group(1)) if match else 0.0


def get_llm_strategy_profile(model_name: str | None) -> LLMStrategyProfile:
    size_in_b = parse_model_size_in_b(model_name)
    normalized = str(model_name or "").lower()
    if size_in_b >= 20 or ":27b" in normalized:
        return REVIEW_27B_PROFILE
    if normalized == "qwen3:8b":
        return STABLE_4B_PROFILE
    if 10 <= size_in_b < 20:
        return MID_REVIEW_PROFILE
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

    if strategy.key == "review_27b":
        if normalized_length <= 1600 and normalized_lines <= 45:
            return LLMRuntimeProfile(
                strategy=replace(
                    strategy,
                    chunk_target_size=max(strategy.chunk_target_size, 2400),
                    chunk_overlap=min(strategy.chunk_overlap, 180),
                    extract_num_predict=min(strategy.extract_num_predict, 720),
                    recall_num_predict=min(strategy.recall_num_predict, 420),
                    classify_num_predict=min(strategy.classify_num_predict, 192),
                    targeted_recall_passes=0,
                    recall_snippet_size=min(strategy.recall_snippet_size, 1200),
                    review_text_limit=min(strategy.review_text_limit, 6000),
                    resolution_num_predict=min(strategy.resolution_num_predict, 960),
                    focus_line_limit=min(strategy.focus_line_limit, 22),
                    review_entity_limit=min(strategy.review_entity_limit, 18),
                    prefer_llm_first=False,
                    high_risk_block_limit=1,
                    specialized_pass_limit=1,
                    complexity_extra_passes=0,
                ),
                budget_tier="review_compact",
                budget_label="27B短文精审",
            )

        if normalized_length <= 8000 and normalized_lines <= 220:
            return LLMRuntimeProfile(
                strategy=replace(
                    strategy,
                    extract_num_predict=min(strategy.extract_num_predict, 980),
                    recall_num_predict=min(strategy.recall_num_predict, 620),
                    classify_num_predict=min(strategy.classify_num_predict, 320),
                    targeted_recall_passes=0,
                    recall_snippet_size=min(strategy.recall_snippet_size, 1600),
                    review_text_limit=min(strategy.review_text_limit, 12000),
                    resolution_num_predict=min(strategy.resolution_num_predict, 1200),
                    focus_line_limit=min(strategy.focus_line_limit, 28),
                    review_entity_limit=min(strategy.review_entity_limit, 28),
                    prefer_llm_first=False,
                    high_risk_block_limit=min(strategy.high_risk_block_limit, 4),
                    specialized_pass_limit=min(strategy.specialized_pass_limit, 3),
                    complexity_extra_passes=min(strategy.complexity_extra_passes, 1),
                ),
                budget_tier="review_balanced",
                budget_label="27B定向精审",
            )

    if strategy.key == "review_mid_14b":
        if normalized_length <= 1600 and normalized_lines <= 45:
            return LLMRuntimeProfile(
                strategy=replace(
                    strategy,
                    chunk_target_size=max(strategy.chunk_target_size, 2200),
                    chunk_overlap=min(strategy.chunk_overlap, 180),
                    extract_num_predict=min(strategy.extract_num_predict, 640),
                    recall_num_predict=min(strategy.recall_num_predict, 380),
                    classify_num_predict=min(strategy.classify_num_predict, 192),
                    targeted_recall_passes=0,
                    recall_snippet_size=min(strategy.recall_snippet_size, 1200),
                    review_text_limit=min(strategy.review_text_limit, 5000),
                    resolution_num_predict=min(strategy.resolution_num_predict, 860),
                    focus_line_limit=min(strategy.focus_line_limit, 20),
                    review_entity_limit=min(strategy.review_entity_limit, 18),
                    high_risk_block_limit=2,
                    specialized_pass_limit=1,
                    complexity_extra_passes=0,
                    prefer_llm_first=False,
                ),
                budget_tier="mid_review_compact",
                budget_label="14B短片段主审查",
            )

        if normalized_length <= 8000 and normalized_lines <= 220:
            return LLMRuntimeProfile(
                strategy=replace(
                    strategy,
                    extract_num_predict=min(strategy.extract_num_predict, 820),
                    recall_num_predict=min(strategy.recall_num_predict, 540),
                    classify_num_predict=min(strategy.classify_num_predict, 300),
                    targeted_recall_passes=0,
                    recall_snippet_size=min(strategy.recall_snippet_size, 1500),
                    review_text_limit=min(strategy.review_text_limit, 9500),
                    resolution_num_predict=min(strategy.resolution_num_predict, 980),
                    focus_line_limit=min(strategy.focus_line_limit, 26),
                    review_entity_limit=min(strategy.review_entity_limit, 26),
                    high_risk_block_limit=min(strategy.high_risk_block_limit, 5),
                    specialized_pass_limit=min(strategy.specialized_pass_limit, 4),
                    complexity_extra_passes=min(strategy.complexity_extra_passes, 1),
                    prefer_llm_first=False,
                ),
                budget_tier="mid_review_balanced",
                budget_label="14B高风险片段审查",
            )

    if normalized_length >= 18000 or normalized_lines >= 420:
        return LLMRuntimeProfile(
            strategy=replace(
                strategy,
                chunk_target_size=1000,
                chunk_overlap=max(strategy.chunk_overlap, 280),
                targeted_recall_passes=max(strategy.targeted_recall_passes, 4),
                recall_snippet_size=min(strategy.recall_snippet_size, 1200),
                focus_line_limit=max(strategy.focus_line_limit, 34),
                review_entity_limit=max(strategy.review_entity_limit, 36),
                high_risk_block_limit=max(strategy.high_risk_block_limit, 10),
                specialized_pass_limit=max(strategy.specialized_pass_limit, 10),
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
                targeted_recall_passes=max(strategy.targeted_recall_passes, 3),
                recall_snippet_size=min(strategy.recall_snippet_size, 1350),
                focus_line_limit=max(strategy.focus_line_limit, 32),
                review_entity_limit=max(strategy.review_entity_limit, 32),
                high_risk_block_limit=max(strategy.high_risk_block_limit, 8),
                specialized_pass_limit=max(strategy.specialized_pass_limit, 8),
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
                targeted_recall_passes=max(strategy.targeted_recall_passes, 3),
                recall_snippet_size=min(strategy.recall_snippet_size, 1500),
                focus_line_limit=max(strategy.focus_line_limit, 30),
                review_entity_limit=max(strategy.review_entity_limit, 28),
                high_risk_block_limit=max(strategy.high_risk_block_limit, 7),
                specialized_pass_limit=max(strategy.specialized_pass_limit, 7),
            ),
            budget_tier="medium",
            budget_label="中长文档稳态分块",
        )

    return LLMRuntimeProfile(
        strategy=strategy,
        budget_tier="standard",
        budget_label="标准精度模式",
    )
