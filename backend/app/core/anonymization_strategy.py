"""Profiles for replacement wording strategies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnonymizationStrategyProfile:
    key: str
    label: str
    description: str


OFFICIAL_STYLE_PROFILE = AnonymizationStrategyProfile(
    key="official",
    label="官方局部某化",
    description="主体名称尽量保留原有阅读感，只对关键部分做“某化”处理，适合正式材料。",
)

SERIAL_ROLE_PROFILE = AnonymizationStrategyProfile(
    key="serial_roles",
    label="甲乙丙主体策略",
    description="人物和主体优先改成甲乙丙类序号称谓，适合快速区分不同参与方。",
)

SYMBOLIC_CODE_PROFILE = AnonymizationStrategyProfile(
    key="symbolic_codes",
    label="ABCD / 希腊 / 甲乙丙编码",
    description="人名改成 a/b/c/d，机构改成 alpha/beta/gamma，地名改成甲乙丙，并对同一主体保持稳定编码。",
)

_STRATEGY_MAP = {
    OFFICIAL_STYLE_PROFILE.key: OFFICIAL_STYLE_PROFILE,
    SERIAL_ROLE_PROFILE.key: SERIAL_ROLE_PROFILE,
    SYMBOLIC_CODE_PROFILE.key: SYMBOLIC_CODE_PROFILE,
}

DEFAULT_ANONYMIZATION_STRATEGY = OFFICIAL_STYLE_PROFILE.key


def get_anonymization_strategy_profile(
    strategy_key: str | None,
) -> AnonymizationStrategyProfile:
    normalized = str(strategy_key or "").strip().lower()
    return _STRATEGY_MAP.get(normalized, OFFICIAL_STYLE_PROFILE)
