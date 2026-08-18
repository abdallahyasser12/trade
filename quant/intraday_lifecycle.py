"""Evidence thresholds specific to the high-velocity intraday strategy.

A strategy generating many trades has enough observations to demand stronger
forward evidence than slower swing systems. Keep these requirements centralized so
promotion CLI, broker runtime and automatic degradation cannot silently disagree.
"""

from __future__ import annotations

from quant.lifecycle import (
    DegradationMonitor,
    DegradationThresholds,
    PromotionRequirements,
)


def intraday_promotion_requirements() -> PromotionRequirements:
    return PromotionRequirements(
        min_research_trades=100,
        min_shadow_trades=100,
        min_shadow_expectancy_r=0.03,
        min_shadow_profit_factor=1.10,
        max_shadow_drawdown_r=5.0,
        min_paper_trades=200,
        min_paper_expectancy_r=0.03,
        min_paper_profit_factor=1.10,
        max_paper_drawdown_r=6.0,
        min_measured_execution_samples=50,
        min_canary_trades=100,
        min_canary_expectancy_r=0.01,
        min_canary_profit_factor=1.05,
        max_canary_drawdown_r=4.0,
        max_open_operational_defects=0,
    )


def intraday_degradation_monitor() -> DegradationMonitor:
    """React only after a real sample, but before a high-turnover edge fully dies."""
    return DegradationMonitor(
        DegradationThresholds(
            min_profit_factor=1.0,
            min_expectancy_r=0.0,
            max_drawdown_r=5.0,
            max_calibration_ece=0.15,
            max_cost_inflation_ratio=1.75,
            min_sample_trades=50,
        )
    )
