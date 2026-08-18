"""Lifecycle thresholds for the high-turnover market-wide US-equity desk."""

from __future__ import annotations

from quant.lifecycle import PromotionRequirements


def marketwide_equity_promotion_requirements() -> PromotionRequirements:
    """Return one shared fail-closed requirement set for CLI and runtime.

    The desk stops at PAPER by policy, but later-rung values remain conservative so a
    future caller cannot accidentally inherit weaker generic defaults merely by using
    the same LifecycleManager.
    """
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
