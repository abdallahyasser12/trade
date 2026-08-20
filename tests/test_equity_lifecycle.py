from quant.equity_lifecycle import marketwide_equity_promotion_requirements


def test_marketwide_equity_promotion_requirements_are_strict_and_valid():
    requirements = marketwide_equity_promotion_requirements()
    assert requirements.validate() == []
    assert requirements.min_research_trades == 100
    assert requirements.min_shadow_trades == 100
    assert requirements.min_shadow_expectancy_r == 0.03
    assert requirements.min_shadow_profit_factor == 1.10
    assert requirements.max_shadow_drawdown_r == 5.0
    assert requirements.min_paper_trades == 200
    assert requirements.min_measured_execution_samples == 50
