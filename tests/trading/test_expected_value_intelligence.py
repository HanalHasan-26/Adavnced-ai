"""
Tests for the Expected Value Intelligence Engine.
"""

import pytest

from app.trading.risk.expected_value_intelligence import (
    ExpectedValueDecision,
    ExpectedValueIntelligenceEngine,
    ExpectedValueIntelligenceError,
    ExpectedValueReasonType,
)


def test_positive_expected_value_is_allowed():
    """Positive gross and net EV should be allowed."""

    engine = ExpectedValueIntelligenceEngine()

    result = engine.analyze(
        win_rate=0.55,
        average_win_r=2.0,
        average_loss_r=1.0,
    )

    assert result.decision == ExpectedValueDecision.ALLOW
    assert result.allowed
    assert result.valid
    assert result.ready

    assert result.expected_value_r == pytest.approx(0.65)
    assert result.net_expected_value_r == pytest.approx(0.65)


def test_loss_rate_is_complement_of_win_rate():
    """Loss rate should equal one minus win rate."""

    engine = ExpectedValueIntelligenceEngine()

    result = engine.analyze(
        win_rate=0.60,
        average_win_r=2.0,
        average_loss_r=1.0,
    )

    assert result.loss_rate == pytest.approx(0.40)


def test_break_even_win_rate():
    """Break-even probability should be calculated correctly."""

    engine = ExpectedValueIntelligenceEngine()

    result = engine.analyze(
        win_rate=0.50,
        average_win_r=2.0,
        average_loss_r=1.0,
    )

    assert result.break_even_win_rate == pytest.approx(
        1.0 / 3.0
    )


def test_exact_zero_ev_is_allowed_by_default():
    """
    Mathematically zero EV must be represented as exactly 0.0.

    This protects against floating-point residuals such as:
        -1.1102230246251565e-16
    """

    engine = ExpectedValueIntelligenceEngine()

    result = engine.analyze(
        win_rate=1.0 / 3.0,
        average_win_r=2.0,
        average_loss_r=1.0,
    )

    assert result.expected_value_r == 0.0
    assert result.net_expected_value_r == 0.0
    assert result.decision == ExpectedValueDecision.ALLOW
    assert result.ready


def test_negative_ev_is_blocked():
    """Negative gross and net EV should be blocked."""

    engine = ExpectedValueIntelligenceEngine()

    result = engine.analyze(
        win_rate=0.30,
        average_win_r=1.0,
        average_loss_r=1.0,
    )

    assert result.expected_value_r == pytest.approx(-0.40)
    assert result.net_expected_value_r == pytest.approx(-0.40)
    assert result.decision == ExpectedValueDecision.BLOCK
    assert result.blocked
    assert result.ready is False

    assert any(
        reason.reason_type
        == ExpectedValueReasonType.NET_EXPECTED_VALUE_TOO_LOW
        for reason in result.reasons
    )


def test_custom_minimum_ev_is_respected():
    """A custom net EV threshold should be respected."""

    engine = ExpectedValueIntelligenceEngine()

    result = engine.analyze(
        win_rate=0.55,
        average_win_r=2.0,
        average_loss_r=1.0,
        minimum_net_expected_value_r=0.70,
    )

    assert result.expected_value_r == pytest.approx(0.65)
    assert result.net_expected_value_r == pytest.approx(0.65)
    assert result.decision == ExpectedValueDecision.BLOCK


def test_custom_minimum_ev_can_be_negative():
    """A negative minimum net EV policy is accepted."""

    engine = ExpectedValueIntelligenceEngine(
        minimum_net_expected_value_r=-0.50,
    )

    result = engine.analyze(
        win_rate=0.40,
        average_win_r=1.0,
        average_loss_r=1.0,
    )

    assert result.expected_value_r == pytest.approx(-0.20)
    assert result.net_expected_value_r == pytest.approx(-0.20)
    assert result.decision == ExpectedValueDecision.ALLOW


def test_transaction_costs_are_calculated():
    """Transaction costs should be summed correctly."""

    engine = ExpectedValueIntelligenceEngine()

    result = engine.analyze(
        win_rate=0.60,
        average_win_r=2.0,
        average_loss_r=1.0,
        spread_cost_r=0.05,
        commission_cost_r=0.10,
        slippage_cost_r=0.05,
    )

    assert result.expected_value_r == pytest.approx(0.80)

    assert result.spread_cost_r == pytest.approx(0.05)
    assert result.commission_cost_r == pytest.approx(0.10)
    assert result.slippage_cost_r == pytest.approx(0.05)

    assert result.total_cost_r == pytest.approx(0.20)
    assert result.net_expected_value_r == pytest.approx(0.60)

    assert result.decision == ExpectedValueDecision.ALLOW


def test_transaction_costs_reduce_ev():
    """Transaction costs must reduce gross EV."""

    engine = ExpectedValueIntelligenceEngine()

    result = engine.analyze(
        win_rate=0.60,
        average_win_r=2.0,
        average_loss_r=1.0,
        spread_cost_r=0.10,
        commission_cost_r=0.10,
        slippage_cost_r=0.10,
    )

    assert result.expected_value_r == pytest.approx(0.80)
    assert result.total_cost_r == pytest.approx(0.30)
    assert result.net_expected_value_r == pytest.approx(0.50)


def test_costs_can_turn_positive_gross_ev_into_negative_net_ev():
    """High costs must be able to block a strategy."""

    engine = ExpectedValueIntelligenceEngine()

    result = engine.analyze(
        win_rate=0.55,
        average_win_r=2.0,
        average_loss_r=1.0,
        spread_cost_r=0.30,
        commission_cost_r=0.20,
        slippage_cost_r=0.20,
    )

    assert result.expected_value_r == pytest.approx(0.65)
    assert result.total_cost_r == pytest.approx(0.70)
    assert result.net_expected_value_r == pytest.approx(-0.05)
    assert result.decision == ExpectedValueDecision.BLOCK


def test_exact_net_ev_boundary_is_allowed():
    """Net EV exactly at the minimum should be allowed."""

    engine = ExpectedValueIntelligenceEngine(
        minimum_net_expected_value_r=0.50,
    )

    result = engine.analyze(
        win_rate=0.60,
        average_win_r=2.0,
        average_loss_r=1.0,
        spread_cost_r=0.10,
        commission_cost_r=0.10,
        slippage_cost_r=0.10,
    )

    assert result.net_expected_value_r == pytest.approx(0.50)
    assert result.decision == ExpectedValueDecision.ALLOW


def test_zero_transaction_costs_preserve_previous_ev():
    """Zero costs should preserve previous EV behavior."""

    engine = ExpectedValueIntelligenceEngine()

    result = engine.analyze(
        win_rate=0.55,
        average_win_r=2.0,
        average_loss_r=1.0,
        spread_cost_r=0.0,
        commission_cost_r=0.0,
        slippage_cost_r=0.0,
    )

    assert result.expected_value_r == pytest.approx(0.65)
    assert result.total_cost_r == pytest.approx(0.0)
    assert result.net_expected_value_r == pytest.approx(0.65)
    assert result.decision == ExpectedValueDecision.ALLOW


@pytest.mark.parametrize(
    "win_rate",
    [-0.01, 1.01, float("nan"), float("inf"), float("-inf")],
)
def test_invalid_win_rate_is_blocked(win_rate):
    """Invalid win rates should be blocked."""

    engine = ExpectedValueIntelligenceEngine()

    result = engine.analyze(
        win_rate=win_rate,
        average_win_r=2.0,
        average_loss_r=1.0,
    )

    assert result.decision == ExpectedValueDecision.BLOCK
    assert result.valid is False
    assert result.ready is False
    assert result.reasons[0].reason_type == (
        ExpectedValueReasonType.INVALID_WIN_RATE
    )


@pytest.mark.parametrize(
    "average_win_r",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_invalid_average_win_is_blocked(average_win_r):
    """Average win must be positive and finite."""

    engine = ExpectedValueIntelligenceEngine()

    result = engine.analyze(
        win_rate=0.55,
        average_win_r=average_win_r,
        average_loss_r=1.0,
    )

    assert result.decision == ExpectedValueDecision.BLOCK
    assert result.valid is False
    assert result.reasons[0].reason_type == (
        ExpectedValueReasonType.INVALID_AVERAGE_WIN
    )


@pytest.mark.parametrize(
    "average_loss_r",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_invalid_average_loss_is_blocked(average_loss_r):
    """Average loss must be positive and finite."""

    engine = ExpectedValueIntelligenceEngine()

    result = engine.analyze(
        win_rate=0.55,
        average_win_r=2.0,
        average_loss_r=average_loss_r,
    )

    assert result.decision == ExpectedValueDecision.BLOCK
    assert result.valid is False
    assert result.reasons[0].reason_type == (
        ExpectedValueReasonType.INVALID_AVERAGE_LOSS
    )


@pytest.mark.parametrize(
    "minimum_ev",
    [float("nan"), float("inf"), float("-inf")],
)
def test_invalid_minimum_ev_is_rejected(minimum_ev):
    """Invalid constructor EV thresholds must be rejected."""

    with pytest.raises(ExpectedValueIntelligenceError):
        ExpectedValueIntelligenceEngine(
            minimum_expected_value_r=minimum_ev,
        )


@pytest.mark.parametrize(
    "minimum_net_ev",
    [float("nan"), float("inf"), float("-inf")],
)
def test_invalid_minimum_net_ev_is_rejected(minimum_net_ev):
    """Invalid net EV constructor thresholds must be rejected."""

    with pytest.raises(ExpectedValueIntelligenceError):
        ExpectedValueIntelligenceEngine(
            minimum_net_expected_value_r=minimum_net_ev,
        )


@pytest.mark.parametrize(
    "minimum_ev",
    [float("nan"), float("inf"), float("-inf")],
)
def test_invalid_per_analysis_minimum_ev_is_blocked(minimum_ev):
    """Invalid per-analysis gross EV thresholds must be blocked."""

    engine = ExpectedValueIntelligenceEngine()

    result = engine.analyze(
        win_rate=0.55,
        average_win_r=2.0,
        average_loss_r=1.0,
        minimum_expected_value_r=minimum_ev,
    )

    assert result.decision == ExpectedValueDecision.BLOCK
    assert result.valid is False
    assert result.reasons[0].reason_type == (
        ExpectedValueReasonType.INVALID_MINIMUM_EXPECTED_VALUE
    )


@pytest.mark.parametrize(
    "minimum_net_ev",
    [float("nan"), float("inf"), float("-inf")],
)
def test_invalid_per_analysis_minimum_net_ev_is_blocked(
    minimum_net_ev,
):
    """Invalid per-analysis net EV thresholds must be blocked."""

    engine = ExpectedValueIntelligenceEngine()

    result = engine.analyze(
        win_rate=0.55,
        average_win_r=2.0,
        average_loss_r=1.0,
        minimum_net_expected_value_r=minimum_net_ev,
    )

    assert result.decision == ExpectedValueDecision.BLOCK
    assert result.valid is False
    assert result.reasons[0].reason_type == (
        ExpectedValueReasonType.INVALID_MINIMUM_EXPECTED_VALUE
    )


@pytest.mark.parametrize(
    "cost_field",
    [
        "spread_cost_r",
        "commission_cost_r",
        "slippage_cost_r",
    ],
)
def test_negative_transaction_cost_is_blocked(cost_field):
    """Negative transaction costs must never be accepted."""

    engine = ExpectedValueIntelligenceEngine()

    kwargs = {
        "spread_cost_r": 0.0,
        "commission_cost_r": 0.0,
        "slippage_cost_r": 0.0,
    }

    kwargs[cost_field] = -0.01

    result = engine.analyze(
        win_rate=0.55,
        average_win_r=2.0,
        average_loss_r=1.0,
        **kwargs,
    )

    assert result.decision == ExpectedValueDecision.BLOCK
    assert result.valid is False


@pytest.mark.parametrize(
    "cost_field",
    [
        "spread_cost_r",
        "commission_cost_r",
        "slippage_cost_r",
    ],
)
@pytest.mark.parametrize(
    "invalid_value",
    [float("nan"), float("inf"), float("-inf")],
)
def test_non_finite_transaction_cost_is_blocked(
    cost_field,
    invalid_value,
):
    """Non-finite transaction costs must be blocked."""

    engine = ExpectedValueIntelligenceEngine()

    kwargs = {
        "spread_cost_r": 0.0,
        "commission_cost_r": 0.0,
        "slippage_cost_r": 0.0,
    }

    kwargs[cost_field] = invalid_value

    result = engine.analyze(
        win_rate=0.55,
        average_win_r=2.0,
        average_loss_r=1.0,
        **kwargs,
    )

    assert result.decision == ExpectedValueDecision.BLOCK
    assert result.valid is False


def test_extreme_win_rates_are_valid():
    """0% and 100% win rates remain mathematically valid."""

    engine = ExpectedValueIntelligenceEngine()

    zero = engine.analyze(
        win_rate=0.0,
        average_win_r=2.0,
        average_loss_r=1.0,
    )

    one = engine.analyze(
        win_rate=1.0,
        average_win_r=2.0,
        average_loss_r=1.0,
    )

    assert zero.valid
    assert one.valid

    assert zero.loss_rate == pytest.approx(1.0)
    assert one.loss_rate == pytest.approx(0.0)


def test_zero_percent_win_rate_produces_negative_ev():
    """A strategy that never wins has negative expectancy."""

    engine = ExpectedValueIntelligenceEngine()

    result = engine.analyze(
        win_rate=0.0,
        average_win_r=2.0,
        average_loss_r=1.0,
    )

    assert result.expected_value_r == pytest.approx(-1.0)
    assert result.net_expected_value_r == pytest.approx(-1.0)
    assert result.decision == ExpectedValueDecision.BLOCK


def test_one_hundred_percent_win_rate_produces_average_win_ev():
    """A 100% win rate produces EV equal to average win."""

    engine = ExpectedValueIntelligenceEngine()

    result = engine.analyze(
        win_rate=1.0,
        average_win_r=2.0,
        average_loss_r=1.0,
    )

    assert result.expected_value_r == pytest.approx(2.0)
    assert result.net_expected_value_r == pytest.approx(2.0)
    assert result.decision == ExpectedValueDecision.ALLOW


def test_ev_formula_is_correct():
    """Verify the exact gross EV calculation."""

    engine = ExpectedValueIntelligenceEngine()

    result = engine.analyze(
        win_rate=0.60,
        average_win_r=1.5,
        average_loss_r=1.0,
    )

    expected = (
        (0.60 * 1.5)
        - (0.40 * 1.0)
    )

    assert result.expected_value_r == pytest.approx(expected)


def test_net_ev_formula_is_correct():
    """Verify the exact net EV calculation."""

    engine = ExpectedValueIntelligenceEngine()

    result = engine.analyze(
        win_rate=0.60,
        average_win_r=1.5,
        average_loss_r=1.0,
        spread_cost_r=0.05,
        commission_cost_r=0.10,
        slippage_cost_r=0.05,
    )

    gross_ev = (
        (0.60 * 1.5)
        - (0.40 * 1.0)
    )

    total_cost = (
        0.05
        + 0.10
        + 0.05
    )

    expected_net_ev = gross_ev - total_cost

    assert result.expected_value_r == pytest.approx(gross_ev)
    assert result.total_cost_r == pytest.approx(total_cost)
    assert result.net_expected_value_r == pytest.approx(
        expected_net_ev
    )


def test_alias_properties_match():
    """Compatibility aliases should expose the same values."""

    engine = ExpectedValueIntelligenceEngine()

    result = engine.analyze(
        win_rate=0.55,
        average_win_r=2.0,
        average_loss_r=1.0,
        spread_cost_r=0.05,
    )

    assert result.ev == result.expected_value_r
    assert result.expected_value == result.expected_value_r

    assert result.net_ev == result.net_expected_value_r
    assert result.net_expected_value == result.net_expected_value_r

    assert result.total_cost == result.total_cost_r

    assert (
        result.breakeven_win_rate
        == result.break_even_win_rate
    )

    assert result.is_allowed == result.allowed
    assert result.is_blocked == result.blocked


def test_timestamp_is_timezone_aware():
    """Audit timestamp should be timezone-aware."""

    engine = ExpectedValueIntelligenceEngine()

    result = engine.analyze(
        win_rate=0.55,
        average_win_r=2.0,
        average_loss_r=1.0,
    )

    assert result.timestamp.tzinfo is not None


def test_result_is_immutable():
    """Frozen dataclass should prevent accidental mutation."""

    engine = ExpectedValueIntelligenceEngine()

    result = engine.analyze(
        win_rate=0.55,
        average_win_r=2.0,
        average_loss_r=1.0,
    )

    with pytest.raises(AttributeError):
        result.win_rate = 0.60


def test_costs_are_explicitly_reported():
    """Every transaction-cost component must be exposed."""

    engine = ExpectedValueIntelligenceEngine()

    result = engine.analyze(
        win_rate=0.60,
        average_win_r=2.0,
        average_loss_r=1.0,
        spread_cost_r=0.02,
        commission_cost_r=0.03,
        slippage_cost_r=0.04,
    )

    assert result.spread_cost_r == pytest.approx(0.02)
    assert result.commission_cost_r == pytest.approx(0.03)
    assert result.slippage_cost_r == pytest.approx(0.04)
    assert result.total_cost_r == pytest.approx(0.09)