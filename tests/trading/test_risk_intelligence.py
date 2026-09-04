from math import isclose

import pytest

from app.trading.risk.risk_intelligence import (
    RiskDecision,
    RiskIntelligenceEngine,
    RiskIntelligenceError,
    RiskReasonType,
)


def make_engine() -> RiskIntelligenceEngine:
    """Create the standard test risk engine."""

    # Use the production defaults for the test suite.
    return RiskIntelligenceEngine()


def make_valid_kwargs() -> dict:
    """Create a safe baseline trade configuration."""

    # Return a conservative valid XAUUSD-style risk scenario.
    return {
        "balance": 10_000.0,
        "equity": 10_000.0,
        "entry_price": 2030.0,
        "stop_loss": 2020.0,
        "risk_percent": 1.0,
        "tick_value": 1.0,
        "contract_size": 1.0,
        "tick_size": 1.0,
        "volume_step": 0.01,
        "broker_min_position_size": 0.01,
        "broker_max_position_size": 100.0,
        "spread": 1.0,
        "commission": 5.0,
        "slippage": 1.0,
        "daily_loss": 0.0,
        "drawdown_percent": 0.0,
        "open_trades": 0,
        "consecutive_losses": 0,
        "session_allowed": True,
        "news_allowed": True,
        "correlation_allowed": True,
        "cooldown_active": False,
    }


def test_valid_trade_is_allowed() -> None:
    # Create the risk engine.
    engine = make_engine()

    # Analyze a completely valid trade.
    result = engine.analyze(
        **make_valid_kwargs(),
    )

    # The trade must be allowed.
    assert result.decision is RiskDecision.ALLOW
    assert result.allowed is True
    assert result.blocked is False

    # Verify the basic risk calculation.
    assert result.risk_percent == 1.0
    assert result.risk_amount == 100.0
    assert result.stop_distance == 10.0
    assert result.position_size == 10.0

    # Verify the success reason.
    assert any(
        reason.reason_type is RiskReasonType.RISK_ALLOWED
        for reason in result.reasons
    )


def test_default_risk_percent_is_used() -> None:
    # Create the engine.
    engine = make_engine()

    # Remove the explicit risk percentage.
    kwargs = make_valid_kwargs()
    kwargs.pop("risk_percent")

    # Analyze using the engine default.
    result = engine.analyze(**kwargs)

    # Default risk is one percent.
    assert result.risk_percent == 1.0
    assert result.risk_amount == 100.0


def test_long_stop_distance_is_calculated() -> None:
    # Create the risk engine.
    engine = make_engine()

    # Use a LONG-style stop.
    kwargs = make_valid_kwargs()
    kwargs["entry_price"] = 2030.0
    kwargs["stop_loss"] = 2025.0

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # Risk distance must be five price units.
    assert result.stop_distance == 5.0


def test_short_stop_distance_is_calculated() -> None:
    # Create the risk engine.
    engine = make_engine()

    # Use a SHORT-style stop.
    kwargs = make_valid_kwargs()
    kwargs["entry_price"] = 2030.0
    kwargs["stop_loss"] = 2035.0

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # Risk distance is direction-independent.
    assert result.stop_distance == 5.0


def test_risk_amount_uses_balance_and_risk_percent() -> None:
    # Create the engine.
    engine = make_engine()

    # Use a two-percent risk configuration.
    kwargs = make_valid_kwargs()
    kwargs["risk_percent"] = 2.0

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # Two percent of 10,000 equals 200.
    assert result.risk_amount == 200.0


def test_position_size_uses_stop_distance() -> None:
    # Create the engine.
    engine = make_engine()

    # Increase the stop distance.
    kwargs = make_valid_kwargs()
    kwargs["stop_loss"] = 2010.0

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # Risk amount remains 100, while risk per unit doubles.
    assert result.risk_amount == 100.0
    assert result.stop_distance == 20.0
    assert result.position_size == 5.0


def test_risk_percent_below_minimum_is_blocked() -> None:
    # Create the engine.
    engine = make_engine()

    # Request less than the configured minimum.
    kwargs = make_valid_kwargs()
    kwargs["risk_percent"] = 0.001

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # The risk policy must block it.
    assert result.blocked is True
    assert any(
        reason.reason_type is RiskReasonType.INVALID_RISK_PERCENT
        for reason in result.reasons
    )


def test_risk_percent_above_maximum_is_blocked() -> None:
    # Create the engine.
    engine = make_engine()

    # Request more than the configured maximum.
    kwargs = make_valid_kwargs()
    kwargs["risk_percent"] = 3.0

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # The risk policy must block it.
    assert result.blocked is True
    assert any(
        reason.reason_type is RiskReasonType.INVALID_RISK_PERCENT
        for reason in result.reasons
    )


def test_stop_distance_too_small_is_blocked() -> None:
    # Create the engine.
    engine = make_engine()

    # Make SL almost equal to entry.
    kwargs = make_valid_kwargs()
    kwargs["stop_loss"] = 2029.999

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # The trade must be blocked.
    assert result.blocked is True
    assert any(
        reason.reason_type is RiskReasonType.INVALID_STOP_DISTANCE
        for reason in result.reasons
    )


def test_position_size_too_large_is_blocked() -> None:
    # Create an engine with a very small maximum position size.
    engine = RiskIntelligenceEngine(
        maximum_position_size=5.0,
    )

    # Baseline position size is ten.
    result = engine.analyze(
        **make_valid_kwargs(),
    )

    # Ten exceeds the configured maximum.
    assert result.blocked is True
    assert any(
        reason.reason_type is RiskReasonType.POSITION_SIZE_TOO_LARGE
        for reason in result.reasons
    )


def test_position_size_too_small_is_blocked() -> None:
    # Create an engine with a large minimum position size.
    engine = RiskIntelligenceEngine(
        minimum_position_size=20.0,
    )

    # Baseline position size is ten.
    result = engine.analyze(
        **make_valid_kwargs(),
    )

    # Ten is below the configured minimum.
    assert result.blocked is True
    assert any(
        reason.reason_type is RiskReasonType.POSITION_SIZE_TOO_SMALL
        for reason in result.reasons
    )


def test_high_spread_is_blocked() -> None:
    # Create the engine.
    engine = make_engine()

    # Set spread above the configured maximum.
    kwargs = make_valid_kwargs()
    kwargs["spread"] = 6.0

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # Spread veto must block it.
    assert result.blocked is True
    assert any(
        reason.reason_type is RiskReasonType.SPREAD_TOO_HIGH
        for reason in result.reasons
    )


def test_high_commission_is_blocked() -> None:
    # Create the engine.
    engine = make_engine()

    # Set commission above the configured maximum.
    kwargs = make_valid_kwargs()
    kwargs["commission"] = 101.0

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # Commission veto must block it.
    assert result.blocked is True
    assert any(
        reason.reason_type is RiskReasonType.COMMISSION_TOO_HIGH
        for reason in result.reasons
    )


def test_high_slippage_is_blocked() -> None:
    # Create the engine.
    engine = make_engine()

    # Set slippage above the configured maximum.
    kwargs = make_valid_kwargs()
    kwargs["slippage"] = 6.0

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # Slippage veto must block it.
    assert result.blocked is True
    assert any(
        reason.reason_type is RiskReasonType.SLIPPAGE_TOO_HIGH
        for reason in result.reasons
    )


def test_daily_loss_limit_is_blocked() -> None:
    # Create the engine.
    engine = make_engine()

    # Five percent of 10,000 is the default daily-loss boundary.
    kwargs = make_valid_kwargs()
    kwargs["daily_loss"] = 500.0

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # Daily loss veto must block it.
    assert result.blocked is True
    assert any(
        reason.reason_type is RiskReasonType.DAILY_LOSS_LIMIT
        for reason in result.reasons
    )


def test_drawdown_limit_is_blocked() -> None:
    # Create the engine.
    engine = make_engine()

    # Ten percent reaches the default drawdown limit.
    kwargs = make_valid_kwargs()
    kwargs["drawdown_percent"] = 10.0

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # Drawdown veto must block it.
    assert result.blocked is True
    assert any(
        reason.reason_type is RiskReasonType.MAX_DRAWDOWN_LIMIT
        for reason in result.reasons
    )


def test_max_open_trades_is_blocked() -> None:
    # Create the engine.
    engine = make_engine()

    # Three open trades reaches the default maximum.
    kwargs = make_valid_kwargs()
    kwargs["open_trades"] = 3

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # Open-trade limit must block it.
    assert result.blocked is True
    assert any(
        reason.reason_type is RiskReasonType.MAX_OPEN_TRADES
        for reason in result.reasons
    )


def test_consecutive_loss_limit_is_blocked() -> None:
    # Create the engine.
    engine = make_engine()

    # Three consecutive losses reaches the default limit.
    kwargs = make_valid_kwargs()
    kwargs["consecutive_losses"] = 3

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # Consecutive-loss veto must block it.
    assert result.blocked is True
    assert any(
        reason.reason_type is RiskReasonType.CONSECUTIVE_LOSS_LIMIT
        for reason in result.reasons
    )


def test_cooldown_blocks_trade() -> None:
    # Create the engine.
    engine = make_engine()

    # Activate cooldown.
    kwargs = make_valid_kwargs()
    kwargs["cooldown_active"] = True

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # Cooldown must block the trade.
    assert result.blocked is True
    assert any(
        reason.reason_type is RiskReasonType.COOLDOWN_ACTIVE
        for reason in result.reasons
    )


def test_session_block_blocks_trade() -> None:
    # Create the engine.
    engine = make_engine()

    # Disable session permission.
    kwargs = make_valid_kwargs()
    kwargs["session_allowed"] = False

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # Session policy must block it.
    assert result.blocked is True
    assert any(
        reason.reason_type is RiskReasonType.SESSION_BLOCKED
        for reason in result.reasons
    )


def test_news_block_blocks_trade() -> None:
    # Create the engine.
    engine = make_engine()

    # Disable news permission.
    kwargs = make_valid_kwargs()
    kwargs["news_allowed"] = False

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # News policy must block it.
    assert result.blocked is True
    assert any(
        reason.reason_type is RiskReasonType.NEWS_BLOCKED
        for reason in result.reasons
    )


def test_correlation_block_blocks_trade() -> None:
    # Create the engine.
    engine = make_engine()

    # Disable correlation permission.
    kwargs = make_valid_kwargs()
    kwargs["correlation_allowed"] = False

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # Correlation policy must block it.
    assert result.blocked is True
    assert any(
        reason.reason_type is RiskReasonType.CORRELATION_BLOCKED
        for reason in result.reasons
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("balance", 0.0),
        ("balance", -1.0),
        ("equity", 0.0),
        ("equity", -1.0),
        ("entry_price", 0.0),
        ("entry_price", -1.0),
        ("stop_loss", 0.0),
        ("stop_loss", -1.0),
        ("tick_value", 0.0),
        ("contract_size", 0.0),
    ],
)
def test_positive_financial_inputs_reject_zero_and_negative(
    field: str,
    value: float,
) -> None:
    # Create the engine.
    engine = make_engine()

    # Modify one financial input.
    kwargs = make_valid_kwargs()
    kwargs[field] = value

    # Invalid financial prices/values must raise immediately.
    with pytest.raises(RiskIntelligenceError):
        engine.analyze(**kwargs)


@pytest.mark.parametrize(
    "field",
    [
        "balance",
        "equity",
        "entry_price",
        "stop_loss",
        "tick_value",
        "contract_size",
    ],
)
def test_nan_financial_inputs_are_rejected(
    field: str,
) -> None:
    # Create the engine.
    engine = make_engine()

    # Inject NaN into the selected field.
    kwargs = make_valid_kwargs()
    kwargs[field] = float("nan")

    # NaN must never enter risk calculations.
    with pytest.raises(RiskIntelligenceError):
        engine.analyze(**kwargs)


@pytest.mark.parametrize(
    "field",
    [
        "balance",
        "equity",
        "entry_price",
        "stop_loss",
        "tick_value",
        "contract_size",
    ],
)
def test_infinite_financial_inputs_are_rejected(
    field: str,
) -> None:
    # Create the engine.
    engine = make_engine()

    # Inject infinity into the selected field.
    kwargs = make_valid_kwargs()
    kwargs[field] = float("inf")

    # Infinity must never enter risk calculations.
    with pytest.raises(RiskIntelligenceError):
        engine.analyze(**kwargs)


@pytest.mark.parametrize(
    "field",
    [
        "balance",
        "equity",
        "entry_price",
        "stop_loss",
        "tick_value",
        "contract_size",
    ],
)
def test_boolean_financial_inputs_are_rejected(
    field: str,
) -> None:
    # Create the engine.
    engine = make_engine()

    # Inject a boolean.
    kwargs = make_valid_kwargs()
    kwargs[field] = True

    # Boolean financial inputs are invalid.
    with pytest.raises(RiskIntelligenceError):
        engine.analyze(**kwargs)


def test_invalid_engine_risk_configuration_is_rejected() -> None:
    # Minimum risk cannot be zero.
    with pytest.raises(RiskIntelligenceError):
        RiskIntelligenceEngine(
            minimum_risk_percent=0.0,
        )

    # Maximum risk cannot be zero.
    with pytest.raises(RiskIntelligenceError):
        RiskIntelligenceEngine(
            maximum_risk_percent=0.0,
        )


def test_minimum_risk_cannot_exceed_maximum() -> None:
    # Invalid risk-policy ordering must be rejected.
    with pytest.raises(RiskIntelligenceError):
        RiskIntelligenceEngine(
            minimum_risk_percent=2.0,
            maximum_risk_percent=1.0,
        )


def test_default_risk_must_be_inside_policy() -> None:
    # Default risk above the maximum is invalid.
    with pytest.raises(RiskIntelligenceError):
        RiskIntelligenceEngine(
            default_risk_percent=3.0,
            maximum_risk_percent=2.0,
        )


def test_minimum_position_size_cannot_exceed_maximum() -> None:
    # Invalid position-size ordering must be rejected.
    with pytest.raises(RiskIntelligenceError):
        RiskIntelligenceEngine(
            minimum_position_size=10.0,
            maximum_position_size=5.0,
        )


def test_negative_spread_is_rejected() -> None:
    # Create the engine.
    engine = make_engine()

    # Negative spread is impossible.
    kwargs = make_valid_kwargs()
    kwargs["spread"] = -1.0

    # The input must be rejected.
    with pytest.raises(RiskIntelligenceError):
        engine.analyze(**kwargs)


def test_negative_commission_is_rejected() -> None:
    # Create the engine.
    engine = make_engine()

    # Negative commission is invalid.
    kwargs = make_valid_kwargs()
    kwargs["commission"] = -1.0

    # The input must be rejected.
    with pytest.raises(RiskIntelligenceError):
        engine.analyze(**kwargs)


def test_negative_slippage_is_rejected() -> None:
    # Create the engine.
    engine = make_engine()

    # Negative slippage is invalid.
    kwargs = make_valid_kwargs()
    kwargs["slippage"] = -1.0

    # The input must be rejected.
    with pytest.raises(RiskIntelligenceError):
        engine.analyze(**kwargs)


def test_negative_daily_loss_is_rejected() -> None:
    # Create the engine.
    engine = make_engine()

    # Negative daily loss is invalid.
    kwargs = make_valid_kwargs()
    kwargs["daily_loss"] = -1.0

    # The input must be rejected.
    with pytest.raises(RiskIntelligenceError):
        engine.analyze(**kwargs)


def test_negative_drawdown_is_rejected() -> None:
    # Create the engine.
    engine = make_engine()

    # Negative drawdown is invalid.
    kwargs = make_valid_kwargs()
    kwargs["drawdown_percent"] = -1.0

    # The input must be rejected.
    with pytest.raises(RiskIntelligenceError):
        engine.analyze(**kwargs)


def test_negative_open_trades_is_rejected() -> None:
    # Create the engine.
    engine = make_engine()

    # Negative trade count is impossible.
    kwargs = make_valid_kwargs()
    kwargs["open_trades"] = -1

    # The input must be rejected.
    with pytest.raises(RiskIntelligenceError):
        engine.analyze(**kwargs)


def test_boolean_open_trades_is_rejected() -> None:
    # Create the engine.
    engine = make_engine()

    # Boolean values are not valid trade counts.
    kwargs = make_valid_kwargs()
    kwargs["open_trades"] = True

    # The input must be rejected.
    with pytest.raises(RiskIntelligenceError):
        engine.analyze(**kwargs)


def test_boolean_session_flag_is_rejected() -> None:
    # Create the engine.
    engine = make_engine()

    # Use a non-boolean session value.
    kwargs = make_valid_kwargs()
    kwargs["session_allowed"] = 1

    # The input must be rejected.
    with pytest.raises(RiskIntelligenceError):
        engine.analyze(**kwargs)


def test_blocked_result_has_zero_position_size() -> None:
    # Create the engine.
    engine = make_engine()

    # Force a spread veto.
    kwargs = make_valid_kwargs()
    kwargs["spread"] = 100.0

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # A blocked trade must never expose an executable size.
    assert result.blocked is True
    assert result.position_size == 0.0


def test_position_size_is_deterministic() -> None:
    # Create the engine.
    engine = make_engine()

    # Analyze the exact same input twice.
    kwargs = make_valid_kwargs()

    first = engine.analyze(**kwargs)
    second = engine.analyze(**kwargs)

    # Results must be identical for deterministic risk logic.
    assert first == second


def test_risk_model_values_are_finite() -> None:
    # Create the engine.
    engine = make_engine()

    # Analyze a valid trade.
    result = engine.analyze(
        **make_valid_kwargs(),
    )

    # Verify all calculated floating-point fields are finite.
    assert isclose(result.risk_amount, 100.0)
    assert isclose(result.stop_distance, 10.0)
    assert isclose(result.position_size, 10.0)


# ---------------------------------------------------------------------------
# P2.16 POSITION-SIZING HARDENING TESTS
# ---------------------------------------------------------------------------


def test_tick_size_changes_position_size() -> None:
    # Create the risk engine.
    engine = make_engine()

    # Use a tick size of 0.5 instead of the default 1.0.
    kwargs = make_valid_kwargs()
    kwargs["tick_size"] = 0.5

    # Keep the tick value at one monetary unit.
    kwargs["tick_value"] = 1.0

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # Ten price units represent twenty ticks.
    assert result.stop_distance == 10.0
    assert result.raw_position_size == 5.0
    assert result.position_size == 5.0


def test_tick_value_changes_position_size() -> None:
    # Create the risk engine.
    engine = make_engine()

    # Double the monetary value of every tick.
    kwargs = make_valid_kwargs()
    kwargs["tick_value"] = 2.0

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # Higher tick value means fewer volume units can be traded.
    assert result.raw_position_size == 5.0
    assert result.position_size == 5.0


def test_contract_size_changes_position_size() -> None:
    # Create the risk engine.
    engine = make_engine()

    # Double the contract size.
    kwargs = make_valid_kwargs()
    kwargs["contract_size"] = 2.0

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # Doubling contract size doubles risk per volume unit.
    assert result.raw_position_size == 5.0
    assert result.position_size == 5.0


def test_volume_step_rounds_position_size_down() -> None:
    # Create the risk engine.
    engine = make_engine()

    # Produce a raw position size of approximately 6.6667.
    kwargs = make_valid_kwargs()
    kwargs["tick_size"] = 1.0
    kwargs["tick_value"] = 1.5
    kwargs["contract_size"] = 1.0
    kwargs["volume_step"] = 0.1

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # Broker step rounding must move downward.
    assert result.raw_position_size == pytest.approx(
        6.6666666667,
        rel=1e-9,
    )
    assert result.position_size == 6.6


def test_volume_step_never_rounds_up() -> None:
    # Create the risk engine.
    engine = make_engine()

    # Create a raw position size that falls between volume steps.
    kwargs = make_valid_kwargs()
    kwargs["tick_value"] = 1.5
    kwargs["volume_step"] = 0.1

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # The engine must never increase the raw risk by rounding upward.
    assert result.position_size <= result.raw_position_size
    assert result.position_size == 6.6


def test_broker_minimum_volume_is_enforced() -> None:
    # Create the risk engine.
    engine = make_engine()

    # The calculated volume is ten, so set broker minimum above it.
    kwargs = make_valid_kwargs()
    kwargs["broker_min_position_size"] = 20.0

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # The broker minimum cannot be satisfied safely.
    assert result.blocked is True
    assert result.position_size == 0.0
    assert any(
        reason.reason_type is RiskReasonType.POSITION_SIZE_TOO_SMALL
        for reason in result.reasons
    )


def test_broker_maximum_volume_is_enforced() -> None:
    # Create the risk engine.
    engine = make_engine()

    # The calculated volume is ten.
    kwargs = make_valid_kwargs()
    kwargs["broker_max_position_size"] = 5.0

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # The broker maximum must block the trade.
    assert result.blocked is True
    assert result.position_size == 0.0
    assert any(
        reason.reason_type is RiskReasonType.POSITION_SIZE_TOO_LARGE
        for reason in result.reasons
    )


def test_policy_maximum_still_applies_with_broker_maximum() -> None:
    # Create an engine with a smaller internal policy maximum.
    engine = RiskIntelligenceEngine(
        maximum_position_size=7.0,
    )

    # Broker allows more, but internal policy does not.
    kwargs = make_valid_kwargs()
    kwargs["broker_max_position_size"] = 50.0

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # The policy maximum remains authoritative.
    assert result.blocked is True
    assert result.position_size == 0.0


def test_broker_minimum_cannot_exceed_broker_maximum() -> None:
    # Create the risk engine.
    engine = make_engine()

    # Supply an impossible broker range.
    kwargs = make_valid_kwargs()
    kwargs["broker_min_position_size"] = 10.0
    kwargs["broker_max_position_size"] = 5.0

    # Invalid broker metadata must raise.
    with pytest.raises(RiskIntelligenceError):
        engine.analyze(**kwargs)


@pytest.mark.parametrize(
    "field",
    [
        "tick_size",
        "tick_value",
        "volume_step",
        "broker_min_position_size",
        "broker_max_position_size",
    ],
)
def test_broker_parameters_reject_zero(
    field: str,
) -> None:
    # Create the risk engine.
    engine = make_engine()

    # Set one broker parameter to zero.
    kwargs = make_valid_kwargs()
    kwargs[field] = 0.0

    # Zero is invalid for these parameters.
    with pytest.raises(RiskIntelligenceError):
        engine.analyze(**kwargs)


@pytest.mark.parametrize(
    "field",
    [
        "tick_size",
        "tick_value",
        "volume_step",
        "broker_min_position_size",
        "broker_max_position_size",
    ],
)
def test_broker_parameters_reject_negative(
    field: str,
) -> None:
    # Create the risk engine.
    engine = make_engine()

    # Set one broker parameter negative.
    kwargs = make_valid_kwargs()
    kwargs[field] = -1.0

    # Negative broker specifications are invalid.
    with pytest.raises(RiskIntelligenceError):
        engine.analyze(**kwargs)


@pytest.mark.parametrize(
    "field",
    [
        "tick_size",
        "tick_value",
        "volume_step",
        "broker_min_position_size",
        "broker_max_position_size",
    ],
)
def test_broker_parameters_reject_nan(
    field: str,
) -> None:
    # Create the risk engine.
    engine = make_engine()

    # Inject NaN into one broker parameter.
    kwargs = make_valid_kwargs()
    kwargs[field] = float("nan")

    # NaN must never enter financial calculations.
    with pytest.raises(RiskIntelligenceError):
        engine.analyze(**kwargs)


@pytest.mark.parametrize(
    "field",
    [
        "tick_size",
        "tick_value",
        "volume_step",
        "broker_min_position_size",
        "broker_max_position_size",
    ],
)
def test_broker_parameters_reject_infinity(
    field: str,
) -> None:
    # Create the risk engine.
    engine = make_engine()

    # Inject infinity into one broker parameter.
    kwargs = make_valid_kwargs()
    kwargs[field] = float("inf")

    # Infinity must never enter financial calculations.
    with pytest.raises(RiskIntelligenceError):
        engine.analyze(**kwargs)


@pytest.mark.parametrize(
    "field",
    [
        "tick_size",
        "tick_value",
        "volume_step",
        "broker_min_position_size",
        "broker_max_position_size",
    ],
)
def test_broker_parameters_reject_boolean(
    field: str,
) -> None:
    # Create the risk engine.
    engine = make_engine()

    # Inject a boolean into one broker parameter.
    kwargs = make_valid_kwargs()
    kwargs[field] = True

    # Boolean broker parameters are invalid.
    with pytest.raises(RiskIntelligenceError):
        engine.analyze(**kwargs)


def test_final_position_size_risk_does_not_exceed_requested_risk() -> None:
    # Create the risk engine.
    engine = make_engine()

    # Force a non-even volume step.
    kwargs = make_valid_kwargs()
    kwargs["tick_value"] = 1.5
    kwargs["volume_step"] = 0.1

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # Reconstruct monetary risk from the final broker volume.
    risk_per_unit = (
        result.stop_distance
        / result.tick_size
        * result.tick_value
        * result.contract_size
    )

    actual_risk = (
        result.position_size
        * risk_per_unit
    )

    # Actual risk must never exceed requested risk.
    assert actual_risk <= result.risk_amount


def test_raw_position_size_is_preserved_for_auditability() -> None:
    # Create the risk engine.
    engine = make_engine()

    # Use a volume step that requires rounding.
    kwargs = make_valid_kwargs()
    kwargs["tick_value"] = 1.5
    kwargs["volume_step"] = 0.1

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # Both theoretical and executable sizes must be available.
    assert result.raw_position_size > result.position_size
    assert result.raw_position_size > 0.0
    assert result.position_size > 0.0


def test_final_position_size_is_aligned_to_volume_step() -> None:
    # Create the risk engine.
    engine = make_engine()

    # Use a broker step of 0.05.
    kwargs = make_valid_kwargs()
    kwargs["volume_step"] = 0.05
    kwargs["tick_value"] = 1.3

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # Convert the final size back into broker steps.
    steps = result.position_size / result.volume_step

    # The result must represent a whole number of steps.
    assert steps == pytest.approx(
        round(steps),
        abs=1e-9,
    )


def test_default_broker_parameters_preserve_existing_position_size() -> None:
    # Create the risk engine.
    engine = make_engine()

    # Analyze using the original baseline configuration.
    result = engine.analyze(
        **make_valid_kwargs(),
    )

    # Existing behavior remains unchanged.
    assert result.position_size == 10.0
    assert result.raw_position_size == 10.0
    assert result.tick_size == 1.0
    assert result.tick_value == 1.0
    assert result.contract_size == 1.0
    assert result.volume_step == 0.01


def test_broker_maximum_lower_than_policy_is_authoritative() -> None:
    # Create an engine whose policy maximum is larger.
    engine = RiskIntelligenceEngine(
        maximum_position_size=100.0,
    )

    # Broker only permits five volume units.
    kwargs = make_valid_kwargs()
    kwargs["broker_max_position_size"] = 5.0

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # Ten units cannot be sent to the broker.
    assert result.blocked is True
    assert result.position_size == 0.0


def test_broker_minimum_equal_to_calculated_volume_is_allowed() -> None:
    # Create the risk engine.
    engine = make_engine()

    # Baseline calculation produces ten volume units.
    kwargs = make_valid_kwargs()
    kwargs["broker_min_position_size"] = 10.0

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # Exactly meeting the broker minimum is valid.
    assert result.allowed is True
    assert result.position_size == 10.0


def test_broker_maximum_equal_to_calculated_volume_is_allowed() -> None:
    # Create the risk engine.
    engine = make_engine()

    # Baseline calculation produces ten volume units.
    kwargs = make_valid_kwargs()
    kwargs["broker_max_position_size"] = 10.0

    # Analyze the trade.
    result = engine.analyze(**kwargs)

    # Exactly meeting the broker maximum is valid.
    assert result.allowed is True
    assert result.position_size == 10.0


def test_larger_stop_distance_reduces_position_size() -> None:
    # Create the risk engine.
    engine = make_engine()

    # Analyze a ten-unit stop.
    first_kwargs = make_valid_kwargs()
    first = engine.analyze(**first_kwargs)

    # Analyze a twenty-unit stop.
    second_kwargs = make_valid_kwargs()
    second_kwargs["stop_loss"] = 2010.0
    second = engine.analyze(**second_kwargs)

    # Larger stop distance means smaller position size.
    assert first.position_size > second.position_size
    assert first.position_size == 10.0
    assert second.position_size == 5.0


def test_higher_risk_percent_increases_position_size() -> None:
    # Create the risk engine.
    engine = make_engine()

    # Analyze at one percent risk.
    first_kwargs = make_valid_kwargs()
    first_kwargs["risk_percent"] = 1.0
    first = engine.analyze(**first_kwargs)

    # Analyze at two percent risk.
    second_kwargs = make_valid_kwargs()
    second_kwargs["risk_percent"] = 2.0
    second = engine.analyze(**second_kwargs)

    # Doubling risk percentage doubles raw position size.
    assert first.raw_position_size == 10.0
    assert second.raw_position_size == 20.0


def test_lower_tick_value_increases_position_size() -> None:
    # Create the risk engine.
    engine = make_engine()

    # Analyze with the default tick value.
    first_kwargs = make_valid_kwargs()
    first_kwargs["tick_value"] = 1.0
    first = engine.analyze(**first_kwargs)

    # Analyze with half the tick value.
    second_kwargs = make_valid_kwargs()
    second_kwargs["tick_value"] = 0.5
    second = engine.analyze(**second_kwargs)

    # Lower monetary tick value permits larger volume.
    assert first.position_size == 10.0
    assert second.position_size == 20.0


def test_position_size_rounding_is_deterministic() -> None:
    # Create the risk engine.
    engine = make_engine()

    # Use a configuration requiring volume-step rounding.
    kwargs = make_valid_kwargs()
    kwargs["tick_value"] = 1.7
    kwargs["volume_step"] = 0.03

    # Analyze twice.
    first = engine.analyze(**kwargs)
    second = engine.analyze(**kwargs)

    # Both results must be exactly reproducible.
    assert first.position_size == second.position_size
    assert first.raw_position_size == second.raw_position_size
    assert first == second