"""
Tests for the deterministic RR + EV decision gate.
"""

import pytest

from app.trading.risk.risk_reward_ev_gate import (
    RiskRewardEVDecision,
    RiskRewardEVGate,
    RiskRewardEVGateError,
    RiskRewardEVReasonType,
)


class TestRiskRewardEVGate:
    """Test the complete RR/EV decision gate."""

    def test_default_configuration(self) -> None:
        """The default RR threshold must be 2:1."""

        gate = RiskRewardEVGate()

        assert gate.minimum_risk_reward == 2.0
        assert gate.minimum_net_expected_value_r == 0.0

    def test_valid_rr_and_positive_ev_are_allowed(self) -> None:
        """A valid RR and positive net EV should be allowed."""

        gate = RiskRewardEVGate()

        result = gate.analyze(
            risk_reward=2.5,
            net_expected_value_r=0.25,
        )

        assert result.decision == RiskRewardEVDecision.ALLOW
        assert result.allowed is True
        assert result.blocked is False
        assert result.valid is True

        assert (
            RiskRewardEVReasonType.RR_VALID.value
            in result.reasons
        )

        assert (
            RiskRewardEVReasonType.EV_VALID.value
            in result.reasons
        )

    def test_exact_rr_threshold_is_allowed(self) -> None:
        """RR exactly equal to the minimum should be allowed."""

        gate = RiskRewardEVGate(
            minimum_risk_reward=2.0,
        )

        result = gate.analyze(
            risk_reward=2.0,
            net_expected_value_r=0.0,
        )

        assert result.allowed is True

    def test_rr_below_threshold_is_blocked(self) -> None:
        """RR below the minimum must block the trade."""

        gate = RiskRewardEVGate(
            minimum_risk_reward=2.0,
        )

        result = gate.analyze(
            risk_reward=1.99,
            net_expected_value_r=1.0,
        )

        assert result.blocked is True

        assert (
            RiskRewardEVReasonType.RR_TOO_LOW.value
            in result.reasons
        )

    def test_exact_zero_ev_is_allowed_by_default(self) -> None:
        """Zero net EV meets the default minimum EV threshold."""

        gate = RiskRewardEVGate()

        result = gate.analyze(
            risk_reward=2.0,
            net_expected_value_r=0.0,
        )

        assert result.allowed is True
        assert result.net_expected_value_r == 0.0

    def test_negative_ev_is_blocked(self) -> None:
        """Negative net EV must be blocked."""

        gate = RiskRewardEVGate()

        result = gate.analyze(
            risk_reward=3.0,
            net_expected_value_r=-0.01,
        )

        assert result.blocked is True

        assert (
            RiskRewardEVReasonType.EV_TOO_LOW.value
            in result.reasons
        )

    def test_custom_ev_threshold(self) -> None:
        """A custom minimum net EV must be enforced."""

        gate = RiskRewardEVGate(
            minimum_risk_reward=2.0,
            minimum_net_expected_value_r=0.20,
        )

        blocked = gate.analyze(
            risk_reward=2.5,
            net_expected_value_r=0.19,
        )

        allowed = gate.analyze(
            risk_reward=2.5,
            net_expected_value_r=0.20,
        )

        assert blocked.blocked is True
        assert allowed.allowed is True

    def test_both_conditions_are_required(self) -> None:
        """RR and EV must both pass."""

        gate = RiskRewardEVGate(
            minimum_risk_reward=2.0,
            minimum_net_expected_value_r=0.10,
        )

        rr_failure = gate.analyze(
            risk_reward=1.5,
            net_expected_value_r=0.50,
        )

        ev_failure = gate.analyze(
            risk_reward=3.0,
            net_expected_value_r=0.05,
        )

        both_pass = gate.analyze(
            risk_reward=3.0,
            net_expected_value_r=0.10,
        )

        assert rr_failure.blocked is True
        assert ev_failure.blocked is True
        assert both_pass.allowed is True

    def test_check_returns_boolean(self) -> None:
        """check() should return only the final decision."""

        gate = RiskRewardEVGate()

        assert gate.check(
            risk_reward=2.0,
            net_expected_value_r=0.0,
        ) is True

        assert gate.check(
            risk_reward=1.0,
            net_expected_value_r=1.0,
        ) is False

    def test_evaluate_is_alias_for_analyze(self) -> None:
        """evaluate() should behave like analyze()."""

        gate = RiskRewardEVGate()

        result = gate.evaluate(
            risk_reward=2.0,
            net_expected_value_r=0.0,
        )

        assert result.allowed is True

    def test_near_zero_ev_is_normalized(self) -> None:
        """Tiny floating-point EV errors should become exact zero."""

        gate = RiskRewardEVGate()

        result = gate.analyze(
            risk_reward=2.0,
            net_expected_value_r=-1.1102230246251565e-16,
        )

        assert result.net_expected_value_r == 0.0
        assert result.allowed is True

    def test_nan_rr_is_blocked(self) -> None:
        """NaN RR must never be accepted."""

        gate = RiskRewardEVGate()

        result = gate.analyze(
            risk_reward=float("nan"),
            net_expected_value_r=0.5,
        )

        assert result.blocked is True
        assert result.valid is False

    def test_infinite_rr_is_blocked(self) -> None:
        """Infinite RR must never be accepted."""

        gate = RiskRewardEVGate()

        result = gate.analyze(
            risk_reward=float("inf"),
            net_expected_value_r=0.5,
        )

        assert result.blocked is True
        assert result.valid is False

    def test_nan_ev_is_blocked(self) -> None:
        """NaN EV must never be accepted."""

        gate = RiskRewardEVGate()

        result = gate.analyze(
            risk_reward=2.0,
            net_expected_value_r=float("nan"),
        )

        assert result.blocked is True
        assert result.valid is False

    def test_infinite_ev_is_blocked(self) -> None:
        """Infinite EV must never be accepted."""

        gate = RiskRewardEVGate()

        result = gate.analyze(
            risk_reward=2.0,
            net_expected_value_r=float("inf"),
        )

        assert result.blocked is True
        assert result.valid is False

    def test_invalid_constructor_rr_threshold(self) -> None:
        """The minimum RR must be positive and finite."""

        with pytest.raises(RiskRewardEVGateError):
            RiskRewardEVGate(minimum_risk_reward=0.0)

        with pytest.raises(RiskRewardEVGateError):
            RiskRewardEVGate(minimum_risk_reward=-1.0)

        with pytest.raises(RiskRewardEVGateError):
            RiskRewardEVGate(
                minimum_risk_reward=float("inf")
            )

    def test_invalid_constructor_ev_threshold(self) -> None:
        """The minimum EV threshold must be finite."""

        with pytest.raises(RiskRewardEVGateError):
            RiskRewardEVGate(
                minimum_net_expected_value_r=float("nan")
            )

    def test_boolean_input_is_rejected(self) -> None:
        """Boolean values must not silently become numeric values."""

        gate = RiskRewardEVGate()

        result = gate.analyze(
            risk_reward=True,
            net_expected_value_r=0.5,
        )

        assert result.blocked is True
        assert result.valid is False

    def test_warning_documents_authority_boundary(self) -> None:
        """The result should document that Risk Engine remains authoritative."""

        gate = RiskRewardEVGate()

        result = gate.analyze(
            risk_reward=2.0,
            net_expected_value_r=0.1,
        )

        assert any(
            "Risk Engine" in warning
            for warning in result.warnings
        )

    def test_result_preserves_thresholds(self) -> None:
        """Audit output should preserve configured thresholds."""

        gate = RiskRewardEVGate(
            minimum_risk_reward=2.5,
            minimum_net_expected_value_r=0.15,
        )

        result = gate.analyze(
            risk_reward=3.0,
            net_expected_value_r=0.20,
        )

        assert result.minimum_risk_reward == 2.5
        assert result.minimum_net_expected_value_r == 0.15

    def test_reason_for_success_contains_combined_reason(self) -> None:
        """Successful decisions should have a combined audit reason."""

        gate = RiskRewardEVGate()

        result = gate.analyze(
            risk_reward=2.0,
            net_expected_value_r=0.0,
        )

        assert (
            RiskRewardEVReasonType.RR_AND_EV_VALID.value
            in result.reasons
        )