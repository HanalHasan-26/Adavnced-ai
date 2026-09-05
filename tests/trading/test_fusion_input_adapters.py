"""
Tests for P2.20.2 fusion input adapters.

These tests verify that existing trading subsystem outputs can be converted
into the canonical SignalFusionEvidence contract without changing the
underlying trading logic.
"""

from datetime import datetime, timezone

import pytest

from app.trading.fusion.signal_fusion_intelligence import (
    FusionDirection,
)

from app.trading.fusion.fusion_input_adapters import (
    FusionInputAdapter,
    FusionInputAdapterError,
    FusionInputAdapterRegistry,
    create_default_fusion_adapters,
)


# Use one deterministic timezone-aware timestamp throughout the tests.
DECISION_TIME = datetime(
    2026,
    1,
    1,
    12,
    0,
    tzinfo=timezone.utc,
)


class FakeAssessment:
    """
    Minimal fake subsystem assessment used to test the adapter layer.

    This represents the type of result that existing trading subsystems
    provide to the fusion layer.
    """

    def __init__(
        self,
        *,
        score: float = 75.0,
        direction: str = "LONG",
        reasons: tuple[str, ...] = ("test evidence",),
    ) -> None:
        """Initialize the fake subsystem assessment."""

        # Store the normalized-test score.
        self.score = score

        # Store the source subsystem direction.
        self.direction = direction

        # Store explanation/reason information.
        self.reasons = reasons


def test_adapter_converts_result_to_fusion_evidence() -> None:
    """
    The adapter should convert a subsystem result into canonical evidence.
    """

    # Create an adapter for the Setup subsystem.
    adapter = FusionInputAdapter("setup")

    # Convert the fake assessment into fusion evidence.
    evidence = adapter.adapt(
        FakeAssessment(),
        decision_timestamp=DECISION_TIME,
    )

    # Verify the canonical source name.
    assert evidence.source == "setup"

    # Verify that the direction is the canonical FusionDirection enum.
    assert evidence.direction == FusionDirection.LONG

    # Verify the normalized score.
    assert evidence.score == 75.0

    # Verify the default source weight.
    assert evidence.weight == 1.0

    # Verify that the subsystem explanation is preserved.
    assert evidence.reason == "test evidence"


def test_adapter_supports_direction_aliases() -> None:
    """
    Common subsystem direction aliases should be normalized correctly.
    """

    # Create an adapter for the Entry subsystem.
    adapter = FusionInputAdapter("entry")

    # Adapt a BUY direction.
    evidence = adapter.adapt(
        FakeAssessment(direction="BUY"),
        decision_timestamp=DECISION_TIME,
    )

    # BUY must become the canonical LONG enum.
    assert evidence.direction == FusionDirection.LONG


def test_adapter_supports_explicit_score_and_direction() -> None:
    """
    Explicit score and direction arguments should override result fields.
    """

    # Create a Macro adapter.
    adapter = FusionInputAdapter("macro")

    # Override both score and direction during adaptation.
    evidence = adapter.adapt(
        FakeAssessment(
            score=10.0,
            direction="LONG",
        ),
        decision_timestamp=DECISION_TIME,
        score=-80.0,
        direction="SHORT",
    )

    # Verify the explicitly supplied score.
    assert evidence.score == -80.0

    # Verify the explicitly supplied canonical direction.
    assert evidence.direction == FusionDirection.SHORT


def test_adapter_rejects_none_result() -> None:
    """
    Missing subsystem output must not become artificial evidence.
    """

    # Create a Setup adapter.
    adapter = FusionInputAdapter("setup")

    # Missing subsystem output must raise an adapter error.
    with pytest.raises(FusionInputAdapterError):
        adapter.adapt(
            None,
            decision_timestamp=DECISION_TIME,
        )


def test_adapter_rejects_invalid_score() -> None:
    """
    Scores outside the canonical fusion range must be rejected.
    """

    # Create a Setup adapter.
    adapter = FusionInputAdapter("setup")

    # Scores above +100 are invalid.
    with pytest.raises(FusionInputAdapterError):
        adapter.adapt(
            FakeAssessment(),
            decision_timestamp=DECISION_TIME,
            score=101.0,
        )


def test_adapter_rejects_invalid_timestamp() -> None:
    """
    Naive timestamps must be rejected to prevent timezone ambiguity.
    """

    # Create a Setup adapter.
    adapter = FusionInputAdapter("setup")

    # A datetime without timezone information is invalid.
    with pytest.raises(FusionInputAdapterError):
        adapter.adapt(
            FakeAssessment(),
            decision_timestamp=datetime(
                2026,
                1,
                1,
                12,
                0,
            ),
        )


def test_registry_contains_all_standard_sources() -> None:
    """
    The registry must expose every canonical fusion source.
    """

    # Create the standard registry.
    registry = create_default_fusion_adapters()

    # Verify the complete canonical source list.
    assert registry.sources() == (
        "setup",
        "entry",
        "confluence",
        "macro",
        "news",
        "risk",
        "rr_ev",
    )


def test_registry_uses_expected_default_weights() -> None:
    """
    The registry must preserve the P2.20.1 canonical source weights.
    """

    # Create the standard registry.
    registry = FusionInputAdapterRegistry()

    # Verify Setup weight.
    assert registry.get("setup").default_weight == 1.0

    # Verify Entry weight.
    assert registry.get("entry").default_weight == 1.0

    # Verify Confluence weight.
    assert registry.get("confluence").default_weight == 1.0

    # Verify Macro weight.
    assert registry.get("macro").default_weight == 0.8

    # Verify News weight.
    assert registry.get("news").default_weight == 0.8

    # Verify Risk weight.
    assert registry.get("risk").default_weight == 1.2

    # Verify RR/EV weight.
    assert registry.get("rr_ev").default_weight == 1.0


def test_registry_adapts_multiple_results() -> None:
    """
    Multiple subsystem results should become ordered fusion evidence.
    """

    # Create the standard adapter registry.
    registry = create_default_fusion_adapters()

    # Adapt two subsystem results in a deterministic order.
    evidence = registry.adapt_many(
        (
            (
                "setup",
                FakeAssessment(
                    score=80.0,
                    direction="LONG",
                ),
            ),
            (
                "entry",
                FakeAssessment(
                    score=60.0,
                    direction="LONG",
                ),
            ),
        ),
        decision_timestamp=DECISION_TIME,
    )

    # Verify that two evidence records were created.
    assert len(evidence) == 2

    # Verify that input ordering is preserved.
    assert evidence[0].source == "setup"
    assert evidence[1].source == "entry"

    # Verify both directions use the canonical enum.
    assert evidence[0].direction == FusionDirection.LONG
    assert evidence[1].direction == FusionDirection.LONG


def test_registry_rejects_unknown_source() -> None:
    """
    Unknown subsystem names must not silently create new architecture.
    """

    # Create the standard registry.
    registry = create_default_fusion_adapters()

    # Unknown source names must be rejected.
    with pytest.raises(FusionInputAdapterError):
        registry.get("unknown")


def test_custom_weight_is_supported() -> None:
    """
    A custom adapter weight should be preserved.
    """

    # Create a Setup adapter with a custom weight.
    adapter = FusionInputAdapter(
        "setup",
        default_weight=2.0,
    )

    # Adapt the fake subsystem result.
    evidence = adapter.adapt(
        FakeAssessment(),
        decision_timestamp=DECISION_TIME,
    )

    # Verify the custom weight.
    assert evidence.weight == 2.0