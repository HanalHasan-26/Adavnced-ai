import pytest

from app.web.auto import WebAutoDecider
from app.web.mode import WebMode


def test_current_query_uses_web():

    decider = WebAutoDecider()

    assert decider.decide(
        "What is the current gold price?"
    ) == WebMode.WEB


def test_latest_query_uses_web():

    decider = WebAutoDecider()

    assert decider.decide(
        "What is the latest forex news?"
    ) == WebMode.WEB


def test_news_query_uses_web():

    decider = WebAutoDecider()

    assert decider.decide(
        "Show me gold news"
    ) == WebMode.WEB


def test_today_query_uses_web():

    decider = WebAutoDecider()

    assert decider.decide(
        "What happened today?"
    ) == WebMode.WEB


def test_live_query_uses_web():

    decider = WebAutoDecider()

    assert decider.decide(
        "Give me the live market update"
    ) == WebMode.WEB


def test_normal_question_stays_offline():

    decider = WebAutoDecider()

    assert decider.decide(
        "What is support and resistance?"
    ) == WebMode.OFFLINE


def test_trading_explanation_stays_offline():

    decider = WebAutoDecider()

    assert decider.decide(
        "Explain order blocks"
    ) == WebMode.OFFLINE


def test_memory_question_stays_offline():

    decider = WebAutoDecider()

    assert decider.decide(
        "What is my name?"
    ) == WebMode.OFFLINE


def test_case_is_ignored():

    decider = WebAutoDecider()

    assert decider.decide(
        "WHAT IS THE LATEST GOLD NEWS?"
    ) == WebMode.WEB


def test_whitespace_is_ignored():

    decider = WebAutoDecider()

    assert decider.decide(
        "   What is the current price?   "
    ) == WebMode.WEB


def test_empty_query_is_rejected():

    decider = WebAutoDecider()

    with pytest.raises(
        ValueError,
        match="query cannot be empty",
    ):
        decider.decide("")


def test_non_string_query_is_rejected():

    decider = WebAutoDecider()

    with pytest.raises(
        ValueError,
        match="query must be a string",
    ):
        decider.decide(None)


def test_should_use_web_returns_boolean():

    decider = WebAutoDecider()

    assert decider.should_use_web(
        "latest news"
    ) is True

    assert decider.should_use_web(
        "explain support"
    ) is False