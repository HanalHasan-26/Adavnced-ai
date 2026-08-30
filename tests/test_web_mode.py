import pytest

from app.web.mode import (
    WebMode,
    WebModeController,
)


def test_default_mode_is_offline():

    controller = WebModeController()

    assert controller.mode == WebMode.OFFLINE


def test_offline_mode_blocks_web():

    controller = WebModeController(
        WebMode.OFFLINE
    )

    assert controller.is_offline()
    assert not controller.allows_web()


def test_web_mode_allows_web():

    controller = WebModeController(
        WebMode.WEB
    )

    assert not controller.is_offline()
    assert controller.allows_web()
    assert not controller.requires_web_decision()


def test_auto_mode_allows_web():

    controller = WebModeController(
        WebMode.AUTO
    )

    assert not controller.is_offline()
    assert controller.allows_web()
    assert controller.requires_web_decision()


def test_set_mode():

    controller = WebModeController()

    controller.set_mode(
        WebMode.WEB
    )

    assert controller.mode == WebMode.WEB

    controller.set_mode(
        WebMode.AUTO
    )

    assert controller.mode == WebMode.AUTO

    controller.set_mode(
        WebMode.OFFLINE
    )

    assert controller.mode == WebMode.OFFLINE


def test_invalid_initial_mode():

    with pytest.raises(
        ValueError,
        match="mode must be a WebMode value",
    ):

        WebModeController(
            "invalid"
        )


def test_invalid_set_mode():

    controller = WebModeController()

    with pytest.raises(
        ValueError,
        match="mode must be a WebMode value",
    ):

        controller.set_mode(
            "invalid"
        )