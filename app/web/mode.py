from __future__ import annotations

from enum import Enum


class WebMode(str, Enum):

    OFFLINE = "offline"
    WEB = "web"
    AUTO = "auto"


class WebModeController:

    def __init__(
        self,
        mode: WebMode = WebMode.OFFLINE,
    ):

        if not isinstance(mode, WebMode):
            raise ValueError(
                "mode must be a WebMode value."
            )

        self._mode = mode

    @property
    def mode(self) -> WebMode:

        return self._mode

    def set_mode(
        self,
        mode: WebMode,
    ) -> None:

        if not isinstance(mode, WebMode):
            raise ValueError(
                "mode must be a WebMode value."
            )

        self._mode = mode

    def is_offline(self) -> bool:

        return self._mode == WebMode.OFFLINE

    def allows_web(self) -> bool:

        return self._mode in {
            WebMode.WEB,
            WebMode.AUTO,
        }

    def requires_web_decision(self) -> bool:

        return self._mode == WebMode.AUTO