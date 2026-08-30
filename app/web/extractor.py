from __future__ import annotations

from html import unescape
from html.parser import HTMLParser


class _ReadableTextParser(HTMLParser):

    IGNORED_TAGS = {
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "canvas",
        "head",
    }

    BLOCK_TAGS = {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "div",
        "dl",
        "dt",
        "dd",
        "fieldset",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }

    BLOCK_MARKER = "\x00BLOCK\x00"

    def __init__(self):
        super().__init__()

        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ):

        tag = tag.lower()

        if tag in self.IGNORED_TAGS:
            self._ignored_depth += 1
            return

        if (
            self._ignored_depth == 0
            and tag in self.BLOCK_TAGS
        ):
            self.parts.append(
                self.BLOCK_MARKER
            )

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ):

        tag = tag.lower()

        if (
            self._ignored_depth == 0
            and tag in self.BLOCK_TAGS
        ):
            self.parts.append(
                self.BLOCK_MARKER
            )

    def handle_endtag(
        self,
        tag: str,
    ):

        tag = tag.lower()

        if tag in self.IGNORED_TAGS:

            if self._ignored_depth > 0:
                self._ignored_depth -= 1

            return

        if (
            self._ignored_depth == 0
            and tag in self.BLOCK_TAGS
        ):
            self.parts.append(
                self.BLOCK_MARKER
            )

    def handle_data(
        self,
        data: str,
    ):

        if self._ignored_depth > 0:
            return

        if data:
            self.parts.append(data)


class HTMLTextExtractor:

    def __init__(
        self,
        max_chars: int = 50_000,
    ):

        if max_chars <= 0:
            raise ValueError(
                "max_chars must be greater than 0."
            )

        self.max_chars = max_chars

    def extract(
        self,
        html: str,
    ) -> str:

        if not isinstance(html, str):
            raise ValueError(
                "html must be a string."
            )

        if not html.strip():
            return ""

        parser = _ReadableTextParser()

        parser.feed(html)
        parser.close()

        text = "".join(
            parser.parts
        )

        # Decode HTML entities.
        text = unescape(text)

        # Split only at actual HTML block boundaries.
        blocks = text.split(
            _ReadableTextParser.BLOCK_MARKER
        )

        cleaned_blocks: list[str] = []

        for block in blocks:

            # Collapse all whitespace inside a block.
            cleaned = " ".join(
                block.split()
            )

            if cleaned:
                cleaned_blocks.append(
                    cleaned
                )

        # Keep separate HTML blocks separate.
        text = "\n".join(
            cleaned_blocks
        )

        # Remove excessive blank lines.
        while "\n\n\n" in text:
            text = text.replace(
                "\n\n\n",
                "\n\n",
            )

        text = text.strip()

        # Enforce the maximum output size.
        if len(text) > self.max_chars:
            text = (
                text[:self.max_chars]
                .rstrip()
            )

        return text