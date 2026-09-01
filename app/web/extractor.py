from __future__ import annotations

from html import unescape
from html.parser import HTMLParser
import re


class _ReadableTextParser(HTMLParser):
    """
    Generic readable HTML extractor.

    It removes scripts/styles/etc. and creates block boundaries.
    The public extractor additionally detects article/main content and
    strips common site furniture.
    """

    IGNORED_TAGS = {
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "canvas",
        "head",
        "iframe",
        "video",
        "audio",
        "source",
    }

    BLOCK_TAGS = {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
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

    def handle_endtag(self, tag: str):
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

    def handle_data(self, data: str):
        if self._ignored_depth > 0:
            return

        if data:
            self.parts.append(data)


class _ArticleParser(HTMLParser):
    """
    Attempts to capture semantic article/main content.

    This is intentionally heuristic because many websites use different
    HTML layouts. If no useful semantic region is found, the extractor
    falls back to the generic parser.
    """

    CANDIDATE_TAGS = {
        "article",
        "main",
    }

    IGNORED_TAGS = {
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "canvas",
        "head",
        "iframe",
        "nav",
        "footer",
        "form",
    }

    BLOCK_TAGS = {
        "br",
        "div",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "p",
        "blockquote",
        "section",
        "article",
    }

    MARKER = "\x00BLOCK\x00"

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.depth = 0
        self.capture_depth: int | None = None
        self.ignored_depth = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ):
        tag = tag.lower()

        if tag in self.IGNORED_TAGS:
            self.ignored_depth += 1
            return

        if self.ignored_depth:
            return

        self.depth += 1

        if (
            self.capture_depth is None
            and tag in self.CANDIDATE_TAGS
        ):
            self.capture_depth = self.depth

        if self.capture_depth is not None and tag in self.BLOCK_TAGS:
            self.parts.append(self.MARKER)

    def handle_endtag(self, tag: str):
        tag = tag.lower()

        if tag in self.IGNORED_TAGS:
            if self.ignored_depth:
                self.ignored_depth -= 1
            return

        if self.ignored_depth:
            return

        if self.capture_depth is not None and tag in self.BLOCK_TAGS:
            self.parts.append(self.MARKER)

        self.depth = max(0, self.depth - 1)

        if (
            self.capture_depth is not None
            and self.depth < self.capture_depth
        ):
            self.capture_depth = None

    def handle_data(self, data: str):
        if self.ignored_depth:
            return

        if self.capture_depth is not None and data:
            self.parts.append(data)


class HTMLTextExtractor:
    """
    Extract readable text from HTML.

    For news/article pages, semantic article/main content is preferred.
    For pages without semantic article containers, the generic readable
    parser is used.
    """

    def __init__(
        self,
        max_chars: int = 50_000,
    ):
        if max_chars <= 0:
            raise ValueError(
                "max_chars must be greater than 0."
            )

        self.max_chars = max_chars

    @staticmethod
    def _clean_blocks(text: str) -> str:
        text = unescape(text)

        blocks = text.split(
            _ReadableTextParser.BLOCK_MARKER
        )

        cleaned: list[str] = []

        for block in blocks:
            block = re.sub(
                r"\s+",
                " ",
                block,
            ).strip()

            if block:
                cleaned.append(block)

        # De-duplicate consecutive identical blocks.
        final: list[str] = []
        previous = None

        for block in cleaned:
            key = block.lower()

            if key == previous:
                continue

            final.append(block)
            previous = key

        return "\n\n".join(final)

    @staticmethod
    def _clean_article(text: str) -> str:
        text = unescape(text)

        blocks = text.split(
            _ArticleParser.MARKER
        )

        cleaned: list[str] = []

        for block in blocks:
            block = re.sub(
                r"\s+",
                " ",
                block,
            ).strip()

            if block:
                cleaned.append(block)

        final: list[str] = []
        previous = None

        for block in cleaned:
            key = block.lower()

            if key == previous:
                continue

            final.append(block)
            previous = key

        return "\n\n".join(final)

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

        # First try semantic article/main content.
        article_parser = _ArticleParser()

        try:
            article_parser.feed(html)
            article_parser.close()
        except Exception:
            article_parser.parts = []

        article_text = self._clean_article(
            "".join(article_parser.parts)
        )

        # Only trust semantic extraction if it is substantial.
        if len(article_text) >= 250:
            return article_text[:self.max_chars]

        # Fallback to generic readable extraction.
        parser = _ReadableTextParser()

        parser.feed(html)
        parser.close()

        text = self._clean_blocks(
            "".join(parser.parts)
        )

        return text[:self.max_chars]
