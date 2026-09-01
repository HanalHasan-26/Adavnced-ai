from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser


class _ArticleParser(HTMLParser):

    IGNORED_TAGS = {
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "canvas",
        "iframe",
        "nav",
        "footer",
        "header",
        "form",
        "aside",
    }

    BLOCK_TAGS = {
        "article",
        "p",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "blockquote",
        "div",
    }

    def __init__(self):
        super().__init__()

        self.parts: list[str] = []
        self.ignored_depth = 0
        self.article_depth = 0

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ):

        tag = tag.lower()

        if tag in self.IGNORED_TAGS:
            self.ignored_depth += 1
            return

        if self.ignored_depth > 0:
            return

        if tag == "article":
            self.article_depth += 1

        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(
        self,
        tag: str,
    ):

        tag = tag.lower()

        if tag in self.IGNORED_TAGS:

            if self.ignored_depth > 0:
                self.ignored_depth -= 1

            return

        if self.ignored_depth > 0:
            return

        if tag == "article" and self.article_depth > 0:
            self.article_depth -= 1

        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(
        self,
        data: str,
    ):

        if self.ignored_depth > 0:
            return

        if data:
            self.parts.append(data)


class WebArticleExtractor:

    def __init__(
        self,
        max_chars: int = 20_000,
        min_paragraph_length: int = 40,
    ):

        if max_chars <= 0:
            raise ValueError(
                "max_chars must be greater than 0."
            )

        if min_paragraph_length <= 0:
            raise ValueError(
                "min_paragraph_length must be greater than 0."
            )

        self.max_chars = max_chars
        self.min_paragraph_length = (
            min_paragraph_length
        )

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

        parser = _ArticleParser()

        parser.feed(html)
        parser.close()

        text = "".join(parser.parts)

        text = unescape(text)

        lines = text.splitlines()

        cleaned_lines: list[str] = []

        for line in lines:

            line = " ".join(
                line.split()
            )

            if not line:
                continue

            cleaned_lines.append(line)

        if not cleaned_lines:
            return ""

        paragraphs: list[str] = []

        current: list[str] = []

        for line in cleaned_lines:

            current.append(line)

            if (
                len(" ".join(current))
                >= self.min_paragraph_length
            ):

                paragraphs.append(
                    " ".join(current)
                )

                current = []

        if current:
            paragraphs.append(
                " ".join(current)
            )

        # Remove obvious navigation/noise lines.
        filtered: list[str] = []

        noise_patterns = (
            r"^sign in$",
            r"^log in$",
            r"^subscribe$",
            r"^advertisement$",
            r"^advertise$",
            r"^cookie",
            r"^privacy policy$",
            r"^terms of use$",
            r"^all rights reserved$",
            r"^follow us$",
            r"^share$",
        )

        for paragraph in paragraphs:

            lower = paragraph.lower().strip()

            if any(
                re.match(
                    pattern,
                    lower,
                    re.IGNORECASE,
                )
                for pattern in noise_patterns
            ):
                continue

            filtered.append(paragraph)

        text = "\n\n".join(filtered)

        text = text.strip()

        if len(text) > self.max_chars:
            text = (
                text[:self.max_chars]
                .rstrip()
            )

        return text