import pytest

from app.web.extractor import HTMLTextExtractor


def test_extracts_visible_text():

    extractor = HTMLTextExtractor()

    html = """
    <html>
        <body>
            <h1>Gold Trading</h1>
            <p>Gold prices are moving.</p>
        </body>
    </html>
    """

    result = extractor.extract(html)

    assert "Gold Trading" in result
    assert "Gold prices are moving." in result


def test_removes_script():

    extractor = HTMLTextExtractor()

    html = """
    <html>
        <body>
            <p>Hello</p>
            <script>
                malicious_or_irrelevant_code();
            </script>
            <p>World</p>
        </body>
    </html>
    """

    result = extractor.extract(html)

    assert "Hello" in result
    assert "World" in result
    assert "malicious_or_irrelevant_code" not in result


def test_removes_style():

    extractor = HTMLTextExtractor()

    html = """
    <style>
        body { color: red; }
    </style>

    <p>Hello world</p>
    """

    result = extractor.extract(html)

    assert "Hello world" in result
    assert "color: red" not in result


def test_removes_head_content():

    extractor = HTMLTextExtractor()

    html = """
    <head>
        <title>Hidden title</title>
        <meta name="description" content="Hidden">
    </head>

    <body>
        <p>Visible content</p>
    </body>
    """

    result = extractor.extract(html)

    assert "Visible content" in result
    assert "Hidden title" not in result


def test_handles_html_entities():

    extractor = HTMLTextExtractor()

    html = """
    <p>Gold &amp; silver &lt; markets</p>
    """

    result = extractor.extract(html)

    assert result == (
        "Gold & silver < markets"
    )


def test_normalizes_whitespace():

    extractor = HTMLTextExtractor()

    html = """
    <p>
        Gold
        prices
        are
        rising.
    </p>
    """

    result = extractor.extract(html)

    assert result == (
        "Gold prices are rising."
    )


def test_preserves_separate_blocks():

    extractor = HTMLTextExtractor()

    html = """
    <p>First paragraph.</p>
    <p>Second paragraph.</p>
    """

    result = extractor.extract(html)

    assert "First paragraph." in result
    assert "Second paragraph." in result
    assert "\n" in result


def test_empty_html_returns_empty_string():

    extractor = HTMLTextExtractor()

    assert extractor.extract("") == ""
    assert extractor.extract("   ") == ""


def test_rejects_non_string_html():

    extractor = HTMLTextExtractor()

    with pytest.raises(
        ValueError,
        match="html must be a string",
    ):
        extractor.extract(None)


def test_rejects_invalid_max_chars():

    with pytest.raises(
        ValueError,
        match="max_chars must be greater than 0",
    ):
        HTMLTextExtractor(
            max_chars=0
        )


def test_limits_output_size():

    extractor = HTMLTextExtractor(
        max_chars=20
    )

    html = """
    <p>
        This is a very long
        webpage containing
        lots of text.
    </p>
    """

    result = extractor.extract(html)

    assert len(result) <= 20


def test_handles_malformed_html():

    extractor = HTMLTextExtractor()

    html = """
    <html>
        <body>
            <p>Gold prices
            <div>are moving
    """

    result = extractor.extract(html)

    assert "Gold prices" in result
    assert "are moving" in result


def test_removes_noscript_content():

    extractor = HTMLTextExtractor()

    html = """
    <p>Visible text</p>

    <noscript>
        This should not be included.
    </noscript>
    """

    result = extractor.extract(html)

    assert "Visible text" in result
    assert "This should not be included" not in result


def test_removes_svg_content():

    extractor = HTMLTextExtractor()

    html = """
    <svg>
        <text>SVG noise</text>
    </svg>

    <p>Useful article text</p>
    """

    result = extractor.extract(html)

    assert "SVG noise" not in result
    assert "Useful article text" in result