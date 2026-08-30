from unittest.mock import Mock

import pytest
from app.knowledge.assistant import KnowledgeAssistant
from app.web.mode import WebMode, WebModeController
from app.web.search import SearchResult


class FakePipeline:

    def run(self, query, limit=5):

        return Mock(
            query=query,
            context="Local knowledge.",
        )


class FakeLLM:

    def __init__(self):

        self.prompts = []

    def generate(self, prompt):

        self.prompts.append(prompt)

        return "Test answer."


class FakeConversationMemory:

    def build_context(self, query, limit=5):

        return ""

    def save_user_message(self, message):

        return "user-id"

    def save_assistant_message(self, message):

        return "assistant-id"


def create_assistant(
    mode=WebMode.OFFLINE,
    web_search=None,
):

    controller = WebModeController(mode)

    return KnowledgeAssistant(
        retrieval_pipeline=FakePipeline(),
        llm=FakeLLM(),
        conversation_memory=FakeConversationMemory(),
        web_mode_controller=controller,
        web_search=web_search,
    )


def test_offline_mode_does_not_call_web_search():

    web_search = Mock()

    assistant = create_assistant(
        mode=WebMode.OFFLINE,
        web_search=web_search,
    )

    assistant.ask("What is support?")

    web_search.search.assert_not_called()


def test_offline_mode_works_without_web_search():

    assistant = create_assistant(
        mode=WebMode.OFFLINE,
        web_search=None,
    )

    answer = assistant.ask(
        "What is support?"
    )

    assert answer == "Test answer."


def test_web_mode_calls_web_search():

    web_search = Mock()

    web_search.search.return_value = [
        SearchResult(
            title="Forex Support",
            url="https://example.com/support",
            snippet="Support is a price level.",
        )
    ]

    assistant = create_assistant(
        mode=WebMode.WEB,
        web_search=web_search,
    )

    assistant.ask(
        "What is support?"
    )

    web_search.search.assert_called_once_with(
        query="What is support?",
        limit=5,
    )


def test_web_results_are_added_to_prompt():

    web_search = Mock()

    web_search.search.return_value = [
        SearchResult(
            title="Forex Support",
            url="https://example.com/support",
            snippet="Support is a price level.",
        )
    ]

    assistant = create_assistant(
        mode=WebMode.WEB,
        web_search=web_search,
    )

    assistant.ask(
        "What is support?"
    )

    llm = assistant.llm

    assert len(llm.prompts) == 1

    prompt = llm.prompts[0]

    assert "Forex Support" in prompt
    assert "https://example.com/support" in prompt
    assert "Support is a price level." in prompt


def test_web_mode_requires_web_search():

    assistant = create_assistant(
        mode=WebMode.WEB,
        web_search=None,
    )

    with pytest.raises(
        ValueError,
        match="web_search cannot be None",
    ):

        assistant.ask(
            "What is support?"
        )


def test_offline_web_retrieve_returns_empty():

    web_search = Mock()

    assistant = create_assistant(
        mode=WebMode.OFFLINE,
        web_search=web_search,
    )

    results = assistant.web_retrieve(
        "What is support?"
    )

    assert results == []

    web_search.search.assert_not_called()


def test_web_retrieve_returns_results():

    web_search = Mock()

    expected_results = [
        SearchResult(
            title="Test",
            url="https://example.com",
            snippet="Test result.",
        )
    ]

    web_search.search.return_value = (
        expected_results
    )

    assistant = create_assistant(
        mode=WebMode.WEB,
        web_search=web_search,
    )

    results = assistant.web_retrieve(
        "forex"
    )

    assert results == expected_results

    web_search.search.assert_called_once_with(
        query="forex",
        limit=5,
    )