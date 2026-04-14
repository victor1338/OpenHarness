"""Tests for the thinking/reasoning delta pipeline (fork-specific).

These tests validate the changes made in victor1338/OpenHarness
to stream thinking tokens from the API through to the OHJSON protocol.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from openharness.api.client import (
    ApiMessageCompleteEvent,
    ApiTextDeltaEvent,
    ApiThinkingDeltaEvent,
)
from openharness.api.usage import UsageSnapshot
from openharness.config.settings import PermissionSettings
from openharness.engine.messages import ConversationMessage, TextBlock
from openharness.engine.query import QueryContext, run_query
from openharness.engine.query_engine import QueryEngine
from openharness.engine.stream_events import (
    AssistantTextDelta,
    AssistantTurnComplete,
    ThinkingDelta,
)
from openharness.permissions import PermissionChecker, PermissionMode
from openharness.tools import create_default_tool_registry
from openharness.ui.backend_host import BackendHostConfig, ReactBackendHost
from openharness.ui.protocol import BackendEvent
from openharness.ui.runtime import build_runtime, close_runtime, start_runtime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _FakeResponse:
    message: ConversationMessage
    usage: UsageSnapshot


class ThinkingApiClient:
    """Fake streaming client that emits thinking deltas before text."""

    def __init__(self, thinking: str, text: str) -> None:
        self._thinking = thinking
        self._text = text

    async def stream_message(self, request):
        del request
        # Emit thinking tokens first
        for chunk in [self._thinking[i:i+5] for i in range(0, len(self._thinking), 5)]:
            yield ApiThinkingDeltaEvent(text=chunk)
        # Then emit text tokens
        yield ApiTextDeltaEvent(text=self._text)
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=[TextBlock(text=self._text)]),
            usage=UsageSnapshot(input_tokens=10, output_tokens=20),
            stop_reason=None,
        )


# ---------------------------------------------------------------------------
# 1. BackendEvent Literal accepts "thinking_delta"
# ---------------------------------------------------------------------------

class TestBackendEventThinkingDelta:
    """Validate that BackendEvent schema accepts thinking_delta."""

    def test_thinking_delta_is_valid_type(self):
        """BackendEvent(type='thinking_delta') must not raise Pydantic validation."""
        event = BackendEvent(type="thinking_delta", message="reasoning text")
        assert event.type == "thinking_delta"
        assert event.message == "reasoning text"

    def test_thinking_delta_serializes_to_json(self):
        """thinking_delta events must round-trip through JSON."""
        event = BackendEvent(type="thinking_delta", message="step 1")
        data = event.model_dump()
        assert data["type"] == "thinking_delta"
        assert data["message"] == "step 1"
        restored = BackendEvent(**data)
        assert restored.type == "thinking_delta"


# ---------------------------------------------------------------------------
# 2. ApiThinkingDeltaEvent exists and is part of ApiStreamEvent
# ---------------------------------------------------------------------------

class TestApiThinkingDeltaEvent:
    """Validate the API-layer thinking event."""

    def test_create_event(self):
        event = ApiThinkingDeltaEvent(text="let me think")
        assert event.text == "let me think"

    def test_is_frozen(self):
        event = ApiThinkingDeltaEvent(text="frozen")
        with pytest.raises(AttributeError):
            event.text = "nope"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 3. ThinkingDelta stream event exists
# ---------------------------------------------------------------------------

class TestThinkingDeltaStreamEvent:
    """Validate the engine-layer thinking stream event."""

    def test_create_event(self):
        event = ThinkingDelta(text="reasoning")
        assert event.text == "reasoning"

    def test_is_frozen(self):
        event = ThinkingDelta(text="frozen")
        with pytest.raises(AttributeError):
            event.text = "nope"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 4. run_query forwards ApiThinkingDeltaEvent → ThinkingDelta
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_query_yields_thinking_delta(tmp_path):
    """run_query must yield ThinkingDelta when the API client emits ApiThinkingDeltaEvent."""
    client = ThinkingApiClient(thinking="Let me reason about this", text="The answer is 42.")
    context = QueryContext(
        api_client=client,
        tool_registry=create_default_tool_registry(),
        permission_checker=PermissionChecker(
            PermissionSettings(mode=PermissionMode.FULL_AUTO),
        ),
        cwd=tmp_path,
        model="test-model",
        system_prompt="You are a test assistant.",
        max_tokens=1024,
    )
    messages = [ConversationMessage.from_user_text("What is the meaning?")]

    events = []
    async for event, _usage in run_query(context, messages):
        events.append(event)

    thinking_events = [e for e in events if isinstance(e, ThinkingDelta)]
    text_events = [e for e in events if isinstance(e, AssistantTextDelta)]
    complete_events = [e for e in events if isinstance(e, AssistantTurnComplete)]

    assert len(thinking_events) > 0, "Expected at least one ThinkingDelta event"
    assert "".join(e.text for e in thinking_events) == "Let me reason about this"
    assert len(text_events) == 1
    assert text_events[0].text == "The answer is 42."
    assert len(complete_events) == 1


# ---------------------------------------------------------------------------
# 5. Backend host emits thinking_delta OHJSON events
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_backend_host_emits_thinking_delta(tmp_path, monkeypatch):
    """ReactBackendHost must emit thinking_delta BackendEvents."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENHARNESS_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("OPENHARNESS_DATA_DIR", str(tmp_path / "data"))

    client = ThinkingApiClient(thinking="I need to think", text="Done thinking.")
    host = ReactBackendHost(BackendHostConfig(api_client=client))
    host._bundle = await build_runtime(api_client=client)
    events: list[BackendEvent] = []

    async def _emit(event):
        events.append(event)

    host._emit = _emit  # type: ignore[method-assign]
    await start_runtime(host._bundle)
    try:
        await host._process_line("test question")
    finally:
        await close_runtime(host._bundle)

    thinking_events = [e for e in events if e.type == "thinking_delta"]
    assistant_events = [e for e in events if e.type == "assistant_delta"]
    complete_events = [e for e in events if e.type == "assistant_complete"]

    assert len(thinking_events) > 0, "Expected thinking_delta events"
    full_thinking = "".join(e.message or "" for e in thinking_events)
    assert full_thinking == "I need to think"

    assert len(assistant_events) > 0, "Expected assistant_delta events"
    assert len(complete_events) > 0, "Expected assistant_complete events"

    # Verify ordering: thinking events come before assistant events
    first_thinking_idx = next(i for i, e in enumerate(events) if e.type == "thinking_delta")
    first_assistant_idx = next(i for i, e in enumerate(events) if e.type == "assistant_delta")
    assert first_thinking_idx < first_assistant_idx, "Thinking must come before assistant text"
