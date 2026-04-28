"""Tests for background task management."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from openharness.tasks.manager import BackgroundTaskManager, _encode_task_worker_payload


@pytest.mark.asyncio
async def test_create_shell_task_and_read_output(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENHARNESS_DATA_DIR", str(tmp_path / "data"))
    manager = BackgroundTaskManager()

    task = await manager.create_shell_task(
        command="printf 'hello task'",
        description="hello",
        cwd=tmp_path,
    )

    await asyncio.wait_for(manager._waiters[task.id], timeout=5)  # type: ignore[attr-defined]
    updated = manager.get_task(task.id)
    assert updated is not None
    assert updated.status == "completed"
    assert "hello task" in manager.read_task_output(task.id)


@pytest.mark.asyncio
async def test_create_agent_task_with_command_override_and_write(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENHARNESS_DATA_DIR", str(tmp_path / "data"))
    manager = BackgroundTaskManager()

    task = await manager.create_agent_task(
        prompt="first",
        description="agent",
        cwd=tmp_path,
        command="while read line; do echo \"got:$line\"; break; done",
    )

    await asyncio.wait_for(manager._waiters[task.id], timeout=5)  # type: ignore[attr-defined]
    assert "got:first" in manager.read_task_output(task.id)


@pytest.mark.asyncio
async def test_create_agent_task_preserves_multiline_prompt(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENHARNESS_DATA_DIR", str(tmp_path / "data"))
    manager = BackgroundTaskManager()

    task = await manager.create_agent_task(
        prompt="line 1\nline 2\nline 3",
        description="agent",
        cwd=tmp_path,
        command=(
            "python -u -c \"import sys, json; "
            "print(json.loads(sys.stdin.readline())['text'].replace(chr(10), '|'))\""
        ),
    )

    await asyncio.wait_for(manager._waiters[task.id], timeout=5)  # type: ignore[attr-defined]
    assert "line 1|line 2|line 3" in manager.read_task_output(task.id)


@pytest.mark.asyncio
async def test_write_to_stopped_agent_task_restarts_process(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENHARNESS_DATA_DIR", str(tmp_path / "data"))
    manager = BackgroundTaskManager()

    task = await manager.create_agent_task(
        prompt="ready",
        description="agent",
        cwd=tmp_path,
        command="while read line; do echo \"got:$line\"; break; done",
    )
    await asyncio.wait_for(manager._waiters[task.id], timeout=5)  # type: ignore[attr-defined]

    await manager.write_to_task(task.id, "follow-up")
    await asyncio.wait_for(manager._waiters[task.id], timeout=5)  # type: ignore[attr-defined]

    output = manager.read_task_output(task.id)
    assert "got:ready" in output
    assert "[OpenHarness] Agent task restarted; prior interactive context was not preserved." in output
    assert "got:follow-up" in output
    updated = manager.get_task(task.id)
    assert updated is not None
    assert updated.metadata["restart_count"] == "1"
    assert updated.metadata["status_note"] == "Task restarted; prior interactive context was not preserved."


def test_encode_task_worker_payload_wraps_multiline_text() -> None:
    payload = _encode_task_worker_payload("alpha\nbeta\n")
    assert json.loads(payload.decode("utf-8")) == {"text": "alpha\nbeta"}


def test_encode_task_worker_payload_preserves_structured_messages() -> None:
    raw = '{"text":"follow up","from":"coordinator"}'
    payload = _encode_task_worker_payload(raw)
    assert payload.decode("utf-8") == raw + "\n"


@pytest.mark.asyncio
async def test_stop_task(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENHARNESS_DATA_DIR", str(tmp_path / "data"))
    manager = BackgroundTaskManager()

    task = await manager.create_shell_task(
        command="sleep 30",
        description="sleeper",
        cwd=tmp_path,
    )
    await manager.stop_task(task.id)
    updated = manager.get_task(task.id)
    assert updated is not None
    assert updated.status == "killed"


@pytest.mark.asyncio
async def test_completion_listener_fires_when_task_finishes(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENHARNESS_DATA_DIR", str(tmp_path / "data"))
    manager = BackgroundTaskManager()
    seen: list[tuple[str, str, int | None]] = []
    done = asyncio.Event()

    async def _listener(task):
        seen.append((task.id, task.status, task.return_code))
        done.set()

    manager.register_completion_listener(_listener)

    task = await manager.create_shell_task(
        command="printf 'done'",
        description="listener",
        cwd=tmp_path,
    )

    await asyncio.wait_for(done.wait(), timeout=5)

    assert seen == [(task.id, "completed", 0)]
