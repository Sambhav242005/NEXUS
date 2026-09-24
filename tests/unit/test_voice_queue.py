"""Speak-while-busy queue/worker tests — pure Python, no mic."""
import threading
import time

import pytest

from nexus.voice.audio_loop import (
    Command,
    PendingQueue,
    ack_text,
    is_exit_phrase,
    normalize_utterance,
    pump,
)


def test_exit_phrases():
    assert is_exit_phrase("Exit.")
    assert is_exit_phrase("  STOP LISTENING  ")
    assert is_exit_phrase("quit!")
    assert not is_exit_phrase("don't stop")
    assert not is_exit_phrase("open notepad")
    assert normalize_utterance("  Hello... ") == "hello"


def test_ack_text_names_position():
    assert "2" in ack_text(2)


def test_queue_fifo_and_positions():
    q = PendingQueue(max_pending=2)
    assert q.submit("a", "a.wav") == ("queued", 1)
    assert q.submit("b", "b.wav") == ("queued", 2)
    assert q.submit("c", "c.wav") == ("full", 2)
    first = q.get(timeout=0.1)
    assert isinstance(first, Command) and first.text == "a" and first.seq == 1
    q.task_done()
    assert q.submit("c", "c.wav") == ("queued", 2)
    assert q.pending == 2


def test_queue_rejects_bad_size():
    with pytest.raises(ValueError):
        PendingQueue(max_pending=0)


def _run_pump(commands, **kwargs):
    stop = threading.Event()
    done: list = []
    thread = threading.Thread(
        target=lambda: done.append(pump(commands, stop_event=stop, **kwargs)),
        daemon=True,
    )
    thread.start()
    return stop, thread, done


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_pump_processes_fifo_and_continues_past_failures():
    q = PendingQueue(max_pending=5)
    replies: list[tuple[str, str]] = []

    def on_task(text):
        if text == "boom":
            raise RuntimeError("task blew up")
        return f"did:{text}"

    busy = threading.Event()
    stop, thread, done = _run_pump(
        q,
        on_task=on_task,
        on_reply=lambda reply, cmd: replies.append((cmd.text, reply)),
        busy=busy,
    )
    q.submit("one", "1.wav")
    q.submit("boom", "2.wav")
    q.submit("three", "3.wav")
    assert _wait_for(lambda: len(replies) == 3), replies
    stop.set()
    thread.join(timeout=5)
    assert [text for text, _ in replies] == ["one", "boom", "three"]
    assert replies[1] == ("boom", "The task failed.")
    assert replies[2] == ("three", "did:three")
    assert done == [3]


def test_pump_busy_flag_tracks_work():
    q = PendingQueue(max_pending=5)
    busy = threading.Event()
    observed: list[bool] = []
    release = threading.Event()

    def on_task(text):
        observed.append(busy.is_set())
        assert release.wait(timeout=5)
        return "ok"

    stop, thread, done = _run_pump(
        q, on_task=on_task, on_reply=lambda r, c: None, busy=busy,
    )
    q.submit("work", "w.wav")
    assert _wait_for(lambda: len(observed) == 1), observed
    assert observed == [True]  # busy set while task runs
    release.set()
    assert _wait_for(lambda: not busy.is_set())  # cleared after
    stop.set()
    thread.join(timeout=5)
    assert done == [1]


def test_pump_stops_leaving_pending():
    q = PendingQueue(max_pending=5)
    stop = threading.Event()
    stop.set()  # stop before any work
    processed = pump(q, on_task=lambda t: "x",
                     on_reply=lambda r, c: None, stop_event=stop)
    assert processed == 0
