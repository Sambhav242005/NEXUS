"""Instruction -> text-to-write extraction (voice drives agent)."""
from nexus.main import extract_text_to_write, classify_intent


def test_write_quoted():
    assert extract_text_to_write('open notepad and write "hello how are you"') == "hello how are you"


def test_write_bare():
    assert extract_text_to_write("create a file where you write hello world") == "hello world"


def test_fallback_default():
    assert extract_text_to_write("open chrome") == "Hello world"


def test_classify_terminal():
    assert classify_intent("Open a new terminal for me.") == "terminal"
    assert classify_intent("launch powershell") == "terminal"


def test_classify_notepad():
    assert classify_intent("open notepad and write hi") == "notepad"


def test_classify_unsupported():
    assert classify_intent("book me a flight to Paris") == "unsupported"
