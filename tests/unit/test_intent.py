import json
import sys
import pytest
from unittest.mock import patch, MagicMock
from nexus.agent.intent import (
    ALLOWED_INTENTS, IntentStep, OllamaIntentRouter, OpenAIIntentRouter,
    build_intent_router, parse_intent_plan, repair_intent_plan,
)
from nexus.config.loader import load_default
from nexus.main import extract_app_name, requests_existing_app


def test_intent_model_is_configured():
    cfg = load_default("configs/default.yaml")
    assert cfg["models"]["intent"]["name"] == "gemma3:1b"
    assert cfg["models"]["intent"]["think"] is False
    assert ALLOWED_INTENTS == {"open_app", "computer_task", "shell_task", "clarify"}


def test_extract_app_name():
    assert extract_app_name("open notepad and type hello") == "notepad"
    assert extract_app_name("launch calculator") == "calculator"
    assert extract_app_name("open a terminal and run npm -v") == "terminal"


def test_existing_terminal_request_does_not_require_launch():
    assert requests_existing_app("in my existing terminal enter a new tab and run WSL")
    assert not requests_existing_app("open a terminal and run pwd")


def test_parse_line_and_dot_intent_plan():
    plan = parse_intent_plan(
        "open_app: terminal\n"
        "run_command: ls. run_command: pwd. run_command: echo Hello world"
    )
    assert [(step.kind, step.value) for step in plan] == [
        ("open_app", "terminal"),
        ("run_command", "ls"),
        ("run_command", "pwd"),
        ("run_command", "echo Hello world"),
    ]


def test_parse_plan_discards_schema_placeholder():
    plan = parse_intent_plan(
        "open_app: application name\nrun_command: ls\n"
        "computer_task: short GUI goal\nclarify: question"
    )
    assert [(step.kind, step.value) for step in plan] == [("run_command", "ls")]


def test_parse_plan_repairs_small_model_intent_mislabels():
    plan = parse_intent_plan(
        "run_command: open the terminal\ncomputer_task: run npm -v command"
    )
    assert [(step.kind, step.value) for step in plan] == [
        ("open_app", "terminal"),
        ("run_command", "npm -v"),
    ]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only path stripping test")
def test_parse_plan_removes_unix_shell_prefix_on_windows():
    plan = parse_intent_plan("run_command: /bin/bash pwd")
    assert [(step.kind, step.value) for step in plan] == [("run_command", "pwd")]


def test_repair_plan_does_not_execute_new_tab_as_shell_command():
    plan = repair_intent_plan(
        "in my existing terminal enter a new tab and run WSL",
        [
            IntentStep(kind="open_app", value="Windows Terminal"),
            IntentStep(kind="run_command", value="new tab"),
            IntentStep(kind="clarify", value="Replace every placeholder with the actual value from the request"),
        ],
    )
    assert [(step.kind, step.value) for step in plan] == [
        ("open_app", "Windows Terminal"),
        ("computer_task", "open a new terminal tab"),
        ("run_command", "wsl"),
    ]


# --- build_intent_router tests ---


def test_build_intent_router_defaults_to_ollama():
    router = build_intent_router({"intent": {"name": "gemma3:1b"}})
    assert isinstance(router, OllamaIntentRouter)
    assert router.provider == "ollama"


def test_build_intent_router_openai_provider():
    cfg = {
        "intent": {
            "provider": "openai",
            "name": "gpt-4o-mini",
            "base_url": "https://my-proxy.example.com/v1",
            "temperature": 0,
            "api_key": "sk-test",
        }
    }
    router = build_intent_router(cfg)
    assert isinstance(router, OpenAIIntentRouter)
    assert router.provider == "openai"
    assert router.model == "gpt-4o-mini"
    assert router.base_url == "https://my-proxy.example.com/v1"
    assert router.api_key == "sk-test"
    assert router.temperature == 0


def test_build_intent_router_openai_api_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")
    cfg = {"intent": {"provider": "openai", "name": "gpt-4o-mini"}}
    router = build_intent_router(cfg)
    assert router.api_key == "sk-env-key"


def test_build_intent_router_openai_custom_timeout():
    cfg = {"intent": {"provider": "openai", "name": "gpt-4o-mini", "timeout": 10.0}}
    router = build_intent_router(cfg)
    assert router.timeout == 10.0


def test_build_intent_router_unknown_provider_raises():
    with pytest.raises(ValueError, match="unknown intent provider"):
        build_intent_router({"intent": {"provider": "anthropic"}})


def test_build_intent_router_from_loaded_config():
    cfg = load_default("configs/default.yaml")
    router = build_intent_router(cfg.get("models", {}))
    assert isinstance(router, OllamaIntentRouter)
    assert router.model == "gemma3:1b"


# --- OpenAIIntentRouter HTTP tests (mocked) ---


def _openai_response(content: str) -> dict:
    return {
        "choices": [
            {"message": {"role": "assistant", "content": content}}
        ]
    }


def test_openai_router_parses_intent_plan(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    router = OpenAIIntentRouter(model="gpt-4o-mini", base_url="https://api.example.com/v1")

    fake_body = json.dumps(
        _openai_response("open_app: notepad\nrun_command: echo hello")
    ).encode()

    mock_response = MagicMock()
    mock_response.read.return_value = fake_body
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = lambda s, *a: False

    with patch("nexus.agent.intent.urlopen", return_value=mock_response) as mock_call:
        steps = router.plan("open notepad and type hello")

    assert [(s.kind, s.value) for s in steps] == [
        ("open_app", "notepad"),
        ("run_command", "echo hello"),
    ]
    # Verify request went to the right endpoint
    request_obj = mock_call.call_args[0][0]
    assert request_obj.full_url == "https://api.example.com/v1/chat/completions"
    # Verify auth header
    auth_header = request_obj.get_header("Authorization")
    assert auth_header == "Bearer sk-test"


def test_openai_router_classify(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    router = OpenAIIntentRouter(model="gpt-4o-mini")

    fake_body = json.dumps(_openai_response("run_command: ls")).encode()
    mock_response = MagicMock()
    mock_response.read.return_value = fake_body
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = lambda s, *a: False

    with patch("nexus.agent.intent.urlopen", return_value=mock_response):
        assert router.classify("list files") == "shell_task"


def test_openai_router_no_auth_header_when_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    router = OpenAIIntentRouter(model="gpt-4o-mini", api_key=None)

    fake_body = json.dumps(_openai_response("clarify: ?")).encode()
    mock_response = MagicMock()
    mock_response.read.return_value = fake_body
    mock_response.__enter__ = lambda s: s
    mock_response.__exit__ = lambda s, *a: False

    with patch("nexus.agent.intent.urlopen", return_value=mock_response) as mock_call:
        router.plan("something unclear")

    request_obj = mock_call.call_args[0][0]
    assert request_obj.get_header("Authorization") is None


def test_openai_router_timeout_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    router = OpenAIIntentRouter(model="gpt-4o-mini", timeout=1.0)

    with patch("nexus.agent.intent.urlopen", side_effect=TimeoutError):
        with pytest.raises(RuntimeError, match="timed out"):
            router.plan("list files")
