from nexus.agent.intent import ALLOWED_INTENTS, IntentStep, parse_intent_plan, repair_intent_plan
from nexus.config.loader import load_default
from nexus.main import extract_app_name, requests_existing_app


def test_intent_model_is_configured():
    cfg = load_default("configs/default.yaml")
    assert cfg["models"]["intent"]["name"] == "minicpm-v4.6"
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
