from nexus.agent.candidate_planner import SemIfCandidatePlanner
from nexus.agent.intent import IntentStep
from nexus.main import build_semif_engine
from nexus.vision.schemas import ScreenObservation, TaskContext


def test_new_file_task_starts_with_new_document_hotkey():
    planner = SemIfCandidatePlanner(build_semif_engine({"backend": "mock"}))
    step = planner.propose(
        TaskContext(task="open notepad and create a new file then type Hello", step=0),
        ScreenObservation(visible_app="Notepad"),
    )
    assert step.action == {"action": "hotkey", "keys": ["ctrl", "n"]}


def test_new_file_task_types_on_second_step():
    planner = SemIfCandidatePlanner(build_semif_engine({"backend": "mock"}))
    step = planner.propose(
        TaskContext(task="open notepad and create a new file then type Hello", step=1),
        ScreenObservation(visible_app="Notepad"),
    )
    assert step.action == {"action": "type", "text": "Hello"}


def test_multiple_terminal_commands_are_sequenced():
    planner = SemIfCandidatePlanner(build_semif_engine({"backend": "mock"}))
    task = "open terminal and run ls command and also run pwd command"
    screen = ScreenObservation(visible_app="PowerShell")
    assert planner.propose(TaskContext(task=task, step=0), screen).action == {"action": "type", "text": "ls"}
    assert planner.propose(TaskContext(task=task, step=1), screen).action == {"action": "keypress", "key": "enter"}
    assert planner.propose(TaskContext(task=task, step=2), screen).action == {"action": "type", "text": "pwd"}
    assert planner.propose(TaskContext(task=task, step=3), screen).action == {"action": "keypress", "key": "enter"}


def test_gemma_intent_plan_preserves_all_commands():
    steps = [
        IntentStep(kind="open_app", value="terminal"),
        IntentStep(kind="run_command", value="ls"),
        IntentStep(kind="run_command", value="pwd"),
        IntentStep(kind="run_command", value="echo Hello world"),
    ]
    planner = SemIfCandidatePlanner(build_semif_engine({"backend": "mock"}), steps)
    task = "open terminal and run ls command and also run pwd command and also echo Hello world"
    screen = ScreenObservation(visible_app="PowerShell")
    assert planner.propose(TaskContext(task=task, step=4), screen).action == {
        "action": "type", "text": "echo Hello world"
    }
    assert planner.propose(TaskContext(task=task, step=5), screen).action == {
        "action": "keypress", "key": "enter"
    }


def test_terminal_typing_requires_enter_before_completion():
    steps = [IntentStep(kind="run_command", value="pwd")]
    planner = SemIfCandidatePlanner(build_semif_engine({"backend": "mock"}), steps)
    task = "open terminal and run pwd command"
    screen = ScreenObservation(visible_app="PowerShell")
    assert planner.propose(TaskContext(task=task, step=0), screen).expected_result == "needs_more_observation"
    assert planner.propose(TaskContext(task=task, step=1), screen).expected_result == "completed"


def test_new_terminal_tab_is_hotkey_then_wsl():
    steps = [
        IntentStep(kind="open_app", value="Windows Terminal"),
        IntentStep(kind="computer_task", value="open a new terminal tab"),
        IntentStep(kind="run_command", value="wsl"),
    ]
    planner = SemIfCandidatePlanner(build_semif_engine({"backend": "mock"}), steps)
    task = "in my existing terminal enter a new tab and run WSL"
    screen = ScreenObservation(visible_app="Windows Terminal")
    assert planner.propose(TaskContext(task=task, step=0), screen).action == {
        "action": "hotkey", "keys": ["ctrl", "shift", "t"]
    }
    assert planner.propose(TaskContext(task=task, step=1), screen).action == {
        "action": "type", "text": "wsl"
    }
    assert planner.propose(TaskContext(task=task, step=2), screen).action == {
        "action": "keypress", "key": "enter"
    }
