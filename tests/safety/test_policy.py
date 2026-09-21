"""Safety + schema tests."""
import pytest
from nexus.safety.policy import check_action
from nexus.computer.schemas import ComputerAction
from nexus.agent.schemas import PlanStep


def test_destructive_needs_confirm():
    assert check_action("shell", {"cmd": "rm -rf /tmp/x"}) == "confirm"
    assert check_action("filesystem", {"op": "delete", "path": "/tmp/x"}) == "confirm"
    assert check_action("browser", {"op": "send email"}) == "confirm"


def test_unknown_tool_denied():
    assert check_action("teleport", {"x": 1}) == "deny"


def test_window_mutations_require_confirmation():
    assert check_action("computer", {"action": "close_window"}) == "confirm"
    assert check_action("computer", {"action": "save"}) == "confirm"


def test_read_only_computer_actions_are_allowed():
    assert check_action("computer", {"action": "screenshot"}) == "allow"
    assert check_action("computer", {"action": "click", "x": 10, "y": 20}) == "allow"


def test_malformed_computer_rejected():
    with pytest.raises(Exception):
        ComputerAction(action="click", x=10)  # missing y
    with pytest.raises(Exception):
        ComputerAction(action="click", x=-5, y=10)  # negative
    with pytest.raises(Exception):
        ComputerAction(action="type")  # missing text
    ok = ComputerAction(action="click", x=10, y=20)
    assert ok.x == 10


def test_plan_step_validation():
    s = PlanStep(goal="open chrome", tool="computer", action={"action": "click"}, expected_result="chrome opens", requires_confirmation=False)
    assert s.tool == "computer"
