from nexus.agent.loop import AutonomousLoop
from nexus.computer.controller import ComputerController, ControllerResult
from nexus.vision.schemas import ScreenObservation, TaskContext


class Controller(ComputerController):
    def __init__(self):
        super().__init__(enabled=False)
        self.executed = []

    def capture_screenshot(self):
        return ControllerResult(ok=True, detail="screen", screenshot_path="screen.png")

    def accessibility_elements(self):
        return []

    def execute(self, action):
        self.executed.append(action)
        return ControllerResult(ok=True, detail="done")


class Planner:
    def propose(self, context: TaskContext, observation: ScreenObservation):
        return type("Step", (), {
            "tool": "computer", "action": {"action": "keypress", "key": "enter"},
            "requires_confirmation": False, "expected_result": "done",
        })()


def test_loop_can_run_without_vision_model():
    controller = Controller()
    results = AutonomousLoop(controller, None, Planner(), max_steps=1).run("press enter")
    assert len(results) == 1
