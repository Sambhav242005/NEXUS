from nexus.agent.loop import AutonomousLoop
from nexus.computer.controller import ComputerController
from nexus.vision.schemas import ScreenObservation, TaskContext


class FakeVision:
    def analyze(self, path, context):
        return ScreenObservation(visible_app="test")


class FakePlanner:
    def propose(self, context: TaskContext, observation):
        return type("Step", (), {"tool": "computer", "action": {"action": "keypress", "key": "enter"},
                                  "requires_confirmation": False, "expected_result": "done"})()


class FakeController(ComputerController):
    def __init__(self):
        super().__init__(enabled=False)
        self.executed = []
        self.screenshots = 0

    def capture_screenshot(self):
        from nexus.computer.controller import ControllerResult
        self.screenshots += 1
        return ControllerResult(ok=True, detail="screen", screenshot_path="screen.png")

    def execute(self, action):
        from nexus.computer.controller import ControllerResult
        self.executed.append(action)
        return ControllerResult(ok=True, detail="done")


def test_loop_executes_one_validated_action():
    controller = FakeController()
    result = AutonomousLoop(controller, FakeVision(), FakePlanner(), max_steps=3).run("press enter")
    assert len(result) == 1
    assert controller.executed[0].action == "keypress"
    assert controller.screenshots == 2


def test_loop_stops_after_failed_action():
    class FailedController(FakeController):
        def execute(self, action):
            from nexus.computer.controller import ControllerResult
            self.executed.append(action)
            return ControllerResult(ok=False, detail="input failed")

    controller = FailedController()
    result = AutonomousLoop(controller, FakeVision(), FakePlanner(), max_steps=3).run("press enter")
    assert len(result) == 1
    assert result[0].ok is False
    assert controller.screenshots == 1


def test_loop_accepts_model_type_alias_at_boundary():
    class AliasPlanner(FakePlanner):
        def propose(self, context, observation):
            step = super().propose(context, observation)
            step.action = {"type": "keypress", "key": "enter"}
            return step

    controller = FakeController()
    AutonomousLoop(controller, FakeVision(), AliasPlanner(), max_steps=3).run("press enter")
    assert controller.executed[0].action == "keypress"
