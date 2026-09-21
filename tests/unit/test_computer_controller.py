import pytest

from nexus.computer.controller import ComputerController
from nexus.computer.schemas import ComputerAction


class FakeMouse:
    def __init__(self):
        self.calls = []

    def move(self, x, y): self.calls.append(("move", x, y))
    def click(self, x, y, count=1, button="left"): self.calls.append(("click", x, y, count, button))
    def drag(self, x, y, x2, y2): self.calls.append(("drag", x, y, x2, y2))
    def scroll(self, x, y, amount): self.calls.append(("scroll", x, y, amount))


class FakeKeyboard:
    def __init__(self):
        self.calls = []

    def type_text(self, text): self.calls.append(("type", text))
    def keypress(self, key): self.calls.append(("keypress", key))
    def hotkey(self, keys): self.calls.append(("hotkey", keys))


def test_enabled_controller_dispatches_real_input_to_injected_backends():
    mouse, keyboard = FakeMouse(), FakeKeyboard()
    controller = ComputerController(".", enabled=True, mouse=mouse, keyboard=keyboard)

    assert controller.execute(ComputerAction(action="move", x=10, y=20)).ok
    assert controller.execute(ComputerAction(action="double_click", x=10, y=20)).ok
    assert controller.execute(ComputerAction(action="right_click", x=10, y=20)).ok
    assert controller.execute(ComputerAction(action="middle_click", x=10, y=20)).ok
    assert controller.execute(ComputerAction(action="drag", x=10, y=20, x2=30, y2=40)).ok
    assert controller.execute(ComputerAction(action="scroll", x=10, y=20, text="-120")).ok
    assert controller.execute(ComputerAction(action="type", text="hello")).ok
    assert controller.execute(ComputerAction(action="keypress", key="enter")).ok
    assert controller.execute(ComputerAction(action="hotkey", keys=["ctrl", "l"])).ok
    assert mouse.calls == [
        ("move", 10, 20),
        ("click", 10, 20, 2, "left"),
        ("click", 10, 20, 1, "right"),
        ("click", 10, 20, 1, "middle"),
        ("drag", 10, 20, 30, 40),
        ("scroll", 10, 20, -120),
    ]
    assert keyboard.calls == [("type", "hello"), ("keypress", "enter"), ("hotkey", ["ctrl", "l"])]


def test_scroll_requires_coordinates_and_delta():
    with pytest.raises(Exception):
        ComputerAction(action="scroll", x=10, y=20)


def test_action_schema_rejects_invalid_drag_and_resize():
    with pytest.raises(Exception):
        ComputerAction(action="drag", x=10, y=20, x2=30)
    with pytest.raises(Exception):
        ComputerAction(action="resize", x=10, y=20, width=0, height=100)
