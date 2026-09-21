from nexus.computer.keyboard import WindowsKeyboard


class FakeUser32:
    def __init__(self):
        self.events = []

    def keybd_event(self, vk, scan, flags, extra):
        self.events.append((vk, scan, flags, extra))


def test_unicode_text_sends_press_release_for_every_character(monkeypatch):
    user32 = FakeUser32()
    keyboard = WindowsKeyboard(user32=user32)
    monkeypatch.setattr("nexus.computer.keyboard.time.sleep", lambda _: None)
    keyboard.type_text("Hello from NEXUS")
    assert len(user32.events) == 32
    assert user32.events[0][1] == ord("H")
    assert user32.events[-2][1] == ord("S")
    assert all(event[2] in (0x0004, 0x0006) for event in user32.events)
