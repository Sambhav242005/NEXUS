"""Recorder logic tests — headless, no tkinter."""
import time
from pathlib import Path
from nexus.ui.recorder import TaskRecorder


class FakeCtl:
    def __init__(self):
        self.calls = 0

    def execute(self, action):
        self.calls += 1

        class R:
            detail = "screenshot 1920x1080"
        return R()


def test_start_stop_records_frames(tmp_path: Path):
    rec = TaskRecorder(FakeCtl(), workdir=tmp_path, interval=0.05)
    r = rec.start("open notepad hello")
    time.sleep(0.22)
    out = rec.stop()
    assert out.task == "open notepad hello"
    assert len(out.frames) >= 2
    assert (out.dir / "task.txt").read_text() == "open notepad hello"
    assert (out.dir / "frames.log").exists()


def test_double_start_rejected(tmp_path: Path):
    rec = TaskRecorder(FakeCtl(), workdir=tmp_path, interval=0.05)
    rec.start("a")
    try:
        import pytest
        with pytest.raises(RuntimeError):
            rec.start("b")
    finally:
        rec.stop()


def test_stop_without_start_rejected(tmp_path: Path):
    import pytest
    rec = TaskRecorder(FakeCtl(), workdir=tmp_path)
    with pytest.raises(RuntimeError):
        rec.stop()
