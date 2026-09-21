from nexus.computer import apps


def test_known_alias_resolves_without_accepting_model_path(monkeypatch):
    monkeypatch.setattr(apps.shutil, "which", lambda value: "C:/Windows/notepad.exe" if value == "notepad.exe" else None)
    assert apps.resolve_app("Notepad") == "C:/Windows/notepad.exe"


def test_unknown_app_is_not_resolved_from_an_arbitrary_path(monkeypatch):
    monkeypatch.setattr(apps.shutil, "which", lambda value: None)
    assert apps.resolve_app("C:/Users/NPC/secret.exe") is None
