import json
import os
import threading
from core.session_manager import SessionManager


def test_profile_roundtrip(tmp_path):
    manager = SessionManager()
    manager.profiles_dir = str(tmp_path / "profiles")
    manager.logs_dir = str(tmp_path / "logs")
    os.makedirs(manager.profiles_dir, exist_ok=True)
    os.makedirs(manager.logs_dir, exist_ok=True)
    manager.reset("Alice", "Math")
    manager.questions_asked = 3
    manager.correct_answers = 2
    assert manager.save_profile() is True

    loaded = SessionManager()
    loaded.profiles_dir = manager.profiles_dir
    loaded.logs_dir = manager.logs_dir
    loaded.reset("Alice", "Math")
    assert loaded.load_profile() is True
    assert loaded.subject == "Math"


def test_atomic_concurrent_profile_writes(tmp_path):
    manager = SessionManager()
    manager.profiles_dir = str(tmp_path / "profiles")
    manager.logs_dir = str(tmp_path / "logs")
    os.makedirs(manager.profiles_dir, exist_ok=True)
    os.makedirs(manager.logs_dir, exist_ok=True)
    manager.reset("ConcurrentUser", "Science")

    def _worker(idx):
        manager.questions_asked = idx
        manager.correct_answers = idx // 2
        manager.save_profile()

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(1, 25)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    profile_path = os.path.join(manager.profiles_dir, "concurrentuser.json")
    with open(profile_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "history" in data
    assert data["subject"] == "Science"

