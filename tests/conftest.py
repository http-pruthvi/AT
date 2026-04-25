import os
import pytest
import tempfile
import shutil


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Provide a temporary directory with profiles and logs subdirs."""
    profiles = tmp_path / "profiles"
    logs = tmp_path / "logs"
    profiles.mkdir()
    logs.mkdir()
    return tmp_path


@pytest.fixture
def subjects_dir():
    """Return the real subjects directory path."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "subjects")
