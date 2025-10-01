"""Unit tests for the desktop GUI API key store."""
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desktop_app.api_key_store import ApiKeyStore


def test_load_returns_empty_when_file_missing(tmp_path: Path) -> None:
    store = ApiKeyStore(tmp_path / "keys.json")
    assert store.load() == []


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    store = ApiKeyStore(tmp_path / "keys.json")
    store.save([" key-one ", "key-two", "key-one", "", None])

    loaded = store.load()
    assert loaded == ["key-one", "key-two"]


def test_load_gracefully_handles_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "keys.json"
    path.write_text("not json", encoding="utf-8")

    store = ApiKeyStore(path)
    assert store.load() == []
