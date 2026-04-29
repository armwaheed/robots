"""Tests for asset cache path resolution."""

from pathlib import Path

from strands_robots import assets


class TestAssetCache:
    def test_custom_assets_dir_is_used(self, monkeypatch, tmp_path):
        custom_dir = tmp_path / "custom_assets"
        monkeypatch.setenv("STRANDS_ASSETS_DIR", str(custom_dir))

        resolved = assets.get_assets_dir()

        assert resolved == custom_dir
        assert resolved.exists()

    def test_default_assets_dir_falls_back_when_home_cache_is_blocked(self, monkeypatch, tmp_path):
        blocked = Path("/blocked/home/.strands_robots/assets")
        fallback = tmp_path / "fallback_assets"

        monkeypatch.delenv("STRANDS_ASSETS_DIR", raising=False)
        monkeypatch.setattr(assets, "_USER_CACHE_DIR", blocked)
        monkeypatch.setattr(assets, "_default_assets_dir_candidates", lambda: [blocked, fallback])

        original_mkdir = Path.mkdir

        def fake_mkdir(self, parents=False, exist_ok=False):
            if self == blocked:
                raise PermissionError("blocked")
            return original_mkdir(self, parents=parents, exist_ok=exist_ok)

        monkeypatch.setattr(Path, "mkdir", fake_mkdir)

        resolved = assets.get_assets_dir()

        assert resolved == fallback
        assert resolved.exists()
