from importlib.metadata import PackageNotFoundError, version

import codex_manager


def test_version_matches_installed_package_metadata() -> None:
    try:
        installed_version = version("warpfoundry")
    except PackageNotFoundError:
        try:
            installed_version = version("codex-manager")
        except PackageNotFoundError:
            assert codex_manager.__version__ == "0.0.0"
        else:
            assert codex_manager.__version__ == installed_version
    else:
        assert codex_manager.__version__ == installed_version
