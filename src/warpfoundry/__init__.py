"""WarpFoundry compatibility package.

This package exposes the same public version string as ``codex_manager``
while the internal module namespace remains ``codex_manager``.
"""

from codex_manager import __version__

__all__ = ["__version__"]
