"""Centralized prompt catalog and optimizer.

All prompts used throughout WarpFoundry are stored in ``templates.yaml``
(next to this module) and loaded by :class:`PromptCatalog`.  This enables:

* **One-place editing** — every prompt the system uses lives in a single YAML file.
* **AI optimization** — ``warpfoundry optimize-prompts`` refines each
  prompt using a strong model.
* **Versioning** — the YAML file is version-controlled; edits are tracked.
"""

from codex_manager.prompts.catalog import PromptCatalog
from codex_manager.prompts.optimizer import PromptOptimizer

__all__ = ["PromptCatalog", "PromptOptimizer"]
