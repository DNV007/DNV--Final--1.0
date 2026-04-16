"""
DNV Scientific Module
---------------------
Role:
    Provides SessionStore — the single in-memory DataAsset registry with
    explicit persistence to ~/.dnv/.

Scientific Context:
    Holds one DataAsset | None representing the current pipeline state;
    auto-persists on write so the asset survives application restarts.

Invariants:
    - At most one DataAsset is held at any time; replacing it atomically
      discards the previous.
    - store.asset is None until a successful ingestion completes.

Assumptions:
    - ~/.dnv/ is writable on the host filesystem.
    - DataAsset is a frozen dataclass importable from ing_provenance.

Failure Modes:
    - Disk write failure: logged; in-memory state is preserved.
    - Corrupt persisted file on restore: logged; store initialises with
      asset=None.

Provenance:
    - Emits no transformation metadata; all provenance is carried in the
      DataAsset itself.
"""
from __future__ import annotations

import logging, os

from ing_provenance import persist_data_asset, restore_data_asset

log = logging.getLogger(__name__)

SESSION_FILE = os.path.join(os.path.expanduser("~"), ".dnv", "session_store.dnvasset")


class SessionStore:
    """Central in-memory dataset registry with explicit persistence.

    Design: explicit lifecycle, survives window close, corrupted bundles
    are logged and discarded rather than crashing the GUI.
    """

    def __init__(self, persistence_path: str = SESSION_FILE):
        self._path  = persistence_path
        self._asset = None
        self._seal_result = None   # SEALResult, in-memory only (not persisted)
        self._restore()

    @property
    def asset(self):
        return self._asset

    @property
    def seal_result(self):
        """Most recent SEALResult, or None.  In-memory only — not persisted."""
        return self._seal_result

    def set_seal_result(self, sr) -> None:
        self._seal_result = sr

    def has_asset(self) -> bool:
        return self._asset is not None

    def set_asset(self, asset) -> None:
        self._asset = asset
        self.persist()

    def clear(self) -> None:
        self._asset = None
        self._seal_result = None
        try:
            os.remove(self._path)
        except FileNotFoundError:
            pass

    def persist(self) -> None:
        if self._asset is None:
            return
        try:
            persist_data_asset(self._asset, self._path)
        except Exception:
            log.exception("SessionStore.persist failed")

    def _restore(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            self._asset = restore_data_asset(self._path)
        except KeyError as e:
            log.warning("SessionStore: stale session payload (missing key %s); clearing.", e)
            self._asset = None
        except Exception:
            log.exception("SessionStore: restore failed; clearing corrupted persistence")
            self._asset = None
