"""Check that evaluation inputs match the checkpoint's training inputs."""
from __future__ import annotations

import hashlib
from pathlib import Path
import warnings


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint_inputs(
    state: dict,
    *,
    data_config: dict,
    data_sha256: str,
    corpus_summary: dict | None = None,
    topology: dict | None = None,
    check_topology: bool = False,
) -> dict:
    """Reject changed inputs; retain explicitly marked historical compatibility."""
    if state.get("metro_cfg") != data_config:
        raise RuntimeError("Data protocol differs from the checkpoint")
    if corpus_summary is not None and state.get("corpus_summary") != corpus_summary:
        raise RuntimeError("Corpus summary differs from the checkpoint")
    recorded_hash = state.get("data_sha256")
    if recorded_hash is not None and recorded_hash != data_sha256:
        raise RuntimeError("Data SHA-256 differs from the checkpoint")
    if check_topology:
        # Identical files may be relocated; only their content and settings matter.
        def identity(metadata):
            return None if metadata is None else {
                key: value for key, value in metadata.items() if key != "path"
            }
        if identity(state.get("topology")) != identity(topology):
            raise RuntimeError("Topology content or settings differ from the checkpoint")
    if recorded_hash is None:
        warnings.warn(
            "Historical checkpoint has no data SHA-256; exact data identity cannot be verified.",
            RuntimeWarning,
            stacklevel=2,
        )
    return {
        "data_sha256_verified": recorded_hash is not None,
        "data_protocol_verified": True,
        "corpus_summary_verified": corpus_summary is not None,
        "topology_verified": check_topology,
    }
