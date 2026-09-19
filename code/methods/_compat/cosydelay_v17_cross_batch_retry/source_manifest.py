"""Packaged source manifest compatibility wrapper."""

from __future__ import annotations

from methods.cosydelay_v16_manuscript_principlewise.source_manifest import (
    GMINI,
    V16_SOURCE_FILES as V17_SOURCE_FILES,
    validate_v16_source_manifest,
)

validate_v17_source_manifest = validate_v16_source_manifest

__all__ = ["GMINI", "V17_SOURCE_FILES", "validate_v17_source_manifest"]
