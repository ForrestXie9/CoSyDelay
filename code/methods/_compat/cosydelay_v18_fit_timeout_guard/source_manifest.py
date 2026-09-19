"""Packaged source manifest compatibility wrapper."""

from __future__ import annotations

from methods.cosydelay_v17_cross_batch_retry.source_manifest import (
    GMINI,
    V17_SOURCE_FILES as V18_SOURCE_FILES,
    validate_v17_source_manifest,
)

validate_v18_source_manifest = validate_v17_source_manifest

__all__ = ["GMINI", "V18_SOURCE_FILES", "validate_v18_source_manifest"]
