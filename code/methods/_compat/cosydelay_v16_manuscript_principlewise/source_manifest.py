"""Packaged source manifest for the public method."""

from __future__ import annotations

from methods.cosydelay_v9_clean_from_scratch.source_manifest import (
    GMINI,
    SOURCE_FILES as V16_SOURCE_FILES,
    validate_source_manifest,
)

validate_v16_source_manifest = validate_source_manifest

__all__ = ["GMINI", "V16_SOURCE_FILES", "validate_v16_source_manifest"]
