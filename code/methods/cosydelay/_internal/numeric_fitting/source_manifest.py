"""Packaged source manifest compatibility wrapper."""

from __future__ import annotations

from methods.cosydelay._internal.data_protocol.source_manifest import (
    GMINI,
    SOURCE_FILES as V15_SOURCE_FILES,
    validate_source_manifest,
)

validate_v15_source_manifest = validate_source_manifest

__all__ = ["GMINI", "V15_SOURCE_FILES", "validate_v15_source_manifest"]
