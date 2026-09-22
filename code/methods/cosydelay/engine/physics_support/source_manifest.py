"""Packaged source manifest compatibility wrapper."""

from __future__ import annotations

from methods.cosydelay.engine.data_protocol.source_manifest import (
    GMINI,
    SOURCE_FILES as V14_SOURCE_FILES,
    validate_source_manifest,
)

validate_v14_source_manifest = validate_source_manifest

__all__ = ["GMINI", "V14_SOURCE_FILES", "validate_v14_source_manifest"]
