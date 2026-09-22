"""Packaged source manifest compatibility wrapper."""

from __future__ import annotations

from methods.cosydelay._internal.data_protocol.source_manifest import (
    GMINI,
    SOURCE_FILES,
    validate_source_manifest,
)

__all__ = ["GMINI", "SOURCE_FILES", "validate_source_manifest"]
