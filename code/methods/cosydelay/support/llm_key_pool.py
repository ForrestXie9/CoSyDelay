"""Resolve LLM API keys from a local pool file or process environment.

Create ``llm_keys.local.json`` next to this module (see the example file),
put one or more API keys in it, and map experiment slots to keys.  Launch
scripts call :func:`install_api_key_for_slot` before any LLM work.

Priority:
1. Existing non-empty ``LLM_API_KEY`` environment variable (unchanged).
2. Slot assignment from ``llm_keys.local.json``.
3. Legacy recovery from supervisor history (``recover_api_key``).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional

POOL_FILE = Path(__file__).resolve().parent / "llm_keys.local.json"
EXAMPLE_FILE = Path(__file__).resolve().parent / "llm_keys.local.example.json"


def masked_api_key(api_key: str) -> str:
    value = api_key.strip()
    if len(value) <= 10:
        return "***"
    return f"{value[:7]}...{value[-4:]}"


def _normalize_keys(raw_keys: Any) -> list[str]:
    if not isinstance(raw_keys, list) or not raw_keys:
        raise ValueError("keys must be a non-empty list")

    normalized: list[str] = []
    for index, item in enumerate(raw_keys):
        if isinstance(item, str):
            key = item.strip()
            label = str(index)
        elif isinstance(item, dict):
            key = str(item.get("api_key", "")).strip()
            label = str(item.get("label", index))
        else:
            raise ValueError(f"keys[{index}] must be a string or object")
        if not key:
            raise ValueError(f"keys[{index}] ({label}) is empty")
        if any(ord(character) < 32 or ord(character) == 127 for character in key):
            raise ValueError(f"keys[{index}] ({label}) contains control characters")
        normalized.append(key)
    return normalized


def _assignment_index(
    assignments: Mapping[str, Any],
    slot: str,
    key_count: int,
) -> int:
    if slot in assignments:
        target = assignments[slot]
        if isinstance(target, int):
            index = target
        elif isinstance(target, str):
            if target.isdigit():
                index = int(target)
            else:
                raise ValueError(
                    f"assignments[{slot!r}] must be an integer index, got {target!r}"
                )
        else:
            raise ValueError(f"assignments[{slot!r}] must be an integer index")
        if index < 0 or index >= key_count:
            raise ValueError(
                f"assignments[{slot!r}]={index} is out of range for {key_count} key(s)"
            )
        return index

    if "default" in assignments:
        return _assignment_index(assignments, "default", key_count)

    digest = hashlib.sha256(slot.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % key_count


def load_pool_config(path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    pool_path = POOL_FILE if path is None else path.expanduser().resolve()
    if not pool_path.is_file():
        return None
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{pool_path} must contain a JSON object")
    return payload


def resolve_api_key(
    slot: str,
    *,
    pool_path: Optional[Path] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> tuple[str, str]:
    """Return ``(api_key, source_label)`` for one experiment slot."""
    if not slot or not slot.strip():
        raise ValueError("slot must be a non-empty string")
    values = os.environ if environ is None else environ

    env_key = values.get("LLM_API_KEY", "").strip()
    if env_key:
        return env_key, "environment:LLM_API_KEY"

    payload = load_pool_config(pool_path)
    if payload is not None:
        keys = _normalize_keys(payload.get("keys"))
        assignments = payload.get("assignments") or {}
        if assignments and not isinstance(assignments, dict):
            raise ValueError("assignments must be an object when provided")
        index = _assignment_index(assignments, slot.strip(), len(keys))
        return keys[index], f"pool:{slot} -> key[{index}]"

    from methods.cosydelay.engine.data_protocol.operations.run04_pythonw_supervisor import (
        recover_api_key,
    )

    return recover_api_key(), "legacy:recover_api_key"


def install_api_key_for_slot(
    slot: str,
    *,
    pool_path: Optional[Path] = None,
    force: bool = False,
) -> str:
    """Set ``LLM_API_KEY`` for the current process unless already set."""
    existing = os.environ.get("LLM_API_KEY", "").strip()
    if existing and not force:
        source = "environment:LLM_API_KEY"
        api_key = existing
    else:
        api_key, source = resolve_api_key(slot, pool_path=pool_path)
        os.environ["LLM_API_KEY"] = api_key
    print(
        f"[llm-key] slot={slot} source={source} key={masked_api_key(api_key)}",
        flush=True,
    )
    return api_key
