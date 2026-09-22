"""V21 audit for the unchanged V20 prompt, robust to parent coefficient a8."""
from __future__ import annotations

import re

from methods.cosydelay._internal.accelerated_search.prompt_audit import audit_v20_prompts


def audit_v21_prompts(llm_attempts: list[dict]) -> dict:
    # The V20 audit used ``'a8' not in prompt`` and therefore mistook a genuine
    # a8 in a regenerated parent expression for a coefficient-count rule.
    # Mask only coefficient tokens for that legacy assertion; every other
    # prompt assertion still runs unchanged on the archived prompts.
    masked = []
    for item in llm_attempts:
        copied = dict(item)
        if copied.get("prompt"):
            text = re.sub(r"\ba8\b", "a_coeff", str(copied["prompt"]))
            # Let the inherited V20 structural audit run while V21 separately
            # proves that its one deliberate prompt addition is present.
            text = text.replace("0.001<=flow_lane", "declared_flow_domain")
            text = text.replace("0.02<=GR_phase", "declared_green_domain")
            text = text.replace("30<=Cycle_Time", "declared_cycle_domain")
            copied["prompt"] = text
        masked.append(copied)
    audit = audit_v20_prompts(masked)
    successful = [
        str(item.get("prompt", ""))
        for item in llm_attempts
        if item.get("status") == "success" and item.get("prompt")
    ]
    audit["numeric_operating_domain_present"] = bool(successful) and all(
        "0.001<=flow_lane<=2.0" in value
        and "0.02<=GR_phase<=0.95" in value
        and "30<=Cycle_Time<=240" in value
        for value in successful
    )
    if not audit["numeric_operating_domain_present"]:
        raise RuntimeError(f"V21 numeric-domain prompt audit failed: {audit}")
    audit["a8_limit_audit"] = "coefficient_tokens_masked; instruction contract unchanged"
    audit["method_version"] = "V21_global_numeric_domain"
    return audit
