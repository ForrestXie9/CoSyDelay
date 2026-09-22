from __future__ import annotations

import unittest

import expression_rules

from .denominator_policy import has_proven_interior_denominator_zero
from . import prompt
from .prompt_audit import audit_v21_prompts


class DenominatorPolicyTests(unittest.TestCase):
    def check(self, expression: str) -> bool:
        parsed, symbols = expression_rules.parse_symbolic_expression(expression)
        return has_proven_interior_denominator_zero(parsed, symbols)

    def test_positive_denominators_pass(self) -> None:
        self.assertFalse(self.check("Cycle_Time / GR_phase + a1 * flow_lane"))
        self.assertFalse(self.check("Cycle_Time / (a1 + GR_phase)"))

    def test_constructively_demonstrated_zeros_reject(self) -> None:
        self.assertTrue(self.check("Cycle_Time / (GR_phase - a1)"))
        self.assertTrue(self.check("Cycle_Time / (1 - a1 * GR_phase)"))


class FormalPromptTests(unittest.TestCase):
    def test_only_selected_numeric_domain_is_present_and_auditable(self) -> None:
        init = prompt.build_init_prompt({}, [], 1)
        regen = prompt.build_regeneration_prompt(
            "Cycle_Time*(a1+a2*flow_lane/GR_phase)", "", "", (True, "Valid"),
            "large", [], 1,
        )
        for value in (init, regen):
            self.assertIn("0.001<=flow_lane<=2.0", value)
            self.assertNotIn("never beyond a8", value)
            self.assertNotIn("retaining useful parent mechanisms", value)
        audit = audit_v21_prompts([
            {"status": "success", "prompt": init},
            {"status": "success", "prompt": regen},
        ])
        self.assertTrue(audit["numeric_operating_domain_present"])


if __name__ == "__main__":
    unittest.main()
