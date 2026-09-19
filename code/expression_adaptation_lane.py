# expression_adaptation_lane.py
"""
Generate universal lane-level expressions using LLM
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
from typing import List, Tuple, Optional
from llm_integration import run_llm
from expression_rules import (
    canonical_expression,
    parse_symbolic_expression,
    validate_candidate_expression,
    validate_candidate_legality,
)


_EXPRESSION_AUDIT_LOCK = threading.Lock()


def _fs_path(path: Path | str) -> Path:
    """Windows extended-length path so audit appends survive MAX_PATH."""
    text = os.path.abspath(str(path))
    if os.name != "nt" or text.startswith("\\\\?\\"):
        return Path(text)
    if text.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + text[2:])
    return Path("\\\\?\\" + text)


def _write_expression_attempt_audit(record: dict) -> None:
    """Append one non-secret generation outcome when audit is enabled."""
    raw_path = os.environ.get("EXPRESSION_ATTEMPT_AUDIT_LOG", "").strip()
    if not raw_path:
        return
    path = Path(raw_path).expanduser()
    fs = _fs_path(path)
    fs.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        **record,
    }
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    with _EXPRESSION_AUDIT_LOCK:
        with fs.open("a", encoding="utf-8", newline="") as handle:
            handle.write(line)
            handle.flush()


STRUCTURAL_RULES = """
GENERAL STRUCTURAL SAFETY RULES:
- The expression must remain finite, real, and non-negative throughout the
  operational domain: 0.001<=flow_lane<=1.2 and 0.04<=GR_phase<=0.85.
- Denominators must stay strictly positive throughout the operational domain;
  never use a subtractive denominator such as 1-k*flow_lane/GR_phase.
- A base raised to a fitted exponent must remain non-negative over the full input domain.
- log arguments must remain strictly positive over the full input domain.
- exp is allowed, but the complete expression must stay finite in the
  operational domain.
- y must be a non-constant function of flow_lane and of GR_phase over the
  declared operational domain.
- Prefer positive constructions using sums, products, 1+x, log(1+x), and positive powers.
- Use no more than 8 fitted coefficients and keep the expression compact.
"""

DIMENSIONAL_RULES = """
MANDATORY DIMENSIONAL CONSISTENCY:
- The complete output must have time exponent one. flow_lane, GR_phase, and all
  fitted coefficients are dimensionless; Cycle_Time is measured in seconds.
- Every additive term must therefore reduce algebraically to Cycle_Time times a
  dimensionless function.
- Arguments of exp() and log() and variable exponents must be dimensionless.
"""

INTERPRETABILITY_AND_PARSIMONY_RULES = """
TRAFFIC INTERPRETABILITY AND MINIMUM NECESSARY COMPLEXITY:
- Use the simplest structure that represents the intended traffic mechanism;
  never add a term merely to appear novel or mathematically sophisticated.
- Every additive term or nonlinear factor must have a clear role such as
  demand amplification, green-ratio attenuation, or a non-negative baseline.
- Prefer fewer terms and 3--6 fitted coefficients. Use 7--8 coefficients only
  when each additional coefficient controls a distinct, explainable mechanism.
- Avoid redundant or canceling terms, nested nonlinear functions, and multiple
  transformations that describe the same effect.
- Use exp, log, or fitted powers only when they express a distinct plausible
  traffic response that a simpler positive algebraic form cannot represent.
- In the explanation, map every term to its traffic meaning and expected
  response direction. If a term cannot be interpreted, remove it.
"""

PHYSICAL_REQUIREMENTS_STANDARD = """
CRITICAL PHYSICAL REQUIREMENTS (mathematical form):
1. Use coefficients a1, a2, a3, etc. (these will be optimized separately for each lane).
2. Use every declared input: flow_lane, GR_phase, and Cycle_Time.
3. Flow monotonicity: at fixed GR_phase and Cycle_Time, y is nondecreasing in
   flow_lane (delay must not decrease as demand increases).
4. Green monotonicity: at fixed flow_lane and Cycle_Time, y is nonincreasing in
   GR_phase (delay must not increase as green time increases).
5. Time dimension: y has units [seconds]; every additive term reduces to
   Cycle_Time times a dimensionless function.
6. Non-negativity and finiteness: throughout 0.001<=flow_lane<=1.2,
   0.04<=GR_phase<=0.85, and 60<=Cycle_Time<=180, y is finite, real, and >= 0.
7. Non-degenerate response: y must genuinely depend on both flow_lane and
   GR_phase over the operational domain (not constant in flow alone, and not
   constant in green alone when the other variables are held fixed).
8. Low-demand boundary: for every fixed positive GR_phase and Cycle_Time, the
   one-sided limit as flow_lane -> 0+ exists, is finite, and is non-negative.
   No condition requires this limit to equal zero.
9. Zero-green singularity: for any fixed flow_lane > 0 and Cycle_Time > 0,
   lim_{GR_phase -> 0+} y = +infinity (severe delay blow-up as green vanishes).
10. Use operations +, -, *, /, **, exp, log; exp/log arguments MUST be dimensionless.
"""

PHYSICAL_REQUIREMENTS_COMPACT = """
Requirements: use all three inputs and coefficients a1, a2, ... (at most 8).
At fixed GR_phase and Cycle_Time, y is nondecreasing in flow_lane; at fixed
flow_lane and Cycle_Time, y is nonincreasing in GR_phase. Throughout the
operational domain, y is finite, real, and non-negative, and genuinely depends
on both flow_lane and GR_phase. As flow_lane -> 0+, its limit must exist, be
finite and non-negative; No condition requires that limit to equal zero. For
fixed positive flow and cycle, lim_{GR_phase -> 0+} y = +infinity. Use only +, -,
*, /, **, exp, and log.
"""

PROMPT_STYLES = ("standard", "compact")

OUTPUT_FORMAT_RULES = """
OUTPUT FORMAT RULES:
- Return ONLY one ```...``` block or one plain-text block with exactly:
  ### Expression
  y = <single expression>
  ### Explanation
  <brief interpretation>
- Do not include reasoning, duplicate analysis, retry commentary, or multiple
  candidate expressions in the response.
"""

_EXPRESSION_BLOCK_PATTERNS = (
    r"###\s*Expression\s*(?:\n|:)*\s*y\s*=\s*(.+?)(?=\n\s*###|\Z)",
    r"(?:^|\n)\s*Expression\s*:?\s*y\s*=\s*(.+?)(?=\n\s*###|\n\s*Expression\s*:?|\Z)",
    r"(?:^|\n)\s*y\s*=\s*(.+?)(?=\n\s*###|\Z)",
)

_PROSE_LINE_PREFIXES = (
    "this ",
    "check ",
    "none ",
    "the ",
    "i will",
    "prohibited",
    "algebraically",
    "banned",
    "now,",
    "a truly",
    "return ",
)


def _looks_like_prose_line(line: str) -> bool:
    lower = line.strip().lower()
    if not lower:
        return False
    if lower.startswith("- ") or "**" in line:
        return True
    return any(lower.startswith(prefix) for prefix in _PROSE_LINE_PREFIXES)


def _sanitize_expression_text(raw: str) -> str:
    text = raw.replace("`", "").strip()
    text = text.replace("\\_", "_").replace("\\", "")
    for delimiter in (" --- ", "\n---", "\n###", "\n\nThis ", "\n\nCheck ", "\n\nNone "):
        if delimiter in text:
            text = text.split(delimiter, 1)[0]

    lines = []
    paren_depth = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if lines and paren_depth == 0:
                break
            continue
        if lines and paren_depth == 0 and _looks_like_prose_line(stripped):
            break
        lines.append(stripped)
        paren_depth += stripped.count("(") - stripped.count(")")

    if lines:
        return " ".join(lines)
    return " ".join(text.split())


def _find_expression_candidates(response: str) -> List[str]:
    candidates: List[str] = []
    headers = list(
        re.finditer(
            r"###\s*Expression\s*(?:\n|:)*\s*y\s*=\s*",
            response,
            re.IGNORECASE,
        )
    )
    for index, header in enumerate(headers):
        start = header.end()
        stop = headers[index + 1].start() if index + 1 < len(headers) else len(response)
        chunk = response[start:stop]
        explanation = re.search(r"\n\s*###\s*Explanation\b", chunk, re.IGNORECASE)
        if explanation:
            chunk = chunk[: explanation.start()]
        candidate = _sanitize_expression_text(chunk)
        if candidate:
            candidates.append(candidate)
    if candidates:
        return candidates

    for pattern in _EXPRESSION_BLOCK_PATTERNS[1:]:
        for match in re.finditer(pattern, response, re.IGNORECASE | re.DOTALL):
            candidate = _sanitize_expression_text(match.group(1))
            if candidate:
                candidates.append(candidate)
    return candidates


def _select_parseable_expression(candidates: List[str]) -> Optional[str]:
    for candidate in reversed(candidates):
        try:
            parse_symbolic_expression(candidate)
            return candidate
        except Exception:
            continue
    return candidates[-1] if candidates else None


def _validate_prompt_style(prompt_style: str) -> str:
    """Return a normalized, predeclared prompt style name."""
    normalized = str(prompt_style).strip().lower()
    if normalized not in PROMPT_STYLES:
        raise ValueError(
            f"Unknown prompt_style {prompt_style!r}; expected one of {PROMPT_STYLES}"
        )
    return normalized


def build_universal_lane_init_prompt(
    feature_explanations: dict,
    universal_features: list,
    intersection_id: int = 1,
    include_physical_knowledge: bool = True,
    prompt_style: str = "standard",
) -> str:
    """Build prompt for initial universal lane expression"""

    prompt_style = _validate_prompt_style(prompt_style)

    if prompt_style == "compact" and not include_physical_knowledge:
        return f"""
Construct one compact UNIVERSAL symbolic expression for lane-level delay at
intersection {intersection_id}. All lanes share the expression structure and
fit their coefficients separately.

Inputs are flow_lane (dimensionless lane flow ratio), GR_phase (dimensionless
effective green ratio), and Cycle_Time (seconds). Predict y in seconds with
coefficients a1, a2, ...; use only +, -, *, /, **, exp, and log; use at most
8 coefficients. Keep denominators and logarithms defined for ordinary positive
traffic inputs. Monotonicity, boundary, and traffic-law hypotheses are
intentionally withheld.

Return exactly:
```
### Expression
y = <expression>
### Explanation
<brief mathematical interpretation of each term>
```
"""

    if prompt_style == "compact":
        return f"""
Create one compact UNIVERSAL lane-delay expression for intersection
{intersection_id}. Every lane uses the same structure with separately fitted
coefficients. Inputs: flow_lane (dimensionless lane flow ratio), GR_phase
(dimensionless effective green ratio), and Cycle_Time (seconds). Output y in
seconds.

Requirements: use all three inputs and coefficients a1, a2, ... (at most 8).
{PHYSICAL_REQUIREMENTS_COMPACT}

{STRUCTURAL_RULES}
{DIMENSIONAL_RULES}
{INTERPRETABILITY_AND_PARSIMONY_RULES}
{OUTPUT_FORMAT_RULES}

Return exactly:
```
### Expression
y = <expression with coefficients a1, a2, etc.>
### Explanation
<brief physical interpretation of each term>
```
"""

    if not include_physical_knowledge:
        return f"""
You are fitting a compact UNIVERSAL symbolic regression expression for lane-level
delay at intersection {intersection_id}. The same expression structure is used
for every lane; coefficients are fitted separately for each lane.

Inputs:
- flow_lane: lane flow ratio
- GR_phase: effective green ratio
- Cycle_Time: signal cycle time

Predict y, measured in seconds. Use fitted coefficients a1, a2, ... and only
the operations +, -, *, /, **, exp, and log. Use no more than 8 coefficients.
Keep every denominator and logarithm numerically defined over ordinary positive
traffic inputs. Do not assume or state any monotonicity, boundary, or traffic-law
requirements; those hypotheses are intentionally withheld in this treatment.

Output EXACTLY in this format:
```
### Expression
y = <your expression with coefficients a1, a2, etc.>
### Explanation
<Brief mathematical interpretation of each term>
```
"""

    return f"""
You are a traffic delay modeling expert tasked with creating a UNIVERSAL mathematical expression for lane-level delay prediction at intersection {intersection_id}.

This expression will be used for ALL lanes (left-turn, through, right-turn) at the intersection, with lane-specific parameters fitted later.

Available features for ANY lane:
- flow_lane: Flow ratio for the lane (lane traffic volume / saturation flow, dimensionless)
- GR_phase: Green ratio for the phase(s) controlling this lane (green time / cycle time, dimensionless)
- Cycle_Time: Signal cycle time (seconds)

Your task: Create a symbolic expression for lane delay y (in seconds) that satisfies:

{PHYSICAL_REQUIREMENTS_STANDARD}

{STRUCTURAL_RULES}
{DIMENSIONAL_RULES}
{INTERPRETABILITY_AND_PARSIMONY_RULES}
{OUTPUT_FORMAT_RULES}


Output EXACTLY in this format:
```
### Expression
y = <your expression with coefficients a1, a2, etc.>
### Explanation
<Brief physical interpretation of each term>
```
"""


def build_universal_lane_mutation_prompt(
    base_expr: str,
    base_thought: str,
    base_explanation: str,
    validation_result: Tuple[bool, str],
    mutation_type: str,
    universal_features: list,
    intersection_id: int = 1,
    include_physical_knowledge: bool = True,
    prompt_style: str = "standard",
    search_feedback: Optional[str] = None,
) -> str:
    """Build prompt for mutating universal lane expression"""

    prompt_style = _validate_prompt_style(prompt_style)

    is_valid, validation_reason = validation_result
    validation_label = "Valid" if is_valid else "Invalid"
    search_feedback = str(search_feedback or "").strip()
    search_feedback_block = (
        f"\n{search_feedback}\n"
        if search_feedback
        else ""
    )

    mutation_instruction = {
        "small": "Make SMALL modifications to the expression structure (e.g., adjust exponents, add/remove small terms)",
        "large": "Make SIGNIFICANT changes to the expression structure (e.g., change fundamental form, try different approaches)"
    }.get(mutation_type, "Modify the expression")

    if prompt_style == "compact" and not include_physical_knowledge:
        return f"""
Evolve this UNIVERSAL lane-delay expression for intersection {intersection_id}:
y = {base_expr}

{search_feedback_block}
{mutation_instruction}. All lanes keep one shared structure and fit coefficients
separately. Predict y in seconds from flow_lane, GR_phase, and Cycle_Time using
a1, a2, ... and only +, -, *, /, **, exp, and log. Use at most 8 coefficients;
keep denominators and logarithms defined for ordinary positive inputs.
Physical-rule feedback and monotonicity/boundary hypotheses are intentionally
withheld.

Return exactly:
```
### Expression
y = <new expression>
### Explanation
<brief mathematical interpretation>
```
"""

    if prompt_style == "compact":
        repair = (
            f"Fix these physical-audit issues:\n{validation_reason}"
            if not is_valid
            else "The parent passed the physical checks; seek a better structure."
        )
        return f"""
Evolve this UNIVERSAL lane-delay expression for intersection {intersection_id}:
y = {base_expr}

Physical audit: {validation_label}. {repair}
{search_feedback_block}
{mutation_instruction}. Inputs are flow_lane and GR_phase (dimensionless) and
Cycle_Time (seconds); every lane shares the structure and fits its coefficients
separately.

Requirements: use all three inputs and coefficients a1, a2, ... (at most 8).
{PHYSICAL_REQUIREMENTS_COMPACT}

{STRUCTURAL_RULES}
{DIMENSIONAL_RULES}
{INTERPRETABILITY_AND_PARSIMONY_RULES}
{OUTPUT_FORMAT_RULES}

Return exactly:
```
### Expression
y = <new expression>
### Explanation
<brief physical interpretation>
```
"""

    if not include_physical_knowledge:
        return f"""
You are evolving a compact UNIVERSAL symbolic regression expression for
lane-level delay at intersection {intersection_id}.

Parent expression:
y = {base_expr}

{search_feedback_block}
Task: {mutation_instruction}. Predict delay y in seconds from flow_lane,
GR_phase, and Cycle_Time. Use fitted coefficients a1, a2, ... and only +, -,
*, /, **, exp, and log. Use no more than 8 coefficients and keep denominators
and logarithms numerically defined over ordinary positive inputs. Physical-rule
feedback is intentionally withheld in this treatment.

Output format:
```
### Expression
y = <new expression>
### Explanation
<Brief mathematical interpretation>
```
"""

    validation_feedback_block = (
        f"Physical-audit feedback:\n{validation_reason}"
        if not is_valid
        else "Expression passed all physical consistency checks."
    )
    repair_instruction = (
        f"IMPORTANT: Fix the physical-audit issues:\n{validation_reason}"
        if not is_valid
        else "The parent is valid, but try to discover an even better structure."
    )

    return f"""
You are evolving a UNIVERSAL lane-level delay expression for intersection {intersection_id}.

Parent expression:
y = {base_expr}

Physical-audit status: {validation_label}
{validation_feedback_block}
{search_feedback_block}

Available features:
- flow_lane: Flow ratio (dimensionless)
- GR_phase: Green ratio (dimensionless)  
- Cycle_Time: Cycle time (seconds)

Task: {mutation_instruction}

{PHYSICAL_REQUIREMENTS_STANDARD}

{STRUCTURAL_RULES}
{DIMENSIONAL_RULES}
{INTERPRETABILITY_AND_PARSIMONY_RULES}

{repair_instruction}
{OUTPUT_FORMAT_RULES}

Output format:
```
### Expression
y = <new expression>
### Explanation
<Physical interpretation>
```
"""


def parse_llm_response(response: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse LLM response to extract expression, thought, and explanation."""
    response = response.replace("```python", "```").replace("```text", "```").replace("\\_", "_")
    response = re.sub(r"```(?:python|text)?", "", response, flags=re.IGNORECASE)

    candidates = _find_expression_candidates(response)
    expression = _select_parseable_expression(candidates)
    if not expression:
        return None, None, None

    expl_match = re.search(
        r"###\s*Explanation\s*(?:\n|:)+\s*(.+?)(?:###|$)",
        response,
        re.IGNORECASE | re.DOTALL,
    )
    explanation = expl_match.group(1).strip() if expl_match else "No explanation provided"
    if "---" in explanation:
        explanation = explanation.split("---", 1)[0].strip()

    return expression, None, explanation


def safe_generate_universal_lane_expression(
        feature_explanations: dict,
        universal_features: list,
        mutation_type: str = "initial",
        base_expr: Optional[str] = None,
        base_thought: Optional[str] = None,
        base_explanation: Optional[str] = None,
        validation_result: Optional[Tuple[bool, str]] = None,
        intersection_id: int = 1,
        max_retries: int = 5,
        include_physical_knowledge: bool = True,
        enforce_physical_prefilter: bool = True,
        prompt_style: str = "standard",
        excluded_expressions: Optional[List[str]] = None,
        search_feedback: Optional[str] = None,
        max_prompt_exclusions: Optional[int] = None,
) -> Tuple[str, str, str]:
    """
    Safely generate universal lane expression with retries

    Returns:
        expression, thought, explanation
    """

    previous_expression = None
    previous_rejection = None
    prompt_style = _validate_prompt_style(prompt_style)
    excluded_expressions = list(excluded_expressions or [])
    excluded_canonical = {
        canonical_expression(expression)
        for expression in excluded_expressions
    }
    if max_prompt_exclusions is None:
        prompt_exclusions = excluded_expressions
    else:
        limit = max(0, int(max_prompt_exclusions))
        prompt_exclusions = excluded_expressions[:limit]

    for attempt in range(max_retries):
        try:
            # Build prompt
            if mutation_type == "initial":
                prompt = build_universal_lane_init_prompt(
                    feature_explanations,
                    universal_features,
                    intersection_id,
                    include_physical_knowledge=include_physical_knowledge,
                    prompt_style=prompt_style,
                )
            else:
                if base_expr is None or validation_result is None:
                    raise ValueError("Base expression and validation result required for mutation")

                prompt = build_universal_lane_mutation_prompt(
                    base_expr, base_thought, base_explanation, validation_result,
                    mutation_type, universal_features, intersection_id,
                    include_physical_knowledge=include_physical_knowledge,
                    prompt_style=prompt_style,
                    search_feedback=search_feedback,
                )

            if prompt_exclusions:
                prohibited = "\n".join(
                    f"- {expression}" for expression in prompt_exclusions
                )
                prompt += f"""

NOVELTY REQUIREMENT:
Return one structurally different expression from every expression below.
Algebraically equivalent rewrites, reordered terms, renamed coefficients, and
extra parentheses still count as duplicates and are prohibited.
Do not analyze or discuss the banned list in the response; output only the
required output block with one final formula.
{prohibited}
"""
                if len(excluded_expressions) > len(prompt_exclusions):
                    prompt += (
                        "\nThe program also checks the candidate against the full "
                        "historical archive, which is not reproduced here.\n"
                    )

            if previous_rejection:
                prompt += f"""

RETRY CORRECTION:
Your previous candidate was rejected.
Previous expression: {previous_expression or '[not parseable]'}
Rejection reason: {previous_rejection}
Generate a structurally different expression that fixes this exact issue.
{("If the issue concerns time units, ensure every additive term reduces to Cycle_Time times a dimensionless function." if include_physical_knowledge else "Correct only the parser or numerical-legality problem stated above.")}
"""

            # Get LLM response
            response = run_llm(prompt)

            # Parse response
            expression, thought, explanation = parse_llm_response(response)

            if expression is None:
                print(f"  Attempt {attempt + 1}: Failed to parse LLM response")
                previous_expression = None
                previous_rejection = (
                    "the response could not be parsed into the required output "
                    "format with a single ### Expression block"
                )
                _write_expression_attempt_audit(
                    {
                        "intersection_id": int(intersection_id),
                        "mutation_type": str(mutation_type),
                        "attempt": attempt + 1,
                        "max_attempts": int(max_retries),
                        "status": "format_invalid",
                        "expression": None,
                        "reason": previous_rejection,
                        "physical_knowledge": bool(include_physical_knowledge),
                        "prompt_style": prompt_style,
                    }
                )
                continue

            try:
                parse_symbolic_expression(expression)
            except Exception as exc:
                print(f"  Attempt {attempt + 1}: Parsed expression is not sympify-able: {exc}")
                previous_expression = expression
                if "unsupported coefficient identifiers" in str(exc):
                    from expression_rules import MAX_COEFFICIENTS

                    previous_rejection = (
                        "coefficient budget violation: use only consecutive "
                        f"coefficient identifiers a1 through a{MAX_COEFFICIENTS}; "
                        f"a{MAX_COEFFICIENTS + 1} and all higher identifiers are "
                        "forbidden. Silently count them before returning the "
                        "corrected expression"
                    )
                else:
                    previous_rejection = (
                        "the extracted expression was not a valid symbolic formula; "
                        "return only one parseable expression in the required format"
                    )
                _write_expression_attempt_audit(
                    {
                        "intersection_id": int(intersection_id),
                        "mutation_type": str(mutation_type),
                        "attempt": attempt + 1,
                        "max_attempts": int(max_retries),
                        "status": "symbolic_parse_invalid",
                        "expression": expression,
                        "reason": f"{previous_rejection}: {exc}",
                        "physical_knowledge": bool(include_physical_knowledge),
                        "prompt_style": prompt_style,
                    }
                )
                continue

            if canonical_expression(expression) in excluded_canonical:
                print(f"  Attempt {attempt + 1}: Rejected canonical duplicate")
                previous_expression = expression
                previous_rejection = (
                    "the expression is algebraically equivalent to an existing "
                    "population member; change the functional structure"
                )
                _write_expression_attempt_audit(
                    {
                        "intersection_id": int(intersection_id),
                        "mutation_type": str(mutation_type),
                        "attempt": attempt + 1,
                        "max_attempts": int(max_retries),
                        "status": "canonical_duplicate",
                        "expression": expression,
                        "reason": previous_rejection,
                        "physical_knowledge": bool(include_physical_knowledge),
                        "prompt_style": prompt_style,
                    }
                )
                continue

            validator = (
                validate_candidate_expression
                if enforce_physical_prefilter
                else validate_candidate_legality
            )
            is_valid, validation_reason = validator(expression)
            if not is_valid:
                print(f"  Attempt {attempt + 1}: Rejected unsafe expression: {validation_reason}")
                previous_expression = expression
                previous_rejection = validation_reason
                _write_expression_attempt_audit(
                    {
                        "intersection_id": int(intersection_id),
                        "mutation_type": str(mutation_type),
                        "attempt": attempt + 1,
                        "max_attempts": int(max_retries),
                        "status": "validator_rejected",
                        "expression": expression,
                        "reason": validation_reason,
                        "physical_knowledge": bool(include_physical_knowledge),
                        "prompt_style": prompt_style,
                    }
                )
                continue

            _write_expression_attempt_audit(
                {
                    "intersection_id": int(intersection_id),
                    "mutation_type": str(mutation_type),
                    "attempt": attempt + 1,
                    "max_attempts": int(max_retries),
                    "status": "accepted",
                    "expression": expression,
                    "reason": validation_reason,
                    "physical_knowledge": bool(include_physical_knowledge),
                    "prompt_style": prompt_style,
                }
            )
            return expression, thought or "", explanation

        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            previous_expression = None
            previous_rejection = str(e)
            _write_expression_attempt_audit(
                {
                    "intersection_id": int(intersection_id),
                    "mutation_type": str(mutation_type),
                    "attempt": attempt + 1,
                    "max_attempts": int(max_retries),
                    "status": "generation_exception",
                    "expression": None,
                    "reason": f"{type(e).__name__}: {e}",
                    "physical_knowledge": bool(include_physical_knowledge),
                    "prompt_style": prompt_style,
                }
            )
            if attempt == max_retries - 1:
                raise

    raise RuntimeError("Failed to generate valid expression after all retries")
