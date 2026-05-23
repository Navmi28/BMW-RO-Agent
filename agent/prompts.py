"""
Versioned prompt library for the BMW RO 3-Agent Pipeline.
All LLM prompt strings live here — no prompt text in any other agent file.

Version history:
  v1.0 — Initial production prompts (SIB 01 01 20 compliant, DCI/RCI coded)
"""

from typing import Any


class PromptLibrary:
    """
    Versioned container for all 3C prompt templates.

    Usage:
        lib = PromptLibrary()
        prompt = lib.get_prompt("complaint", "v1.0", {"vin_record": "...", ...})
        # Returns {"version": "v1.0", "c_type": "complaint", "system": "...", "user": "..."}
    """

    VERSION_REGISTRY: dict[str, dict[str, str]] = {
        "complaint":   {"v1.0": "COMPLAINT_PROMPT_V1"},
        "cause":       {"v1.0": "CAUSE_PROMPT_V1"},
        "correction":  {"v1.0": "CORRECTION_PROMPT_V1"},
    }

    # ─────────────────────────────────────────────────────────────────────────
    # COMPLAINT PROMPT — v1.0
    # ─────────────────────────────────────────────────────────────────────────

    COMPLAINT_PROMPT_V1: dict[str, str] = {
        "version": "v1.0",
        "c_type": "complaint",

        "system": """\
You are a BMW service documentation specialist. Your job is to convert
raw technician notes into a clean, professional customer complaint
statement for a BMW Repair Order. The complaint captures what the
customer experienced in customer language. It begins with "CUST STATES"
and contains no diagnosis, no DTC references, and no cause language.
Base your output entirely on the technician notes and customer
description provided.""",

        "user": """\
VEHICLE CONTEXT (from vin_records.json):
{vin_record}

TECHNICIAN NOTES:
{technician_notes}

CUSTOMER DESCRIPTION (if provided):
{customer_description}

INSTRUCTION:
Write the complaint statement beginning with "CUST STATES".
Base the statement entirely on the technician notes and customer
description provided above. The complaint captures what the customer
experienced — written in customer language, one to two sentences,
with no diagnosis, no DTC references, and no cause language.\
"""
    }

    # ─────────────────────────────────────────────────────────────────────────
    # CAUSE PROMPT — v1.0
    # ─────────────────────────────────────────────────────────────────────────

    CAUSE_PROMPT_V1: dict[str, str] = {
        "version": "v1.0",
        "c_type": "cause",

        "system": """\
You are a BMW STEP-certified master diagnostic technician. Your job is
to write the Cause field of a BMW Repair Order using the vehicle context,
matched cause pattern, and DTC codes provided to you. You write cause
statements that begin with the correct BMW Diagnosis Category Identifier
(DCI) code, explicitly reference the DTC, and use the suggested_cause
language from the matched pattern as your primary reference. All
information in your output comes from the data provided to you.""",

        "user": """\
VEHICLE CONTEXT (from vin_records.json):
{vin_record}

MATCHED CAUSE PATTERN (from common_causes.json — matched by DTC or keyword):
{matched_cause_pattern}

TECHNICIAN INPUT:
Complaint: {complaint_statement}
DTC Codes: {dtc_codes}

INSTRUCTION:
Write the cause statement using the vehicle context and matched cause
pattern provided above as your only reference. The cause statement
must begin with the correct DCI code(s), explicitly reference the
DTC code(s), and reflect the suggested_cause language from the matched
pattern. When matched_cause_pattern is "NO MATCH FOUND", state that
no matching pattern was found and mark the field for manual review.\
"""
    }

    # ─────────────────────────────────────────────────────────────────────────
    # CORRECTION PROMPT — v1.0
    # ─────────────────────────────────────────────────────────────────────────

    CORRECTION_PROMPT_V1: dict[str, str] = {
        "version": "v1.0",
        "c_type": "correction",

        "system": """\
You are a BMW warranty language specialist. Your job is to write the
Correction field of a BMW Repair Order using the data provided to you.

OUTPUT FORMAT — follow this exactly. No extra text, no headers, no labels outside this structure.

PART 1 — CORRECTION:
[RCI code] [component name].
Labor op [exact op_code from labor_op_record].
[One short sentence per action taken — one sentence per line.]

PART 2 — FLAGGED RULES:
[rule_id]: [one-line description of the violation]
If no rules are violated, write: None.

RULES FOR PART 1:
- First word must be the correct RCI code (R0–R8). R2 = part replaced, R3 = campaign/SIB, R5 = control unit programmed, R8 = post-repair verification.
- Second line must contain the exact labor op code from the labor_op_record. If labor_op_record is "NO MATCH FOUND", write: Labor op: manual review required.
- Each action line must be one sentence only. Use the suggested_correction language from the matched cause pattern as your reference.
- Do not use vague language: no "fixed", "repaired the issue", "took care of", "resolved it".

RULES FOR PART 2:
- Check the correction against every rule in the warranty_rules list.
- Flag only rules that are actually violated. List each by rule_id on its own line.
- Just name the violated rule code — do not explain the violation. 
- Do not explain non-violations. Do not list rules that are satisfied.
""",

        "user": """\
VEHICLE CONTEXT (from vin_records.json):
{vin_record}

MATCHED CAUSE PATTERN (from common_causes.json):
{matched_cause_pattern}

LABOR OPERATION (from labor_ops.json — matched by system and repair type):
{labor_op_record}

WARRANTY RULES (from warranty_guidelines.json — full rule list):
{warranty_rules}

CAUSE STATEMENT:
{cause_statement}

TECHNICIAN NOTES:
{technician_notes}

INSTRUCTION:
Write the correction statement using the data blocks above as your
only reference. The correction must begin with the correct RCI code(s),
include the labor op code from the labor_op_record, and reflect the
suggested_correction language from the matched cause pattern.
Check the correction against every rule in the warranty_rules list
and flag each violated rule by its rule_id.
Dont write more than 200 words.
When labor_op_record is "NO MATCH FOUND", set labor_op_code to null
and mark the field for manual review.\
"""
    }

    # ─────────────────────────────────────────────────────────────────────────
    # Public interface
    # ─────────────────────────────────────────────────────────────────────────

    def get_prompt(
        self,
        c_type: str,
        version: str,
        variables: dict[str, Any],
    ) -> dict[str, str]:
        """
        Return a filled prompt dict with 'system' and 'user' keys.

        Args:
            c_type:    One of 'complaint', 'cause', 'correction'.
            version:   Template version string, e.g. 'v1.0'.
            variables: Dict mapping placeholder names to their values.
                       Only the user prompt is templated; the system prompt is fixed.

        Returns:
            dict with keys: version, c_type, system, user (user prompt is filled).

        Raises:
            ValueError: If c_type or version is unknown.
        """
        if c_type not in self.VERSION_REGISTRY:
            raise ValueError(
                f"Unknown c_type '{c_type}'. Valid options: {list(self.VERSION_REGISTRY)}"
            )
        if version not in self.VERSION_REGISTRY[c_type]:
            raise ValueError(
                f"Version '{version}' not found for '{c_type}'. "
                f"Available: {list(self.VERSION_REGISTRY[c_type])}"
            )

        attr_name = self.VERSION_REGISTRY[c_type][version]
        template: dict[str, str] = getattr(self, attr_name)

        user_prompt = template["user"]
        for key, value in variables.items():
            placeholder = "{" + key + "}"
            user_prompt = user_prompt.replace(placeholder, str(value) if value is not None else "N/A")

        return {
            "version":  template["version"],
            "c_type":   c_type,
            "system":   template["system"],
            "user":     user_prompt,
        }

    def list_versions(self, c_type: str) -> list[str]:
        """Return all registered versions for a given c_type."""
        if c_type not in self.VERSION_REGISTRY:
            raise ValueError(f"Unknown c_type '{c_type}'.")
        return list(self.VERSION_REGISTRY[c_type])
