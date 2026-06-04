"""
CorrectionWriter — generates the third C (Correction) using RCI codes.

Role: BMW Warranty Language Specialist.
Loads all four knowledge files and injects serialized records into the prompt
so the LLM produces an RCI-coded correction grounded in real dealership data.
"""

import re
import json
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_OP_CODE_RE = re.compile(r'\b\d{2} \d{2} \d{3}\b')

import groq
from groq import Groq
from dotenv import load_dotenv
from agent.prompts import PromptLibrary

load_dotenv()
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "knowledge" / "data"
VALID_RCI = {"R0", "R1", "R2", "R3", "R5", "R6", "R7", "R8"}


class CorrectionWriter:
    """
    Generates a BMW-compliant correction statement with RCI codes and labor op.

    Uses the matched cause pattern from CauseAnalyzer (looked up by pattern_id)
    to find the correct labor operation and inject all four knowledge records
    into the correction prompt as structured context.
    """

    def __init__(self) -> None:
        self.client = Groq()
        self.prompts = PromptLibrary()
        self.vin_records:   list[dict] = self._load_json("vin_records.json")
        self.common_causes: list[dict] = self._load_json("common_causes.json")
        self.labor_ops:     list[dict] = self._load_json("labor_ops.json")
        self.warranty_rules: list[dict] = self._load_json("warranty_guidelines.json")

    def _load_json(self, filename: str) -> list:
        path = DATA_DIR / filename
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.error("Failed to load %s: %s", filename, exc)
            return []

    def write(
        self,
        cause_statement: str,
        dtc_codes: list[str],
        vin: str,
        technician_notes: str,
        matched_pattern_id: str | None = None,
        labor_hours: float | None = None,
        labor_op_code: str | None = None,
    ) -> dict:
        """
        Generate a BMW-compliant correction statement using injected knowledge data.

        Looks up the VIN record, resolves the cause pattern by matched_pattern_id,
        finds the matching labor operation by looking up labor_op_code (from
        sample_ros.json when available) directly in labor_ops.json, and serializes
        all four data sources into the prompt before calling the LLM.

        Args:
            cause_statement:     Output from CauseAnalyzer.
            dtc_codes:           List of DTCs from scan tool.
            vin:                 Vehicle Identification Number.
            technician_notes:    Tech's description of the repair performed.
            matched_pattern_id:  Pattern ID from CauseAnalyzer for direct lookup.
            labor_hours:         Total labor time (diagnosis + repair) in FRU/hrs.
            labor_op_code:       Op code from sample_ros.json for direct lookup.

        Returns:
            dict with correction_statement, rci_codes_used, labor_op_code,
            parts_referenced, warranty_compliant, flags.
        """
        logger.info("CorrectionWriter.write() — VIN: %s, pattern_id: %s", vin, matched_pattern_id)

        flags: list[str] = []

        # ── VIN lookup ────────────────────────────────────────────────────────
        vin_record = "NO MATCH FOUND"
        vin_upper = vin.upper().strip()
        for record in self.vin_records:
            if record.get("vin", "").upper() == vin_upper:
                vin_record = json.dumps(record, indent=2)
                break
        if vin_record == "NO MATCH FOUND":
            flags.append("VIN not found in vin_records.json")

        # ── Cause pattern lookup by pattern_id ────────────────────────────────
        matched_cause = None
        if matched_pattern_id:
            for pattern in self.common_causes:
                if pattern.get("pattern_id") == matched_pattern_id:
                    matched_cause = pattern
                    break

        if matched_cause:
            matched_cause_pattern = json.dumps(matched_cause, indent=2)
        else:
            matched_cause_pattern = "NO MATCH FOUND"
            flags.append("No matching cause pattern found in common_causes.json")

        # ── Labor op lookup: sample op_code → regex fallback ─────────────────────
        resolved_op_code = None
        labor_op_record = "NO MATCH FOUND"

        # Priority 1: op code passed directly from sample_ros.json
        target = labor_op_code

        # Priority 2: extract from suggested_correction text if not provided
        if not target and matched_cause:
            suggested = matched_cause.get("suggested_correction", "")
            code_match = _OP_CODE_RE.search(suggested)
            if code_match:
                target = code_match.group()

        if target:
            for op in self.labor_ops:
                if op.get("op_code") == target:
                    resolved_op_code = op.get("op_code")
                    labor_op_record = json.dumps(op, indent=2)
                    break

        if labor_op_record == "NO MATCH FOUND":
            flags.append("No matching labor op found in labor_ops.json")

        # ── FRU threshold check ───────────────────────────────────────────────
        if (
            labor_hours is not None
            and labor_hours > 5
            and matched_cause_pattern != "NO MATCH FOUND"
        ):
            flags.append(
                "Diagnosis exceeds 5 FRU — "
                "DCI codes insufficient, high-level notes required per SIB 01 01 20"
            )

        # ── Serialize full warranty rules list ────────────────────────────────
        warranty_rules = json.dumps(self.warranty_rules, indent=2)

        # ── Build and call prompt ─────────────────────────────────────────────
        prompt = self.prompts.get_prompt(
            c_type="correction",
            version="v1.0",
            variables={
                "vin_record":            vin_record,
                "matched_cause_pattern": matched_cause_pattern,
                "labor_op_record":       labor_op_record,
                "warranty_rules":        warranty_rules,
                "cause_statement":       cause_statement,
                "technician_notes":      technician_notes or "No repair notes provided.",
            },
        )

        response_text = "API_ERROR"
        try:
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user",   "content": prompt["user"]},
                ],
                temperature=1,
                max_completion_tokens=1024,
                top_p=1,
                stream=False,
            )
            response_text = completion.choices[0].message.content
        except groq.APIConnectionError as exc:
            logger.error("Groq connection error in CorrectionWriter: %s", exc)
            flags.append(f"API_ERROR: {exc}")
        except Exception as exc:
            logger.error("Groq API error in CorrectionWriter: %s", exc)
            flags.append(f"API_ERROR: {exc}")

        # ── Parse PART 1 (correction text) and PART 2 (flagged rules) ───────────
        correction_text, part2_rule_ids = self._parse_correction_parts(response_text)

        # ── Extract RCI codes from correction text ────────────────────────────
        rci_codes_used = sorted(
            code for code in VALID_RCI
            if (code + ".") in correction_text or (code + " ") in correction_text
        )

        # Add PART 2 rule violations to flags
        for rule_id in part2_rule_ids:
            flags.append(f"Warranty rule flagged by LLM: {rule_id}")

        warranty_compliant = len(flags) == 0

        return {
            "correction_statement": correction_text,
            "rci_codes_used":       rci_codes_used,
            "labor_op_code":        resolved_op_code,
            "parts_referenced":     [],
            "warranty_compliant":   warranty_compliant,
            "flags":                flags,
        }

    def _parse_correction_parts(self, response_text: str) -> tuple[str, list[str]]:
        """
        Split LLM correction response into correction text (PART 1) and
        flagged rule IDs (PART 2).

        Returns:
            (correction_text, flagged_rule_ids)
        """
        if not response_text or response_text == "API_ERROR":
            return response_text, []

        # Find where PART 2 starts — handle em-dash, regular dash, or colon variants
        part2_start = -1
        for marker in ["PART 2", "Part 2", "FLAGGED RULES"]:
            idx = response_text.find(marker)
            if idx != -1:
                if part2_start == -1 or idx < part2_start:
                    part2_start = idx

        if part2_start == -1:
            # No PART 2 found — return full response as correction
            return response_text.strip(), []

        part1_raw = response_text[:part2_start].strip()
        part2_raw = response_text[part2_start:].strip()

        # Clean PART 1: remove the "PART 1 — CORRECTION:" header line if present
        lines = part1_raw.splitlines()
        correction_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            upper = stripped.upper()
            if upper.startswith("PART 1") or upper.startswith("PART1"):
                continue  # skip header line
            correction_lines.append(stripped)

        correction_text = "\n".join(correction_lines).strip()

        # Extract WR-xxx rule IDs from PART 2
        rule_ids = re.findall(r'WR-\d{3}', part2_raw)
        # Exclude if PART 2 says "None"
        if "none" in part2_raw.lower() and not rule_ids:
            rule_ids = []

        return correction_text, rule_ids
