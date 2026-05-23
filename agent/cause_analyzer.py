"""
CauseAnalyzer — generates the second C (Cause) using DCI codes and vehicle context.

Role: BMW STEP-certified Master Diagnostic Technician.
Loads VIN records and common cause patterns, then produces a DCI-coded cause
statement grounded in the matched pattern from common_causes.json.
"""

import json
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import groq
from groq import Groq
from dotenv import load_dotenv
from agent.prompts import PromptLibrary

load_dotenv()
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "knowledge" / "data"
VALID_DCI = {"D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"}


class CauseAnalyzer:
    """
    Generates a DCI-coded BMW cause statement from complaint + DTCs + VIN context.

    Looks up the VIN in vin_records.json and matches DTCs or symptom keywords
    against common_causes.json. Both matched records are serialized and injected
    into the prompt so the LLM is grounded in real dealership data.
    """

    def __init__(self) -> None:
        self.client = Groq()
        self.prompts = PromptLibrary()
        self.vin_records: list[dict]    = self._load_json("vin_records.json")
        self.common_causes: list[dict]  = self._load_json("common_causes.json")

    def _load_json(self, filename: str) -> list:
        path = DATA_DIR / filename
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.error("Failed to load %s: %s", filename, exc)
            return []

    def analyze(
        self,
        complaint_statement: str,
        dtc_codes: list[str],
        vin: str,
    ) -> dict:
        """
        Generate a BMW-compliant cause statement using DCI codes and vehicle context.

        Looks up the VIN record and matches DTCs (priority) or complaint keywords
        against common_causes.json. Serializes both records into the prompt so
        the LLM uses only provided data. Flags missing matches before the LLM call.

        Args:
            complaint_statement: Output from ComplaintExtractor.
            dtc_codes:           List of DTCs from scan tool.
            vin:                 Vehicle Identification Number.

        Returns:
            dict with cause_statement, dci_codes_used, matched_pattern_id,
            dtc_referenced, confidence, flags.
        """
        logger.info("CauseAnalyzer.analyze() — VIN: %s, DTCs: %s", vin, dtc_codes)

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

        # ── Cause pattern match ───────────────────────────────────────────────
        matched_cause_pattern = "NO MATCH FOUND"
        matched_pattern_id = None
        valid_dtcs = {d.upper() for d in dtc_codes if d and d.upper() not in ("NONE", "")}
        complaint_lower = complaint_statement.lower()

        # Priority 1: DTC match
        matched_pattern = None
        for pattern in self.common_causes:
            pattern_dtcs = {d.upper() for d in pattern.get("common_dtcs", [])}
            if valid_dtcs & pattern_dtcs:
                matched_pattern = pattern
                break

        # Priority 2: exact phrase match against complaint
        if not matched_pattern:
            for pattern in self.common_causes:
                keywords = [kw.lower() for kw in pattern.get("symptom_keywords", [])]
                if any(kw in complaint_lower for kw in keywords):
                    matched_pattern = pattern
                    break

        # Priority 3: word-level match — any significant word (≥5 chars) from any keyword
        if not matched_pattern:
            complaint_words = set(complaint_lower.split())
            for pattern in self.common_causes:
                keywords = [kw.lower() for kw in pattern.get("symptom_keywords", [])]
                keyword_words = {w for kw in keywords for w in kw.split() if len(w) >= 5}
                if keyword_words & complaint_words:
                    matched_pattern = pattern
                    break

        if matched_pattern:
            matched_cause_pattern = json.dumps(matched_pattern, indent=2)
            matched_pattern_id = matched_pattern.get("pattern_id")
        else:
            flags.append("No matching cause pattern found in common_causes.json")

        # ── Build and call prompt ─────────────────────────────────────────────
        dtc_str = json.dumps(dtc_codes)

        prompt = self.prompts.get_prompt(
            c_type="cause",
            version="v1.0",
            variables={
                "vin_record":            vin_record,
                "matched_cause_pattern": matched_cause_pattern,
                "complaint_statement":   complaint_statement,
                "dtc_codes":             dtc_str,
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
            logger.error("Groq connection error in CauseAnalyzer: %s", exc)
            flags.append(f"API_ERROR: {exc}")
        except Exception as exc:
            logger.error("Groq API error in CauseAnalyzer: %s", exc)
            flags.append(f"API_ERROR: {exc}")

        # ── Extract DCI codes from response ───────────────────────────────────
        dci_codes_used = sorted(
            code for code in VALID_DCI
            if (code + ".") in response_text or (code + " ") in response_text
        )

        vin_matched = vin_record != "NO MATCH FOUND"
        pattern_matched = matched_cause_pattern != "NO MATCH FOUND"
        confidence = 1.0 if (vin_matched and pattern_matched) else 0.5

        return {
            "cause_statement":    response_text,
            "dci_codes_used":     dci_codes_used,
            "matched_pattern_id": matched_pattern_id,
            "dtc_referenced":     dtc_codes,
            "confidence":         confidence,
            "flags":              flags,
        }
