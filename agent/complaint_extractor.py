"""
ComplaintExtractor — generates the first C (Complaint) from raw technician input.

Role: BMW Service Documentation Specialist.
Transforms raw technician notes and customer description into a BMW-compliant
complaint statement that starts with "CUST STATES" and contains zero diagnostic
language. Grounds the LLM in real vehicle data by injecting the serialized
VIN record from vin_records.json into every prompt.
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
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "knowledge" / "data"


class ComplaintExtractor:
    """
    Converts raw technician notes and customer description into a BMW-compliant
    Complaint statement (first of the three Cs).

    Looks up the VIN in vin_records.json and injects the full serialized vehicle
    record into the prompt so the LLM is grounded in real dealership data rather
    than hallucinated BMW knowledge.
    """

    def __init__(self) -> None:
        self.client = Groq()
        self.prompts = PromptLibrary()
        self.vin_records: list[dict] = self._load_json("vin_records.json")

    def _load_json(self, filename: str) -> list:
        path = DATA_DIR / filename
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.error("Failed to load %s: %s", filename, exc)
            return []

    def extract(
        self,
        raw_technician_notes: str,
        customer_description: str = "",
        vin: str = "",
        year_model: str = "",
    ) -> dict:
        """
        Generate a BMW-compliant complaint statement from raw technician input.

        Looks up the VIN in vin_records.json and serializes the matching record
        into the prompt. If no match is found, sets vin_record to "NO MATCH FOUND"
        and appends a flag before calling the LLM.

        Args:
            raw_technician_notes: Tech's raw notes from the initial write-up.
            customer_description: Optional verbatim customer statement.
            vin:                  Vehicle Identification Number for data lookup.
            year_model:           Accepted for backward compatibility; not used.

        Returns:
            dict with complaint_statement (str), confidence (float), flags (list[str]).
        """
        logger.info("ComplaintExtractor.extract() — VIN: %s", vin or "not provided")

        flags: list[str] = []
        vin_record = "NO MATCH FOUND"

        if vin:
            vin_upper = vin.upper().strip()
            for record in self.vin_records:
                if record.get("vin", "").upper() == vin_upper:
                    vin_record = json.dumps(record, indent=2)
                    break

        if vin_record == "NO MATCH FOUND":
            flags.append("VIN not found in vin_records.json")

        prompt = self.prompts.get_prompt(
            c_type="complaint",
            version="v1.0",
            variables={
                "vin_record":           vin_record,
                "technician_notes":     raw_technician_notes or "No notes provided.",
                "customer_description": customer_description or "Not provided.",
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
            logger.error("Groq connection error in ComplaintExtractor: %s", exc)
            flags.append(f"API_ERROR: {exc}")
        except Exception as exc:
            logger.error("Groq API error in ComplaintExtractor: %s", exc)
            flags.append(f"API_ERROR: {exc}")

        return {
            "complaint_statement": response_text,
            "confidence": 1.0 if vin_record != "NO MATCH FOUND" else 0.5,
            "flags": flags,
        }
