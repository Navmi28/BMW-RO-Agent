"""
ROAgent — orchestrates the full 3-agent pipeline and renders the formatted RO.

Role: AI Pipeline Architect.
Runs ComplaintExtractor → CauseAnalyzer → CorrectionWriter in sequence,
then renders the complete Repair Order to console using the exact BMW RO format.

Usage:
    python -m agent.ro_agent                    # runs 3 test cases from sample_ros.json
    python -m agent.ro_agent RO-2026-0001       # runs a specific sample RO by number
"""

import sys
import json
import uuid
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from agent.complaint_extractor import ComplaintExtractor
from agent.cause_analyzer import CauseAnalyzer
from agent.correction_writer import CorrectionWriter

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "knowledge" / "data"

_LINE  = "─" * 81
_DLINE = "═" * 81


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class RODraft:
    """
    Complete structured Repair Order draft produced by the 3-agent pipeline.
    Every field maps directly to a section of the BMW RO output format.
    """
    ro_number:          str
    vin:                str
    year_model:         str
    technician_id:      str
    complaint:          str
    cause:              str
    dci_codes_used:     list[str]
    correction:         str
    rci_codes_used:     list[str]
    labor_op_code:      str
    parts:              list[dict]
    labor_hours:        float
    warranty_compliant: bool
    flags:              list[str]
    confidence_scores:  dict       # {"complaint": float, "cause": float, "correction": float}


# ── Orchestrator ──────────────────────────────────────────────────────────────

class ROAgent:
    """
    Orchestrates ComplaintExtractor → CauseAnalyzer → CorrectionWriter,
    assembles an RODraft, and renders the full formatted RO to console.

    The pipeline runs sequentially — each agent's output feeds the next:
        Step 1: ComplaintExtractor  →  complaint_statement
        Step 2: CauseAnalyzer       →  cause_statement  (receives complaint + DTCs + VIN)
        Step 3: CorrectionWriter    →  correction_statement (receives cause + DTCs + VIN + notes)
    """

    def __init__(self) -> None:
        self.complaint_extractor = ComplaintExtractor()
        self.cause_analyzer      = CauseAnalyzer()
        self.correction_writer   = CorrectionWriter()
        self.vin_records:   list[dict] = self._load_json("vin_records.json")
        self.sample_ros:    list[dict] = self._load_json("sample_ros.json")

    def _load_json(self, filename: str) -> list:
        path = DATA_DIR / filename
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Could not load %s: %s", filename, exc)
            return []

    def _get_vehicle(self, vin: str) -> dict:
        """Return vehicle record from vin_records.json or a default shell."""
        vin_upper = vin.upper().strip()
        for r in self.vin_records:
            if r.get("vin", "").upper() == vin_upper:
                return r
        return {
            "vin": vin, "year": "N/A", "model": "BMW Vehicle",
            "engine": "N/A", "warranty_status": "unknown",
            "mileage_at_decode": 0, "last_known_owner": "Unknown",
            "customer_phone": "N/A", "customer_email": "N/A",
            "in_service_date": "N/A",
        }

    # ── Main generate method ───────────────────────────────────────────────

    def generate(
        self,
        vin:                 str,
        technician_notes:    str,
        dtc_codes:           list[str],
        customer_description: str       = "",
        labor_hours:         float      = 0.0,
        technician_id:       str        = "TECH00",
        advisor_id:          str        = "ADV00",
        labor_op_code:       str | None = None,
    ) -> RODraft:
        """
        Run the full 3-agent pipeline and return an RODraft.

        Args:
            vin:                  Vehicle Identification Number.
            technician_notes:     Raw tech notes from the initial write-up.
            dtc_codes:            Fault codes pulled from scan tool.
            customer_description: Optional verbatim customer statement.
            labor_hours:          Total labor time (diagnosis + repair) in FRU/hrs.
            technician_id:        Tech ID badge number for the RO header.
            advisor_id:           Service advisor ID for the RO header.

        Returns:
            RODraft dataclass with all 3Cs, codes, parts, flags, and confidence scores.
        """
        vehicle = self._get_vehicle(vin)
        year_model = f"{vehicle.get('year', 'N/A')} BMW {vehicle.get('model', 'Vehicle')}"

        # ── Step 1: Complaint ──────────────────────────────────────────────
        print(f"\n{_DLINE}")
        print(f"  STEP 1 — COMPLAINT EXTRACTION   (VIN: {vin})")
        print(_DLINE)
        complaint_result = self.complaint_extractor.extract(
            raw_technician_notes=technician_notes,
            customer_description=customer_description,
            vin=vin,
            year_model=year_model,
        )
        print(f"\n  → Complaint: {complaint_result['complaint_statement']}")
        if complaint_result["flags"]:
            print(f"  → Flags:     {complaint_result['flags']}")

        # ── Step 2: Cause ──────────────────────────────────────────────────
        print(f"\n{_DLINE}")
        dtc_display = ", ".join(dtc_codes) if dtc_codes else "None"
        print(f"  STEP 2 — CAUSE ANALYSIS          (DTCs: {dtc_display})")
        print(_DLINE)
        cause_result = self.cause_analyzer.analyze(
            complaint_statement=complaint_result["complaint_statement"],
            dtc_codes=dtc_codes,
            vin=vin,
        )
        print(f"\n  → Cause:     {cause_result['cause_statement']}")
        print(f"  → DCI codes: {cause_result['dci_codes_used']}")
        if cause_result["flags"]:
            print(f"  → Flags:     {cause_result['flags']}")

        # ── Step 3: Correction ─────────────────────────────────────────────
        print(f"\n{_DLINE}")
        print(f"  STEP 3 — CORRECTION WRITING")
        print(_DLINE)
        correction_result = self.correction_writer.write(
            cause_statement=cause_result["cause_statement"],
            dtc_codes=dtc_codes,
            vin=vin,
            technician_notes=technician_notes,
            matched_pattern_id=cause_result.get("matched_pattern_id"),
            labor_hours=labor_hours,
            labor_op_code=labor_op_code,
        )
        print(f"\n  → Correction: {correction_result['correction_statement']}")
        print(f"  → RCI codes:  {correction_result['rci_codes_used']}")
        if correction_result["flags"]:
            print(f"  → Flags:      {correction_result['flags']}")

        # ── Aggregate flags ────────────────────────────────────────────────
        all_flags = (
            [f"[COMPLAINT] {f}" for f in complaint_result["flags"]]
            + [f"[CAUSE] {f}"      for f in cause_result["flags"]]
            + [f"[CORRECTION] {f}" for f in correction_result["flags"]]
        )

        # ── Assemble RODraft ───────────────────────────────────────────────
        ro = RODraft(
            ro_number=f"RO{date.today().strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}",
            vin=vin,
            year_model=year_model,
            technician_id=technician_id,
            complaint=complaint_result["complaint_statement"],
            cause=cause_result["cause_statement"],
            dci_codes_used=cause_result["dci_codes_used"],
            correction=correction_result["correction_statement"],
            rci_codes_used=correction_result["rci_codes_used"],
            labor_op_code=correction_result["labor_op_code"] or "N/A",
            parts=correction_result["parts_referenced"],
            labor_hours=labor_hours,
            warranty_compliant=correction_result["warranty_compliant"],
            flags=all_flags,
            confidence_scores={
                "complaint":  complaint_result["confidence"],
                "cause":      cause_result["confidence"],
                "correction": 1.0 if correction_result["warranty_compliant"] else 0.45,
            },
        )

        self._render_ro(ro, vehicle, advisor_id)
        return ro

    # ── run_from_sample ────────────────────────────────────────────────────

    def run_from_sample(self, ro_number: str) -> RODraft | None:
        """
        Load a sample RO from sample_ros.json by ro_number and run the full pipeline.

        The sample's 'complaint' (or 'technician_notes') field is used as raw input.
        The pipeline generates new 3Cs — this tests whether the AI can reproduce
        BMW-compliant output given the same raw notes a real technician would write.

        Args:
            ro_number: The ro_number field from sample_ros.json.

        Returns:
            RODraft, or None if the ro_number is not found.
        """
        sample = next(
            (s for s in self.sample_ros if s.get("ro_number") == ro_number), None
        )
        if not sample:
            print(f"\n[ERROR] RO number '{ro_number}' not found in sample_ros.json")
            return None

        quality = sample.get("ro_quality", "unknown")
        print(f"\n{'═' * 60}")
        print(f"ORIGINAL RO FROM DMS (ro_quality: {quality})")
        print(f"{'═' * 60}")
        print(f"COMPLAINT:        {sample.get('complaint', 'N/A')}")
        print(f"CAUSE:            {sample.get('cause', 'N/A')}")
        print(f"CORRECTION:       {sample.get('correction', 'N/A')}")
        print(f"REJECTION REASON: {sample.get('rejection_reason', 'N/A')}")
        print()
        print(f"{'═' * 60}")
        print("AGENT-GENERATED RO")
        print(f"{'═' * 60}")

        # Build raw input from sample — use existing complaint as customer description
        technician_notes    = sample.get("technician_notes", sample.get("complaint", ""))
        customer_description = sample.get("complaint", "")
        dtc_codes = [
            d for d in sample.get("dtc_codes", [])
            if d and str(d).upper() not in ("NONE", "")
        ]

        ro = self.generate(
            vin=sample.get("vin", ""),
            technician_notes=technician_notes,
            dtc_codes=dtc_codes,
            customer_description=customer_description,
            labor_hours=float(sample.get("labor_hours", 0.0)),
            technician_id=sample.get("technician_id", sample.get("tech_id", "TECH00")),
            advisor_id=sample.get("advisor_id", "ADV00"),
            labor_op_code=sample.get("labor_op_code"),
        )

        print("FLAGS FIRED:")
        if ro.flags:
            for f in ro.flags:
                print(f"  ⚠ {f}")
        else:
            print("  None")

        return ro

    # ── RO renderer ───────────────────────────────────────────────────────

    def _render_ro(self, ro: RODraft, vehicle: dict, advisor_id: str) -> None:
        """
        Print the complete formatted Repair Order to console.
        Follows the exact BMW RO format standard (Parkview BMW style):
            Header Block → LINE A (Complaint / Cause / Correction / Labor / Parts)
            → Cost Summary Block → Footer → Flags → Confidence Scores
        """
        today      = date.today().strftime("%Y-%m-%d")
        owner      = vehicle.get("last_known_owner", "N/A")
        phone      = vehicle.get("customer_phone",   "N/A")
        email      = vehicle.get("customer_email",   "N/A")
        odometer   = vehicle.get("mileage_at_decode", 0)
        prod_date  = vehicle.get("in_service_date",  "N/A")
        warranty   = vehicle.get("warranty_status",  "unknown").upper().replace("_", " ")

        # ── HEADER BLOCK ───────────────────────────────────────────────────
        print(f"\n{_DLINE}")
        print("                        BMW REPAIR ORDER")
        print(_DLINE)
        print(f"  REG DATE:   {today:<12}  REPAIR ORDER#: {ro.ro_number:<18}  TAG: N/A")
        print(f"  READY DATE: {today:<12}  ADVISOR:       {advisor_id:<18}  TECH ID: {ro.technician_id}")
        print(f"  INV DATE:   {today:<12}  STATUS:        {'OPEN':<18}  TERMS: {warranty}")
        print(_LINE)
        print(f"  CUSTOMER: {owner:<38}  VIN:         {ro.vin}")
        print(f"  CELL:     {phone:<38}  YEAR/MODEL:  {ro.year_model}")
        print(f"  EMAIL:    {email:<38}  PROD DATE:   {prod_date}")
        print(f"  {'':38}  ODOMETER IN:  {odometer:,} km")
        print(f"  {'':38}  ODOMETER OUT: {odometer:,} km")
        print(_LINE)

        # ── LINE A ─────────────────────────────────────────────────────────
        print("\n  LINE A: WARRANTY REPAIR")
        print()
        print("  COMPLAINT:")
        # Word-wrap long complaint at 72 chars
        for part in _wrap(ro.complaint, 72):
            print(f"    {part}")
        print()
        print("  CAUSE:")
        for part in _wrap(f"CAUSE: {ro.cause}", 72):
            print(f"    {part}")
        print()
        print("  CORRECTION:")
        for part in _wrap(f"CORRECTION: {ro.correction}", 72):
            print(f"    {part}")
        print()

        # ── LABOR TABLE ────────────────────────────────────────────────────
        labor_rate  = 125.00          # $/hr (illustrative dealership rate)
        labor_amt   = round(ro.labor_hours * labor_rate, 2)
        print(f"  {'LABOR CODE':<14}  {'DESCRIPTION':<38}  {'FRU/HRS':>7}    AMOUNT")
        print(f"  {'-'*14}  {'-'*38}  {'-'*7}    {'-'*9}")
        print(
            f"  {ro.labor_op_code:<14}  {'Warranty Repair Labor':<38}  "
            f"{ro.labor_hours:>7.1f}    ${labor_amt:>8.2f}"
        )
        print()

        # ── PARTS TABLE ────────────────────────────────────────────────────
        parts_total = 0.0
        if ro.parts:
            print(f"  {'QTY':<5}  {'PART NUMBER':<16}  {'DESCRIPTION':<28}  {'UNIT PRICE':>10}    TOTAL")
            print(f"  {'-'*5}  {'-'*16}  {'-'*28}  {'-'*10}    {'-'*9}")
            for part in ro.parts:
                qty   = int(part.get("qty", 1))
                unit  = float(part.get("unit_price", 0.0))
                total = unit * qty
                parts_total += total
                desc  = str(part.get("description", "OEM Part"))[:27]
                print(
                    f"  {qty:<5}  {part['part_number']:<16}  {desc:<28}  "
                    f"${unit:>9.2f}    ${total:>8.2f}"
                )
            print()

        # ── COST SUMMARY ───────────────────────────────────────────────────
        shop_fees = 0.00
        subtotal  = round(labor_amt + parts_total + shop_fees, 2)
        hst       = round(subtotal * 0.13, 2)
        total_due = round(subtotal + hst, 2)

        print(_LINE)
        print(f"  {'TOTAL LABOR:':<64} ${labor_amt:>9.2f}")
        print(f"  {'TOTAL PARTS:':<64} ${parts_total:>9.2f}")
        print(f"  {'SHOP SUPPLIES / HAZMAT FEES:':<64} ${shop_fees:>9.2f}")
        print(f"  {'SUBTOTAL:':<64} ${subtotal:>9.2f}")
        print(f"  {'HST (13%):':<64} ${hst:>9.2f}")
        print(f"  {'TOTAL AMOUNT DUE:':<64} ${total_due:>9.2f}")
        print(_LINE)

        # ── FOOTER ─────────────────────────────────────────────────────────
        print()
        print("  DISCLAIMER: All parts installed are Genuine BMW Parts carrying a 2-Year/")
        print("  Unlimited Mileage Warranty.")
        print(f"  CUSTOMER SIGNATURE: X___________________________  DATE: {today}")
        print()

        # ── FLAGS ──────────────────────────────────────────────────────────
        if ro.flags:
            print(f"  {'─'*40}")
            print("  ⚠  FLAGS — ADVISOR REVIEW REQUIRED BEFORE SUBMISSION:")
            for f in ro.flags:
                print(f"     •  {f}")
            print(f"  {'─'*40}")
        else:
            print("  ✓  No flags detected — RO is ready for submission.")

        # ── CONFIDENCE ─────────────────────────────────────────────────────
        c = ro.confidence_scores
        print()
        print(
            f"  CONFIDENCE SCORES:  "
            f"Complaint {c['complaint']:.0%}  |  "
            f"Cause {c['cause']:.0%}  |  "
            f"Correction {c['correction']:.0%}"
        )
        print(
            f"  WARRANTY COMPLIANT: "
            f"{'YES — Claim ready' if ro.warranty_compliant else 'NO — Review required before submission'}"
        )
        print(f"\n{_DLINE}\n")


# ── Utility ───────────────────────────────────────────────────────────────────

def _wrap(text: str, width: int) -> list[str]:
    """Simple word-wrap that breaks on spaces at width."""
    words  = text.split()
    lines  = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = (current + " " + word).strip()
    if current:
        lines.append(current)
    return lines if lines else [""]


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = ROAgent()

    # If a specific RO number is passed as CLI arg, run just that one
    if len(sys.argv) > 1:
        ro_number = sys.argv[1]
        result = agent.run_from_sample(ro_number)
        sys.exit(0 if result else 1)

    # Default: run one good, one bad, one partial test case
    def _pick_sample(quality: str) -> dict | None:
        return next(
            (s for s in agent.sample_ros if s.get("ro_quality") == quality), None
        )

    test_cases = [
        ("GOOD",    _pick_sample("good")),
        ("BAD",     _pick_sample("bad")),
        ("PARTIAL", _pick_sample("partial")),
    ]

    for label, sample in test_cases:
        if not sample:
            print(f"\n[WARN] No '{label.lower()}' quality sample found in sample_ros.json")
            continue

        print(f"\n{'*' * 81}")
        print(f"  TEST CASE: {label} RO  —  {sample['ro_number']}")
        print(f"{'*' * 81}")

        ro = agent.run_from_sample(sample["ro_number"])

        if ro is None:
            continue

        # Summarise flags for this test case
        print(f"\n  TEST RESULT SUMMARY — {label} RO ({sample['ro_number']}):")
        if ro.flags:
            print(f"  {len(ro.flags)} flag(s) fired:")
            for f in ro.flags:
                print(f"    ✗ {f}")
        else:
            print("  ✓ Clean — no flags.")
        print(f"  Warranty compliant: {'YES' if ro.warranty_compliant else 'NO'}")
