"""
ROAgent — orchestrates the full 3-agent pipeline and renders the formatted RO.

Role: AI Pipeline Architect.
Runs ComplaintExtractor → CauseAnalyzer → CorrectionWriter in sequence,
then renders the complete Repair Order to console using the exact BMW RO format.

Usage:
    python -m agent.ro_agent                        # interactive: prompts for complaint, name, VIN
    python -m agent.ro_agent "complaint" "name" VIN # CLI args
"""

import sys
import json
import uuid
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from agent.complaint_extractor import ComplaintExtractor
from agent.cause_analyzer import CauseAnalyzer
from agent.correction_writer import CorrectionWriter
from agent.ro_validator import ROValidator, ValidationReport

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "knowledge" / "data"

_LINE  = "-" * 81
_DLINE = "=" * 81


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
    customer_name:      str
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
    confidence_scores:  dict


# ── Orchestrator ──────────────────────────────────────────────────────────────

class ROAgent:
    """
    Orchestrates ComplaintExtractor → CauseAnalyzer → CorrectionWriter,
    assembles an RODraft, and renders the full formatted RO to console.

    Primary entry point: run_from_complaint(complaint_text, customer_name, vin)
    — finds the 3 most similar existing ROs, generates one complete RO grounded
    in each, and returns all three for the advisor to choose from.
    """

    def __init__(self) -> None:
        self.complaint_extractor = ComplaintExtractor()
        self.cause_analyzer      = CauseAnalyzer()
        self.correction_writer   = CorrectionWriter()
        self.vin_records:  list[dict] = self._load_json("vin_records.json")
        self.sample_ros:   list[dict] = self._load_json("sample_ros.json")

    def _load_json(self, filename: str) -> list:
        path = DATA_DIR / filename
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Could not load %s: %s", filename, exc)
            return []

    def _get_vehicle(self, vin: str) -> dict:
        """Return vehicle record from vin_records.json or a minimal default."""
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

    # ── Similarity search ─────────────────────────────────────────────────

    def _find_similar_ros(self, complaint_text: str, n: int = 3) -> list[dict]:
        """
        Return the n sample ROs whose complaint has the most word overlap with
        complaint_text. Tokens under 4 characters are ignored (removes stop words).
        Ties are broken by keeping the earlier RO in the list.

        Args:
            complaint_text: Raw complaint text entered by the technician.
            n:              Number of similar ROs to return.

        Returns:
            List of up to n sample RO dicts, most similar first.
        """
        input_words = {w.lower().strip(".,!?") for w in complaint_text.split() if len(w) >= 4}

        scored: list[tuple[int, dict]] = []
        for ro in self.sample_ros:
            ro_complaint = ro.get("complaint", "")
            ro_words = {w.lower().strip(".,!?") for w in ro_complaint.split() if len(w) >= 4}
            overlap = len(input_words & ro_words)
            if overlap > 0:
                scored.append((overlap, ro))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Log top matches for transparency
        for score, ro in scored[:n]:
            logger.info(
                "Similar RO: %s  score=%d  complaint=%.60s",
                ro.get("ro_number"), score, ro.get("complaint", ""),
            )

        return [ro for _, ro in scored[:n]]

    # ── Primary entry point ───────────────────────────────────────────────

    def run_from_complaint(
        self,
        complaint_text: str,
        customer_name:  str,
        vin:            str,
        advisor_id:     str = "ADV00",
        technician_id:  str = "TECH00",
    ) -> list[RODraft]:
        """
        Generate 3 complete Repair Orders from a typed technician complaint.

        Finds the 3 most similar sample ROs in sample_ros.json using word-overlap
        scoring, then runs the full pipeline (ComplaintExtractor → CauseAnalyzer →
        CorrectionWriter → ROValidator) once per similar RO, using each similar RO's
        DTC codes and labor op code as grounding. Returns 3 RODraft objects.

        Args:
            complaint_text: Raw complaint text typed by the technician.
            customer_name:  Customer's full name for the RO header.
            vin:            Vehicle Identification Number.
            advisor_id:     Service advisor ID (optional).
            technician_id:  Technician ID badge number (optional).

        Returns:
            List of up to 3 RODraft objects, each validated and rendered to console.
        """
        print(f"\n{_DLINE}")
        print("  BMW RO AGENT  —  GENERATING 3 REPAIR ORDERS")
        print(_DLINE)
        print(f"  VIN:       {vin}")
        print(f"  Customer:  {customer_name or 'Not provided'}")
        print(f"  Complaint: {complaint_text[:75]}{'...' if len(complaint_text) > 75 else ''}")
        print(_DLINE)

        similar = self._find_similar_ros(complaint_text, n=3)

        if not similar:
            print("\n  [WARN] No similar ROs found in sample database.")
            print("  Generating one RO with no sample grounding.\n")
            similar = [{"dtc_codes": [], "labor_op_code": None, "labor_hours": 1.0, "ro_number": "N/A"}]

        validator = ROValidator()
        results: list[RODraft] = []

        for i, sample in enumerate(similar, 1):
            ref_number = sample.get("ro_number", "N/A")
            ref_complaint = sample.get("complaint", "")[:70]

            print(f"\n{'*' * 81}")
            print(f"  OPTION {i} OF {len(similar)}")
            print(f"  Reference RO : {ref_number}")
            print(f"  Ref Complaint: {ref_complaint}")
            print(f"{'*' * 81}")

            dtc_codes = [
                d for d in sample.get("dtc_codes", [])
                if d and str(d).upper() not in ("NONE", "")
            ]

            ro = self.generate(
                vin=vin,
                technician_notes=complaint_text,
                dtc_codes=dtc_codes,
                customer_name=customer_name,
                labor_hours=float(sample.get("labor_hours", 1.0)),
                labor_op_code=sample.get("labor_op_code"),
                technician_id=technician_id,
                advisor_id=advisor_id,
            )

            report = validator.validate(ro)
            validator.format_report(report)
            results.append(ro)

        print(f"\n{_DLINE}")
        print(f"  DONE  —  {len(results)} Repair Order(s) Generated")
        verdicts = [validator.validate(r).submission_verdict for r in results]
        for i, v in enumerate(verdicts, 1):
            print(f"  Option {i}: {v}")
        print(_DLINE)

        return results

    # ── Core pipeline ─────────────────────────────────────────────────────

    def generate(
        self,
        vin:                  str,
        technician_notes:     str,
        dtc_codes:            list[str],
        customer_name:        str       = "",
        customer_description: str       = "",
        labor_hours:          float     = 0.0,
        technician_id:        str       = "TECH00",
        advisor_id:           str       = "ADV00",
        labor_op_code:        str | None = None,
    ) -> RODraft:
        """
        Run the full 3-agent pipeline and return an RODraft.

        Args:
            vin:                  Vehicle Identification Number.
            technician_notes:     Raw complaint/notes from the technician.
            dtc_codes:            Fault codes from scan tool (may be empty).
            customer_name:        Customer name for the RO header.
            customer_description: Optional verbatim customer statement.
            labor_hours:          Total labor time in FRU/hrs.
            technician_id:        Tech ID badge number.
            advisor_id:           Service advisor ID.
            labor_op_code:        BMW labor op code for direct lookup.

        Returns:
            RODraft with all 3Cs, codes, parts, flags, and confidence scores.
        """
        vehicle    = self._get_vehicle(vin)
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
            customer_name=customer_name,
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

    # ── Utility methods ───────────────────────────────────────────────────

    def run_from_sample(self, ro_number: str) -> RODraft | None:
        """
        Load a sample RO by ro_number and run the full pipeline against it.
        Useful for development and regression testing.
        """
        sample = next(
            (s for s in self.sample_ros if s.get("ro_number") == ro_number), None
        )
        if not sample:
            print(f"\n[ERROR] RO number '{ro_number}' not found in sample_ros.json")
            return None

        dtc_codes = [
            d for d in sample.get("dtc_codes", [])
            if d and str(d).upper() not in ("NONE", "")
        ]

        return self.generate(
            vin=sample.get("vin", ""),
            technician_notes=sample.get("technician_notes", sample.get("complaint", "")),
            dtc_codes=dtc_codes,
            customer_name=sample.get("customer_name", ""),
            customer_description=sample.get("complaint", ""),
            labor_hours=float(sample.get("labor_hours", 0.0)),
            technician_id=sample.get("technician_id", "TECH00"),
            advisor_id=sample.get("advisor_id", "ADV00"),
            labor_op_code=sample.get("labor_op_code"),
        )

    def run_validate_only(self, ro_draft: RODraft) -> ValidationReport:
        """Re-validate an existing RODraft without re-running the pipeline."""
        validator = ROValidator()
        report = validator.validate(ro_draft)
        validator.format_report(report)
        return report

    # ── RO renderer ───────────────────────────────────────────────────────

    def _render_ro(self, ro: RODraft, vehicle: dict, advisor_id: str) -> None:
        """Print the complete formatted BMW Repair Order to console."""
        today     = date.today().strftime("%Y-%m-%d")
        owner     = ro.customer_name or vehicle.get("last_known_owner", "N/A")
        phone     = vehicle.get("customer_phone",   "N/A")
        email     = vehicle.get("customer_email",   "N/A")
        odometer  = vehicle.get("mileage_at_decode", 0)
        prod_date = vehicle.get("in_service_date",  "N/A")
        warranty  = vehicle.get("warranty_status",  "unknown").upper().replace("_", " ")

        # ── HEADER ────────────────────────────────────────────────────────
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

        # ── LINE A ────────────────────────────────────────────────────────
        print("\n  LINE A: WARRANTY REPAIR\n")
        print("  COMPLAINT:")
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

        # ── LABOR TABLE ───────────────────────────────────────────────────
        labor_rate = 125.00
        labor_amt  = round(ro.labor_hours * labor_rate, 2)
        print(f"  {'LABOR CODE':<14}  {'DESCRIPTION':<38}  {'FRU/HRS':>7}    AMOUNT")
        print(f"  {'-'*14}  {'-'*38}  {'-'*7}    {'-'*9}")
        print(
            f"  {ro.labor_op_code:<14}  {'Warranty Repair Labor':<38}  "
            f"{ro.labor_hours:>7.1f}    ${labor_amt:>8.2f}"
        )
        print()

        # ── PARTS TABLE ───────────────────────────────────────────────────
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

        # ── COST SUMMARY ──────────────────────────────────────────────────
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

        # ── FOOTER ────────────────────────────────────────────────────────
        print()
        print("  DISCLAIMER: All parts installed are Genuine BMW Parts carrying a 2-Year/")
        print("  Unlimited Mileage Warranty.")
        print(f"  CUSTOMER SIGNATURE: X___________________________  DATE: {today}")
        print()

        # ── FLAGS ─────────────────────────────────────────────────────────
        if ro.flags:
            print(f"  {'-'*40}")
            print("  [!] FLAGS — ADVISOR REVIEW REQUIRED BEFORE SUBMISSION:")
            for f in ro.flags:
                print(f"     •  {f}")
            print(f"  {'-'*40}")
        else:
            print("  [OK] No flags detected — RO is ready for submission.")

        # ── CONFIDENCE ────────────────────────────────────────────────────
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
    """Word-wrap text at width characters, breaking on spaces."""
    words, lines, current = text.split(), [], ""
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

    # Accept args from CLI or prompt interactively
    if len(sys.argv) == 4:
        complaint_text = sys.argv[1]
        customer_name  = sys.argv[2]
        vin            = sys.argv[3]
    else:
        print(f"\n{_DLINE}")
        print("  BMW RO AGENT")
        print(_DLINE)
        complaint_text = input("  Customer Complaint : ").strip()
        customer_name  = input("  Customer Name      : ").strip()
        vin            = input("  VIN                : ").strip()

    if not complaint_text or not vin:
        print("\n[ERROR] Complaint and VIN are required.")
        sys.exit(1)

    ros = agent.run_from_complaint(complaint_text, customer_name, vin)
    sys.exit(0 if ros else 1)
