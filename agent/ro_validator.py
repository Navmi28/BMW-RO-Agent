"""
ROValidator — final gate before RO submission to BMW.

Takes a completed RODraft, checks every field for rejection triggers using
WarrantyRules as the rule engine plus BMW SIB 01 01 20 structural requirements,
and returns a ValidationReport that tells the advisor exactly what to fix.
"""

from __future__ import annotations

import re
import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).parent.parent))

from knowledge.warranty_rules import WarrantyRules, RuleViolation

if TYPE_CHECKING:
    from agent.ro_agent import RODraft

logger = logging.getLogger(__name__)

_DCI_RE = re.compile(r'\bD[1-8]\b')
_RCI_RE = re.compile(r'\bR[0-8]\b')
_DTC_IN_TEXT_RE = re.compile(r'\b([A-Z][0-9]{4}|[0-9A-F]{4})\b', re.IGNORECASE)
_INSPECTION_WORDS = [
    "inspect", "visual", "measured", "verified", "confirmed",
    "observed", "traced", "checked", "found",
]
_DIAG_LANGUAGE = ["diagnosed", "found", "replaced", "repaired", "dtc", "fault code"]


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class FieldFlag:
    """A BMW-standard structural issue detected in a single RO field."""
    field:            str    # "complaint" | "cause" | "correction"
    flag_description: str
    rejection_reason: str
    severity:         str    # "hard" | "soft"


@dataclass
class ValidationReport:
    """Complete validation result for one RODraft."""
    ro_number:          str
    passed:             bool    # True only if zero hard violations and zero field flags
    field_flags:        list[FieldFlag]
    rule_violations:    list[RuleViolation]
    hard_count:         int
    soft_count:         int
    submission_verdict: str     # "READY FOR SUBMISSION" | "REQUIRES ADVISOR REVIEW" |
                                # "DO NOT SUBMIT — CRITICAL ISSUES"


# ── Validator ─────────────────────────────────────────────────────────────────

class ROValidator:
    """
    Final validation gate for completed RODraft objects.

    Combines two layers of checking:
        1. WarrantyRules.check_ro() — text-detectable warranty rule violations
        2. Structural field checks   — BMW SIB 01 01 20 format requirements
           (CUST STATES prefix, DCI/RCI codes, labor op code presence, etc.)

    All violations are aggregated into a single ValidationReport with a
    submission_verdict that tells the advisor exactly what action to take.
    """

    def __init__(self) -> None:
        """Instantiate the WarrantyRules rule engine."""
        self.warranty_rules = WarrantyRules()

    # ─────────────────────────────────────────────────────────────────────────

    def validate(self, ro_draft: RODraft) -> ValidationReport:
        """
        Run all field checks and warranty rule checks against the RODraft.

        Structural checks (FieldFlag):
            Complaint: CUST STATES prefix, minimum length, no diagnosis language.
            Cause:     DCI code present, DTC or inspection finding referenced.
            Correction: RCI code present, labor op code present, warranty_compliant flag.

        Rule checks (RuleViolation):
            Delegated to WarrantyRules.check_ro() for all text-detectable violations.

        Args:
            ro_draft: Completed RODraft produced by ROAgent.generate().

        Returns:
            ValidationReport with verdict, field flags, rule violations, and counts.
        """
        field_flags: list[FieldFlag] = []

        complaint  = ro_draft.complaint  or ""
        cause      = ro_draft.cause      or ""
        correction = ro_draft.correction or ""

        # ── COMPLAINT checks ──────────────────────────────────────────────────

        if not complaint.strip().upper().startswith("CUST STATES"):
            field_flags.append(FieldFlag(
                field="complaint",
                flag_description="Complaint does not begin with CUST STATES",
                rejection_reason=(
                    "Complaint must open with CUST STATES "
                    "per BMW documentation standard"
                ),
                severity="hard",
            ))

        if len(complaint.split()) < 10:
            field_flags.append(FieldFlag(
                field="complaint",
                flag_description=f"Complaint is only {len(complaint.split())} word(s) — minimum is 10",
                rejection_reason=(
                    "Complaint too brief — insufficient customer "
                    "description for claim support"
                ),
                severity="hard",
            ))

        for word in _DIAG_LANGUAGE:
            if word in complaint.lower():
                field_flags.append(FieldFlag(
                    field="complaint",
                    flag_description=f"Complaint contains diagnosis language: '{word}'",
                    rejection_reason=(
                        "Complaint field contains diagnosis language — "
                        "move technical findings to Cause field"
                    ),
                    severity="hard",
                ))
                break  # one flag per field is enough

        # ── CAUSE checks ──────────────────────────────────────────────────────

        if not _DCI_RE.search(cause):
            field_flags.append(FieldFlag(
                field="cause",
                flag_description="Cause does not contain a DCI code (D1–D8)",
                rejection_reason=(
                    "Cause must reference a BMW Diagnosis Category "
                    "Identifier per SIB 01 01 20"
                ),
                severity="hard",
            ))

        has_dtc_ref    = bool(_DTC_IN_TEXT_RE.search(cause))
        has_inspection = any(w in cause.lower() for w in _INSPECTION_WORDS)
        if not has_dtc_ref and not has_inspection:
            field_flags.append(FieldFlag(
                field="cause",
                flag_description=(
                    "Cause references neither a DTC code nor a "
                    "visual/mechanical inspection finding"
                ),
                rejection_reason=(
                    "Cause must reference either a DTC or an "
                    "inspection finding"
                ),
                severity="hard",
            ))

        # ── CORRECTION checks ─────────────────────────────────────────────────

        if not _RCI_RE.search(correction):
            field_flags.append(FieldFlag(
                field="correction",
                flag_description="Correction does not contain an RCI code (R0–R8)",
                rejection_reason=(
                    "Correction must reference a BMW Repair Category "
                    "Identifier per SIB 01 01 20"
                ),
                severity="hard",
            ))

        if not ro_draft.labor_op_code or ro_draft.labor_op_code in ("N/A", ""):
            field_flags.append(FieldFlag(
                field="correction",
                flag_description="Labor operation code is missing or N/A",
                rejection_reason=(
                    "Labor operation code missing — claim cannot be "
                    "processed without a valid BMW labor op code"
                ),
                severity="hard",
            ))

        if not ro_draft.warranty_compliant:
            field_flags.append(FieldFlag(
                field="correction",
                flag_description=(
                    "CorrectionWriter flagged one or more warranty rule violations"
                ),
                rejection_reason=(
                    "One or more warranty rules violated — "
                    "review flagged rules before submission"
                ),
                severity="soft",
            ))

        # ── Warranty rule check ───────────────────────────────────────────────

        ro_check = self.warranty_rules.check_ro(complaint, cause, correction)

        # ── Aggregate ─────────────────────────────────────────────────────────

        hard_flags = sum(1 for f in field_flags if f.severity == "hard")
        soft_flags = sum(1 for f in field_flags if f.severity == "soft")
        hard_count = hard_flags + ro_check.hard_violation_count
        soft_count = soft_flags + ro_check.soft_violation_count

        passed = hard_count == 0 and len(field_flags) == 0

        if hard_count > 0:
            verdict = "DO NOT SUBMIT — CRITICAL ISSUES"
        elif soft_count > 0 or len(field_flags) > 0:
            verdict = "REQUIRES ADVISOR REVIEW"
        else:
            verdict = "READY FOR SUBMISSION"

        return ValidationReport(
            ro_number=ro_draft.ro_number,
            passed=passed,
            field_flags=field_flags,
            rule_violations=ro_check.violations,
            hard_count=hard_count,
            soft_count=soft_count,
            submission_verdict=verdict,
        )

    def format_report(self, validation_report: ValidationReport) -> None:
        """
        Print the full validation report to console.

        If the RO passed all checks, prints a single green-light line.
        Otherwise prints a structured report grouped by field with hard
        violations marked ✗ and soft violations marked ⚠.

        Args:
            validation_report: Result object returned by validate().
        """
        dline = "=" * 54

        print(f"\n{dline}")
        print(f"RO VALIDATION REPORT — {validation_report.ro_number}")
        print(dline)

        if validation_report.passed:
            print("  ✓  RO PASSED ALL CHECKS — READY FOR SUBMISSION")
            print(dline)
            return

        print(f"VERDICT: {validation_report.submission_verdict}")
        print(f"Hard issues: {validation_report.hard_count}   "
              f"Soft issues: {validation_report.soft_count}")

        # ── Field flags grouped by field ──────────────────────────────────
        fields_order = ["complaint", "cause", "correction"]
        by_field: dict[str, list[FieldFlag]] = {}
        for flag in validation_report.field_flags:
            by_field.setdefault(flag.field, []).append(flag)

        for field_name in fields_order:
            flags = by_field.get(field_name, [])
            if not flags:
                continue
            print(f"\n{field_name.upper()}:")
            for flag in flags:
                icon = "✗" if flag.severity == "hard" else "⚠"
                print(f"  {icon}  {flag.flag_description}")
                print(f"         Rejection reason: {flag.rejection_reason}")

        # ── Warranty rule violations ──────────────────────────────────────
        if validation_report.rule_violations:
            print("\nWARRANTY RULE VIOLATIONS:")
            for v in validation_report.rule_violations:
                icon = "✗" if v.severity == "hard" else "⚠"
                print(f"  {icon}  {v.rule_id} {v.rule_title} — {v.consequence}")

        print(dline)
