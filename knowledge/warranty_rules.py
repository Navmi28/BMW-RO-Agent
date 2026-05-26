"""
WarrantyRules — structured rule engine for BMW warranty claim validation.

Loads warranty_guidelines.json and exposes methods for checking completed
RO fields against BMW claim rules. Used by ROValidator as its rule engine.
"""

import re
import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"

_QTY_RE = re.compile(
    r'\d+(\.\d+)?\s*(ml|l\b|liters?|litres?|quarts?|grams?|g\b|oz\b|kg\b)',
    re.IGNORECASE,
)


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class RuleViolation:
    """A single warranty rule violation detected in an RO field."""
    rule_id:     str
    rule_title:  str
    field:       str        # "complaint" | "cause" | "correction"
    consequence: str        # from the JSON "consequence_of_violation"
    severity:    str        # "hard" | "soft"


@dataclass
class ROCheckResult:
    """Aggregated result of running check_ro() across all three Cs."""
    violations:            list[RuleViolation]
    hard_violation_count:  int
    soft_violation_count:  int
    warranty_compliant:    bool   # True only if hard_violation_count == 0
    summary:               str    # one-line human-readable verdict


# ── Rule engine ───────────────────────────────────────────────────────────────

class WarrantyRules:
    """
    Loads warranty_guidelines.json and exposes structured rule-checking methods.

    On init, builds two secondary indexes:
        rules_by_id       — keyed by rule_id for O(1) lookup
        rules_by_category — keyed by category, grouping all rules of the same
                            category together
    """

    def __init__(self) -> None:
        """Load warranty_guidelines.json and build lookup indexes."""
        self.rules: list[dict] = self._load_json("warranty_guidelines.json")
        self.rules_by_id: dict[str, dict] = {r["rule_id"]: r for r in self.rules}
        self.rules_by_category: dict[str, list[dict]] = {}
        for rule in self.rules:
            cat = rule.get("category", "unknown")
            self.rules_by_category.setdefault(cat, []).append(rule)

    def _load_json(self, filename: str) -> list:
        path = DATA_DIR / filename
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.error("Failed to load %s: %s", filename, exc)
            return []

    def _severity_for(self, rule: dict) -> str:
        """Map a rule's category to hard/soft severity."""
        category = rule.get("category", "")
        if category in ("documentation", "parts", "time_limits", "eligibility"):
            return "hard"
        return "soft"

    # ─────────────────────────────────────────────────────────────────────────

    def get_all(self) -> list[dict]:
        """Return the full list of rule dicts from warranty_guidelines.json."""
        return self.rules

    def get_by_category(self, category: str) -> list[dict]:
        """
        Return all rules matching the given category string.

        Args:
            category: One of "documentation", "parts", "time_limits",
                      "eligibility", or "labor".

        Returns:
            List of rule dicts, or [] if the category is not found.
        """
        return self.rules_by_category.get(category, [])

    def check_field(self, field_name: str, field_value: str) -> list[RuleViolation]:
        """
        Check a single RO field value against all text-detectable warranty rules.

        Only rules where a violation can be inferred from the field text are
        evaluated — procedural rules (e.g. submission timing, customer signatures)
        cannot be determined from text content and are not flagged here.

        Rules checked:
            WR-001 (all fields)   — flag if field text is empty
            WR-003 (correction)   — flag if aftermarket parts referenced
            WR-011 (correction)   — flag if programming mentioned but no I-Level
            WR-013 (correction)   — flag if fluid mentioned without a quantity

        Args:
            field_name:  One of "complaint", "cause", "correction".
            field_value: The generated text for that field.

        Returns:
            List of RuleViolation dataclasses for each detected violation.
        """
        violations: list[RuleViolation] = []
        text_lower = (field_value or "").lower()

        # WR-001: Three Cs Documentation — field must not be empty
        wr001 = self.rules_by_id.get("WR-001")
        if wr001 and not (field_value or "").strip():
            violations.append(RuleViolation(
                rule_id="WR-001",
                rule_title=wr001["title"],
                field=field_name,
                consequence=wr001["consequence_of_violation"],
                severity="hard",
            ))

        if field_name == "correction":
            # WR-003: Mandatory OEM Parts — flag if aftermarket parts referenced
            wr003 = self.rules_by_id.get("WR-003")
            if wr003 and any(
                kw in text_lower for kw in ["aftermarket", "non-oem", "non-bmw", "generic part"]
            ):
                violations.append(RuleViolation(
                    rule_id="WR-003",
                    rule_title=wr003["title"],
                    field=field_name,
                    consequence=wr003["consequence_of_violation"],
                    severity="hard",
                ))

            # WR-011: Programming/Coding — flag if programming mentioned without I-Level
            wr011 = self.rules_by_id.get("WR-011")
            if wr011:
                prog_kw    = ["programmed", "programming", "flashed", "software update", "coded", "coding"]
                ilevel_kw  = ["i-level", "ilevel", "integration level", "ispi", "ista"]
                has_prog   = any(kw in text_lower for kw in prog_kw)
                has_ilevel = any(kw in text_lower for kw in ilevel_kw)
                if has_prog and not has_ilevel:
                    violations.append(RuleViolation(
                        rule_id="WR-011",
                        rule_title=wr011["title"],
                        field=field_name,
                        consequence=wr011["consequence_of_violation"],
                        severity="hard",
                    ))

            # WR-013: Fluid documentation — flag if fluid mentioned without a quantity
            wr013 = self.rules_by_id.get("WR-013")
            if wr013:
                fluid_kw  = ["coolant", "refrigerant", "brake fluid", "transmission fluid", "oil"]
                has_fluid = any(
                    re.search(r'\b' + re.escape(kw) + r'\b', text_lower)
                    for kw in fluid_kw
                )
                has_qty   = bool(_QTY_RE.search(field_value or ""))
                if has_fluid and not has_qty:
                    violations.append(RuleViolation(
                        rule_id="WR-013",
                        rule_title=wr013["title"],
                        field=field_name,
                        consequence=wr013["consequence_of_violation"],
                        severity="soft",
                    ))

        return violations

    def check_ro(
        self,
        complaint:  str,
        cause:      str,
        correction: str,
    ) -> ROCheckResult:
        """
        Run check_field for all three Cs and return an aggregated result.

        Args:
            complaint:  Generated complaint statement.
            cause:      Generated cause statement.
            correction: Generated correction statement.

        Returns:
            ROCheckResult with all violations and aggregate counts.
        """
        all_violations: list[RuleViolation] = (
            self.check_field("complaint",  complaint)
            + self.check_field("cause",      cause)
            + self.check_field("correction", correction)
        )

        hard_count = sum(1 for v in all_violations if v.severity == "hard")
        soft_count = sum(1 for v in all_violations if v.severity == "soft")

        if hard_count == 0 and soft_count == 0:
            summary = "No warranty rule violations detected."
        elif hard_count > 0:
            summary = (
                f"{hard_count} hard and {soft_count} soft warranty rule violation(s) "
                "— claim likely to be rejected."
            )
        else:
            summary = (
                f"{soft_count} soft warranty rule violation(s) "
                "— review recommended before submission."
            )

        return ROCheckResult(
            violations=all_violations,
            hard_violation_count=hard_count,
            soft_violation_count=soft_count,
            warranty_compliant=hard_count == 0,
            summary=summary,
        )

    def format_violations(self, ro_check_result: ROCheckResult) -> None:
        """
        Print a clean console summary of all violations grouped by field.

        Hard violations are marked with ✗ and soft violations with ⚠.

        Args:
            ro_check_result: Result object returned by check_ro().
        """
        if not ro_check_result.violations:
            print("  No warranty rule violations detected.")
            return

        fields_order = ["complaint", "cause", "correction"]
        by_field: dict[str, list[RuleViolation]] = {}
        for v in ro_check_result.violations:
            by_field.setdefault(v.field, []).append(v)

        for field_name in fields_order:
            violations = by_field.get(field_name, [])
            if not violations:
                continue
            print(f"\n  {field_name.upper()}:")
            for v in violations:
                icon = "✗" if v.severity == "hard" else "⚠"
                print(f"    {icon}  [{v.rule_id}] {v.rule_title}")
                print(f"          Consequence: {v.consequence}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rules = WarrantyRules()
    print(f"Loaded {len(rules.get_all())} warranty rules\n")
    for category, rule_list in sorted(rules.rules_by_category.items()):
        print(f"  {category:<16} — {len(rule_list)} rule(s): "
              f"{', '.join(r['rule_id'] for r in rule_list)}")
