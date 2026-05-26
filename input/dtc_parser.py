"""
DTCParser — extracts and classifies fault codes from raw scan tool output.

Parses SAE format (P0171, B1234) and BMW hex format (2A82, 29CD) codes from
free-text scan tool output, matches each against common_causes.json, and
returns a structured ParseResult the CauseAnalyzer can consume directly.
"""

import re
import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "knowledge" / "data"

# SAE format: 1 letter + 4 digits (P0171)
# BMW hex format: 4 or 6 hex chars (2A82, 140010, 8040BD)
_DTC_RE = re.compile(r'\b([A-Z][0-9]{4}|[0-9A-F]{6}|[0-9A-F]{4})\b', re.IGNORECASE)
_NO_FAULT_PHRASES = [
    "no faults found", "no fault found", "no faults", "keine fehler",
    "no dtc", "no codes found", "no codes stored", "system ok",
]


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class DTCRecord:
    """A single parsed fault code with lookup result."""
    code:               str
    description:        str         # inline text from scan output, or ""
    system:             str         # from matched pattern, or "unknown"
    matched_pattern_id: str | None  # pattern_id from common_causes.json, or None
    severity:           str         # "matched" | "unmatched"


@dataclass
class ParseResult:
    """Full result of a single parse() call."""
    raw_input:       str
    dtc_records:     list[DTCRecord]
    codes_only:      list[str]   # code strings only — for easy passing to CauseAnalyzer
    matched_count:   int
    unmatched_count: int
    no_fault_found:  bool        # True if a "no faults" phrase was detected


# ── Parser ────────────────────────────────────────────────────────────────────

class DTCParser:
    """
    Extracts DTC codes from raw scan tool text and matches them against
    the known cause patterns in common_causes.json.

    Build a flat dtc_lookup on init so every code in every pattern's
    common_dtcs list can be resolved in O(1) during parse().
    """

    def __init__(self) -> None:
        """Load common_causes.json and build the flat DTC → pattern lookup."""
        self.common_causes: list[dict] = self._load_json("common_causes.json")
        self.dtc_lookup: dict[str, dict] = {}
        for pattern in self.common_causes:
            for code in pattern.get("common_dtcs", []):
                if code:
                    self.dtc_lookup[code.upper()] = pattern

    def _load_json(self, filename: str) -> list:
        path = DATA_DIR / filename
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.error("Failed to load %s: %s", filename, exc)
            return []

    # ─────────────────────────────────────────────────────────────────────────

    def parse(self, raw_scan_output: str) -> ParseResult:
        """
        Extract all DTC codes from raw scan tool text and classify each one.

        Scans the input line by line. Matches SAE codes (P0171) and BMW hex
        codes (2A82) using the regex ``r'\\b([A-Z][0-9]{4}|[0-9A-F]{4})\\b'``.
        For each matched code, pulls the trailing description text from the
        same line, looks it up in dtc_lookup, and builds a DTCRecord.
        Duplicate codes within the same scan output are deduplicated.

        Args:
            raw_scan_output: Free-text string pasted or piped from a scan tool.

        Returns:
            ParseResult with all extracted records plus aggregate counts.
        """
        if not raw_scan_output or not raw_scan_output.strip():
            return ParseResult(
                raw_input=raw_scan_output or "",
                dtc_records=[],
                codes_only=[],
                matched_count=0,
                unmatched_count=0,
                no_fault_found=True,
            )

        raw_lower = raw_scan_output.lower()
        no_fault_found = any(phrase in raw_lower for phrase in _NO_FAULT_PHRASES)

        dtc_records: list[DTCRecord] = []
        seen_codes: set[str] = set()

        for line in raw_scan_output.splitlines():
            for match in _DTC_RE.finditer(line):
                code = match.group(1).upper()
                if code in seen_codes:
                    continue
                seen_codes.add(code)

                # Extract inline description: text after the code on the same line
                rest = line[match.end():].strip()
                description = re.sub(r'^[-:–\s]+', '', rest).strip()

                pattern = self.dtc_lookup.get(code)
                dtc_records.append(DTCRecord(
                    code=code,
                    description=description,
                    system=pattern.get("system", "unknown") if pattern else "unknown",
                    matched_pattern_id=pattern.get("pattern_id") if pattern else None,
                    severity="matched" if pattern else "unmatched",
                ))

        matched_count   = sum(1 for r in dtc_records if r.severity == "matched")
        unmatched_count = len(dtc_records) - matched_count

        return ParseResult(
            raw_input=raw_scan_output,
            dtc_records=dtc_records,
            codes_only=[r.code for r in dtc_records],
            matched_count=matched_count,
            unmatched_count=unmatched_count,
            no_fault_found=no_fault_found,
        )

    def parse_from_file(self, file_path: str) -> ParseResult:
        """
        Read a text file containing raw scan tool output and parse it.

        Args:
            file_path: Path to a .txt file with scan tool output.

        Returns:
            ParseResult from calling parse() on the file contents.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return self.parse(content)
        except Exception as exc:
            logger.error("Failed to read scan file '%s': %s", file_path, exc)
            return self.parse("")

    def format_summary(self, parse_result: ParseResult) -> None:
        """
        Print a clean console summary of a ParseResult.

        Matched codes are listed with their system and pattern ID.
        Unmatched codes are listed with any inline description and flagged
        for manual review.

        Args:
            parse_result: Result object returned by parse() or parse_from_file().
        """
        sep = "-" * 41
        print(f"\n{sep}")
        print("DTC PARSE RESULT")
        print(sep)
        print(f"Total codes found:   {len(parse_result.dtc_records)}")
        print(f"Matched to patterns: {parse_result.matched_count}")
        print(f"Unmatched:           {parse_result.unmatched_count}")
        print(f"No fault found:      {parse_result.no_fault_found}")

        matched   = [r for r in parse_result.dtc_records if r.severity == "matched"]
        unmatched = [r for r in parse_result.dtc_records if r.severity == "unmatched"]

        if matched:
            print("\nMATCHED:")
            for r in matched:
                print(f"  {r.code} - {r.system} - pattern {r.matched_pattern_id}")

        if unmatched:
            print("\nUNMATCHED (flag for manual review):")
            for r in unmatched:
                suffix = f" - {r.description}" if r.description else ""
                print(f"  {r.code}{suffix}")

        print(sep)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = DTCParser()

    # Test 1 — two known codes from common_causes.json
    sample_1 = (
        "140010 Misfire detected cylinder 3\n"
        "Fault: 801222 - A/C system pressure out of range"
    )
    print("\n=== TEST 1: Two known DTC codes ===")
    parser.format_summary(parser.parse(sample_1))

    # Test 2 — one unknown code
    sample_2 = "FAULT: A9B3 - Undocumented module fault"
    print("\n=== TEST 2: Unknown DTC code ===")
    parser.format_summary(parser.parse(sample_2))

    # Test 3 — no faults
    sample_3 = "No faults found"
    print("\n=== TEST 3: No faults ===")
    parser.format_summary(parser.parse(sample_3))
