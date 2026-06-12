"""
FastAPI backend for the BMW RO Agent.

Serves the front-end (ui/index.html) and exposes a small JSON API that wraps
the existing 3-agent pipeline (ComplaintExtractor → CauseAnalyzer →
CorrectionWriter → ROValidator). No agent logic is duplicated here — this
layer only orchestrates the pipeline and serializes the result to JSON.

Run:
    uvicorn api.main:app --reload --port 8000
Then open http://localhost:8000
"""

import sys
import logging
from pathlib import Path

# The pipeline prints Unicode (→, ✓) to stdout; force UTF-8 so the Windows
# console (cp1252) doesn't raise UnicodeEncodeError mid-request.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from agent.ro_agent import ROAgent, RODraft
from agent.ro_validator import ROValidator, ValidationReport

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("api")

UI_DIR = Path(__file__).parent.parent / "ui"

LABOR_RATE = 125.00
HST_RATE   = 0.13

app = FastAPI(title="BMW RO Agent", version="1.0")

# Build the heavy objects once — each ROAgent spins up 3 Groq clients and loads
# the knowledge base, so we reuse a single instance across requests.
_agent: ROAgent | None = None
_validator: ROValidator | None = None


def get_agent() -> ROAgent:
    global _agent, _validator
    if _agent is None:
        logger.info("Initialising ROAgent + ROValidator …")
        _agent = ROAgent()
        _validator = ROValidator()
    return _agent


# ── Request / response models ───────────────────────────────────────────────

class GenerateRequest(BaseModel):
    customer_name: str = Field(..., min_length=1, description="Customer full name")
    vin:           str = Field(..., min_length=1, description="Vehicle Identification Number")
    complaint:     str = Field(..., min_length=1, description="Raw technician / customer complaint")


# ── Serialization helpers ───────────────────────────────────────────────────

def _financials(ro: RODraft) -> dict:
    labor_amount = round(ro.labor_hours * LABOR_RATE, 2)
    parts_total  = round(
        sum(float(p.get("unit_price", 0.0)) * int(p.get("qty", 1)) for p in ro.parts), 2
    )
    subtotal  = round(labor_amount + parts_total, 2)
    hst       = round(subtotal * HST_RATE, 2)
    total_due = round(subtotal + hst, 2)
    return {
        "labor_rate":   LABOR_RATE,
        "labor_amount": labor_amount,
        "parts_total":  parts_total,
        "subtotal":     subtotal,
        "hst":          hst,
        "total_due":    total_due,
    }


def _serialize_validation(report: ValidationReport) -> dict:
    return {
        "verdict":     report.submission_verdict,
        "passed":      report.passed,
        "hard_count":  report.hard_count,
        "soft_count":  report.soft_count,
        "field_flags": [
            {
                "field":       f.field,
                "description": f.flag_description,
                "reason":      f.rejection_reason,
                "severity":    f.severity,
            }
            for f in report.field_flags
        ],
        "rule_violations": [
            {
                "rule_id":     v.rule_id,
                "title":       v.rule_title,
                "consequence": v.consequence,
                "severity":    v.severity,
            }
            for v in report.rule_violations
        ],
    }


def _serialize_option(index: int, reference_ro: str, ro: RODraft, report: ValidationReport) -> dict:
    return {
        "option":             index,
        "reference_ro":       reference_ro,
        "ro_number":          ro.ro_number,
        "complaint":          ro.complaint,
        "cause":              ro.cause,
        "correction":         ro.correction,
        "dci_codes":          ro.dci_codes_used,
        "rci_codes":          ro.rci_codes_used,
        "labor_op_code":      ro.labor_op_code,
        "labor_hours":        ro.labor_hours,
        "parts":              ro.parts,
        "warranty_compliant": ro.warranty_compliant,
        "flags":              ro.flags,
        "confidence":         ro.confidence_scores,
        "financials":         _financials(ro),
        "validation":         _serialize_validation(report),
    }


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
def index() -> FileResponse:
    """Serve the single-page front-end."""
    return FileResponse(UI_DIR / "index.html")


@app.get("/api/vehicles")
def vehicles() -> dict:
    """Return known VINs so the UI can offer a quick-pick datalist."""
    agent = get_agent()
    return {
        "vehicles": [
            {
                "vin":             r.get("vin"),
                "owner":           r.get("last_known_owner"),
                "year":            r.get("year"),
                "model":           r.get("model"),
                "warranty_status": r.get("warranty_status"),
            }
            for r in agent.vin_records
        ]
    }


@app.post("/api/generate")
def generate(req: GenerateRequest) -> dict:
    """
    Run the full pipeline for the given complaint and return up to 3 RO options,
    each with the 3Cs, codes, cost summary, flags, and validation verdict.
    """
    agent = get_agent()
    assert _validator is not None

    complaint = req.complaint.strip()
    vin       = req.vin.strip()
    name      = req.customer_name.strip()

    logger.info("Generate request — VIN=%s customer=%s", vin, name)

    try:
        similar = agent._find_similar_ros(complaint, n=3)
        if not similar:
            similar = [{"dtc_codes": [], "labor_op_code": None, "labor_hours": 1.0, "ro_number": "N/A"}]

        options: list[dict] = []
        for i, sample in enumerate(similar, 1):
            dtc_codes = [
                d for d in sample.get("dtc_codes", [])
                if d and str(d).upper() not in ("NONE", "")
            ]
            ro = agent.generate(
                vin=vin,
                technician_notes=complaint,
                dtc_codes=dtc_codes,
                customer_name=name,
                labor_hours=float(sample.get("labor_hours", 1.0)),
                labor_op_code=sample.get("labor_op_code"),
            )
            report = _validator.validate(ro)
            options.append(_serialize_option(i, sample.get("ro_number", "N/A"), ro, report))

    except Exception as exc:  # surface a clean error to the UI
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")

    vehicle = agent._get_vehicle(vin)
    return {
        "customer_name": name,
        "vin":           vin,
        "complaint":     complaint,
        "vehicle": {
            "year_model":      f"{vehicle.get('year', 'N/A')} BMW {vehicle.get('model', 'Vehicle')}",
            "engine":          vehicle.get("engine", "N/A"),
            "warranty_status": vehicle.get("warranty_status", "unknown"),
            "warranty_type":   vehicle.get("warranty_type", "N/A"),
            "mileage":         vehicle.get("mileage_at_decode", 0),
            "found":           vehicle.get("year", "N/A") != "N/A",
        },
        "options": options,
    }
