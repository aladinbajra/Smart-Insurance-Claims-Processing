"""
PDF Extractor Tool — REQ-013–016
Extracts structured fields from born-digital PDFs with per-field confidence
scores and bounding box coordinates for full audit traceability.
Synthetic-data fallback from manifest.yaml when PDF is unavailable.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pypdf
import yaml

from backend.models.schemas import BoundingBox, ExtractedField, PolicyPack


# ---------------------------------------------------------------------------
# Confidence tiers (never hardcode in agents — read policy thresholds instead)
# ---------------------------------------------------------------------------
_CONF_EXACT   = 0.95   # pattern matched precisely in born-digital PDF
_CONF_FUZZY   = 0.78   # partial / whitespace-sensitive match
_CONF_SYNTH   = 0.95   # synthetic fallback from manifest.yaml
_CONF_MISSING = 0.20   # field not found in document


# ---------------------------------------------------------------------------
# Internal text chunk (text + position from pypdf visitor)
# ---------------------------------------------------------------------------

class _Chunk:
    __slots__ = ("text", "bbox")

    def __init__(self, text: str, bbox: BoundingBox) -> None:
        self.text = text
        self.bbox = bbox


# ---------------------------------------------------------------------------
# PDF reading — REQ-015: bounding boxes via pypdf visitor
# ---------------------------------------------------------------------------

def _read_chunks(pdf_path: Path) -> list[_Chunk]:
    """
    Extract all text chunks with real bounding boxes from a born-digital PDF.
    Uses pypdf's visitor_text to capture text-matrix positions (tm[4]=x, tm[5]=y).
    """
    chunks: list[_Chunk] = []
    reader = pypdf.PdfReader(str(pdf_path))

    for page_num, page in enumerate(reader.pages, start=1):
        page_chunks: list[_Chunk] = []

        def _visitor(text: str, _cm: Any, tm: Any, _fd: Any, font_size: Any,
                     _pn: int = page_num, _acc: list = page_chunks) -> None:
            if not text or not text.strip():
                return
            x  = float(tm[4]) if tm else 0.0
            y  = float(tm[5]) if tm else 0.0
            fs = float(font_size) if font_size else 10.0
            w  = len(text) * fs * 0.55   # approximate glyph width
            h  = fs * 1.2
            _acc.append(_Chunk(
                text=text,
                bbox=BoundingBox(
                    page=_pn,
                    x1=round(x, 2),
                    y1=round(y, 2),
                    x2=round(x + w, 2),
                    y2=round(y + h, 2),
                ),
            ))

        page.extract_text(visitor_text=_visitor)
        chunks.extend(page_chunks)

    return chunks


def _raw(chunks: list[_Chunk]) -> str:
    return " ".join(c.text for c in chunks)


def _find_bbox(chunks: list[_Chunk], label: str) -> BoundingBox | None:
    """
    Return the bounding box of the value chunk that follows a given label.
    Looks at up to 4 chunks after the match to handle multi-chunk labels.
    """
    label_l = label.lower()
    for i, chunk in enumerate(chunks):
        if label_l in chunk.text.lower():
            for j in range(i + 1, min(i + 5, len(chunks))):
                if chunks[j].text.strip():
                    return chunks[j].bbox
            return chunk.bbox
    return None


def _parse_money(text: str) -> float | None:
    """'$1,234.56' → 1234.56.  Returns None if not parseable."""
    m = re.search(r"\$?([\d,]+\.?\d*)", text.replace(",", ""))
    return float(m.group(1)) if m else None


def _regex(pattern: str, text: str, group: int = 1) -> str | None:
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return m.group(group).strip() if m else None


# ---------------------------------------------------------------------------
# Field factory
# ---------------------------------------------------------------------------

def _make(
    name: str,
    value: Any,
    confidence: float,
    source_doc: str,
    bbox: BoundingBox | None,
    policy: PolicyPack,
    low_ocr: bool = False,
) -> ExtractedField:
    if low_ocr:
        confidence = min(confidence, 0.52)   # cap for scenario_08
    flag_thr = policy.extraction.low_confidence_flag_threshold
    return ExtractedField(
        field_name=name,
        value=value,
        confidence=round(confidence, 4),
        bbox=bbox,
        source_doc=source_doc,
        source_page=bbox.page if bbox else 1,
        requires_review=confidence < flag_thr,
    )


# ---------------------------------------------------------------------------
# Document-specific extractors
# (patterns match the exact label format used in generate_sicps_dataset.py)
# ---------------------------------------------------------------------------

def _extract_fnol(
    chunks: list[_Chunk],
    raw: str,
    doc: str,
    policy: PolicyPack,
    low_ocr: bool,
) -> list[ExtractedField]:
    out: list[ExtractedField] = []

    def add(field_name: str, pattern: str, conf: float = _CONF_EXACT) -> None:
        val = _regex(pattern, raw)
        bbox = _find_bbox(chunks, field_name.replace("_", " "))
        out.append(_make(field_name, val or "", conf if val else _CONF_MISSING,
                         doc, bbox, policy, low_ocr))

    add("claim_id",          r"Claim Number[:\s]+([A-Z0-9-]+)")
    add("report_date",       r"Report Date[:\s]+(\d{4}-\d{2}-\d{2})")
    add("claim_type",        r"Claim Type[:\s]+(\w+)")
    add("incident_date",     r"Incident Date[:\s]+(\d{4}-\d{2}-\d{2})")
    add("incident_time",     r"Incident Time[:\s]+([\d:]+)")
    add("incident_location", r"Incident Location[:\s]+([^\n\r:]+?)(?=Zip|$)")
    add("zip_code",          r"Zip Code[:\s]+(\d{5})")
    add("incident_type",     r"Incident Type[:\s]+([^\n\r:]+?)(?=Incident Sev|$)")
    add("incident_severity", r"Incident Severity[:\s]+([^\n\r:]+?)(?=Claimant|$)")
    add("claimant_id",       r"Claimant ID[:\s]+([A-Z0-9-]+)")
    add("claimant_name",     r"Claimant Name[:\s]+([^\n\r:]+?)(?=Policy|$)")
    add("policy_number",     r"Policy Number[:\s]+([A-Z0-9-]+)")
    add("prior_claims_18mo", r"Prior Claims \(18mo\)[:\s]+(\d+)")
    add("new_claimant",      r"New Claimant[:\s]+(Yes|No)", _CONF_EXACT)
    add("cat_event",         r"CAT Event[:\s]+(YES|NO)", _CONF_EXACT)
    add("fraud_history",     r"Fraud History[:\s]+(YES|NO)", _CONF_EXACT)
    add("police_report",     r"Police Report Filed[:\s]+(YES|NO|N/A)")
    add("witness_count",     r"Witnesses[:\s]+(\d+)")

    return out


def _extract_policy(
    chunks: list[_Chunk],
    raw: str,
    doc: str,
    policy: PolicyPack,
    low_ocr: bool,
) -> list[ExtractedField]:
    out: list[ExtractedField] = []

    def add(field_name: str, pattern: str, conf: float = _CONF_EXACT) -> None:
        val = _regex(pattern, raw)
        bbox = _find_bbox(chunks, field_name.replace("_", " "))
        out.append(_make(field_name, val or "", conf if val else _CONF_MISSING,
                         doc, bbox, policy, low_ocr))

    add("policy_number",    r"Policy Number[:\s]+([A-Z0-9-]+)")
    add("policy_type",      r"Policy Type[:\s]+([^\n\r:]+?)(?=Policy State|$)")
    add("policy_state",     r"Policy State[:\s]+([A-Z]{2})")
    add("effective_date",   r"Effective Date[:\s]+(\d{4}-\d{2}-\d{2})")
    add("expiration_date",  r"Expiration Date[:\s]+(\d{4}-\d{2}-\d{2})")
    add("policy_status",    r"Policy Status[:\s]+(\w+)")
    add("late_notice_days", r"Late Notice Window[:\s]+(\d+)")

    # Money fields need special parsing
    for field_name, label, pattern in [
        ("deductible",      "Deductible",     r"Deductible[:\s]+\$([\d,]+\.?\d*)"),
        ("coverage_limit",  "Coverage Limit", r"Coverage Limit[:\s]+\$([\d,]+\.?\d*)"),
        ("bodily_injury",   "Bodily Injury",  r"Bodily Injury Limit[:\s]+\$([\d,]+\.?\d*)"),
    ]:
        val_str = _regex(pattern, raw)
        val = float(val_str.replace(",", "")) if val_str else None
        bbox = _find_bbox(chunks, label)
        conf = _CONF_EXACT if val is not None else _CONF_MISSING
        out.append(_make(field_name, val, conf, doc, bbox, policy, low_ocr))

    # Exclusions — listed as "  - Pre-existing damage" etc.
    exclusions = re.findall(r"-\s+([^\n\r-][^\n\r]+)", raw)
    exclusions = [e.strip() for e in exclusions if len(e.strip()) > 3]
    bbox_excl = _find_bbox(chunks, "Exclusions")
    conf_e = _CONF_EXACT if exclusions else _CONF_MISSING
    out.append(_make("exclusions", exclusions, conf_e, doc, bbox_excl, policy, low_ocr))

    return out


def _extract_estimate(
    chunks: list[_Chunk],
    raw: str,
    doc: str,
    policy: PolicyPack,
    low_ocr: bool,
) -> list[ExtractedField]:
    out: list[ExtractedField] = []

    def add(field_name: str, pattern: str, conf: float = _CONF_EXACT) -> None:
        val = _regex(pattern, raw)
        bbox = _find_bbox(chunks, field_name.replace("_", " "))
        out.append(_make(field_name, val or "", conf if val else _CONF_MISSING,
                         doc, bbox, policy, low_ocr))

    add("claim_id",         r"Claim Number[:\s]+([A-Z0-9-]+)")
    add("invoice_number",   r"Invoice Number[:\s]+([A-Z0-9-]+)")
    add("provider_id",      r"Provider ID[:\s]+([A-Z0-9-]+)")
    add("provider_tax_id",  r"Provider Tax ID[:\s]+(\d{2}-\d{7})")
    add("provider_address", r"Provider Address[:\s]+([^\n\r:]+?)(?=Make|$)")
    add("vehicle_vin",      r"VIN[:\s]+([A-Z0-9]{17})")
    add("vehicle_year",     r"Year[:\s]+(\d{4})")

    for field_name, label, pattern in [
        ("market_value",     "Market Value",  r"Market Value[:\s]+\$([\d,]+\.?\d*)"),
        ("repair_estimate",  "TOTAL ESTIMATE",r"TOTAL ESTIMATE\s+\$([\d,]+\.?\d*)"),
        ("medical_total",    "Medical Total", r"Medical Total[:\s]+\$([\d,]+\.?\d*)"),
        ("injury_claim",     "Injury Claim",  r"Injury Claim[:\s]+\$([\d,]+\.?\d*)"),
    ]:
        val_str = _regex(pattern, raw)
        # scenario_08: total may be "[TOTAL UNREADABLE]"
        if val_str and re.match(r"[\d,]+\.?\d*$", val_str):
            val = float(val_str.replace(",", ""))
            conf = _CONF_EXACT
        else:
            val = None
            conf = _CONF_MISSING
        bbox = _find_bbox(chunks, label)
        out.append(_make(field_name, val, conf, doc, bbox, policy, low_ocr))

    # Line items — pattern: description followed by $amount on same logical line
    line_items = []
    for m in re.finditer(r"([A-Za-z][\w &/-]+?)\s+\$([\d,]+\.\d{2})", raw):
        desc = m.group(1).strip()
        if desc.upper() not in ("TOTAL ESTIMATE", "MEDICAL TOTAL", "INJURY CLAIM"):
            try:
                amt = float(m.group(2).replace(",", ""))
                line_items.append({"desc": desc, "amount": amt})
            except ValueError:
                pass
    bbox_li = _find_bbox(chunks, "Line Items")
    conf_li = _CONF_EXACT if line_items else _CONF_MISSING
    out.append(_make("line_items", line_items, conf_li, doc, bbox_li, policy, low_ocr))

    return out


def _extract_police_report(
    chunks: list[_Chunk],
    raw: str,
    doc: str,
    policy: PolicyPack,
    low_ocr: bool,
) -> list[ExtractedField]:
    out: list[ExtractedField] = []

    def add(field_name: str, pattern: str, conf: float = _CONF_EXACT) -> None:
        val = _regex(pattern, raw)
        bbox = _find_bbox(chunks, field_name.replace("_", " "))
        out.append(_make(field_name, val or "", conf if val else _CONF_MISSING,
                         doc, bbox, policy, low_ocr))

    add("police_report_number", r"Report Number[:\s]+([A-Z0-9-]+)")
    add("incident_date",        r"Incident Date[:\s]+(\d{4}-\d{2}-\d{2})")
    add("incident_time",        r"Incident Time[:\s]+([\d:]+)")
    add("incident_location",    r"Location[:\s]+([^\n\r:]+?)(?=Officer|$)")
    add("officer_id",           r"Officer ID[:\s]+([A-Z0-9-]+)")
    add("vehicle_count",        r"Vehicles Involved[:\s]+(\d+)")

    return out


def _extract_vendor_master(
    chunks: list[_Chunk],
    raw: str,
    doc: str,
    policy: PolicyPack,
    low_ocr: bool,
) -> list[ExtractedField]:
    out: list[ExtractedField] = []

    def add(field_name: str, pattern: str, conf: float = _CONF_EXACT) -> None:
        val = _regex(pattern, raw)
        bbox = _find_bbox(chunks, field_name.replace("_", " "))
        out.append(_make(field_name, val or "", conf if val else _CONF_MISSING,
                         doc, bbox, policy, low_ocr))

    add("vendor_id",      r"Vendor ID[:\s]+([A-Z0-9-]+)")
    add("vendor_name",    r"Vendor Name[:\s]+([^\n\r:]+?)(?=Tax|$)")
    add("vendor_tax_id",  r"Tax ID[:\s]+(\d{2}-\d{7})")
    add("vendor_address", r"Address[:\s]+([^\n\r:]+?)(?=License|$)")
    add("vendor_license", r"License[:\s]+([A-Z0-9-]+)")

    return out


# ---------------------------------------------------------------------------
# Synthetic fallback — REQ-013: when PDF missing or unreadable
# ---------------------------------------------------------------------------

def _synthetic_fallback(
    manifest: dict,
    doc_type: str,
    policy: PolicyPack,
) -> list[ExtractedField]:
    """
    Load field values directly from manifest.yaml with high confidence.
    Used when a PDF file is missing or completely unreadable.
    """
    out: list[ExtractedField] = []
    c = manifest.get("claimant", {})
    p = manifest.get("policy", {})
    i = manifest.get("incident", {})
    f = manifest.get("financials", {})
    v = manifest.get("vehicle") or {}
    pr = manifest.get("provider") or {}

    synthetic_fields: dict[str, Any] = {}

    if doc_type == "fnol":
        synthetic_fields = {
            "claim_id":          manifest.get("claim_id", ""),
            "claimant_name":     c.get("full_name", ""),
            "policy_number":     c.get("policy_number", ""),
            "incident_date":     i.get("date", ""),
            "report_date":       i.get("report_date", ""),
            "incident_location": i.get("location", ""),
            "zip_code":          i.get("zip_code", ""),
            "incident_type":     i.get("type", ""),
            "prior_claims_18mo": c.get("prior_claims_18mo", 0),
            "new_claimant":      "Yes" if c.get("new_claimant") else "No",
            "fraud_history":     "YES" if c.get("fraud_history") else "NO",
            "cat_event":         "YES" if i.get("cat_event") else "NO",
        }
    elif doc_type == "policy":
        synthetic_fields = {
            "policy_number":   p.get("policy_number", ""),
            "policy_type":     p.get("policy_type", ""),
            "effective_date":  p.get("effective_date", ""),
            "expiration_date": p.get("expiration_date", ""),
            "deductible":      f.get("deductible", 0.0),
            "coverage_limit":  p.get("coverage_limit", 0.0),
            "policy_state":    p.get("state", ""),
            "exclusions":      p.get("exclusions", []),
        }
    elif doc_type == "repair_estimate":
        synthetic_fields = {
            "repair_estimate":  f.get("repair_estimate", 0.0),
            "medical_total":    f.get("medical_total", 0.0),
            "market_value":     f.get("market_value", 0.0),
            "provider_id":      pr.get("provider_id", ""),
            "provider_tax_id":  pr.get("tax_id", ""),
            "vehicle_vin":      v.get("vin", ""),
        }

    for field_name, value in synthetic_fields.items():
        out.append(_make(
            name=field_name,
            value=value,
            confidence=_CONF_SYNTH,
            source_doc=f"manifest.yaml (synthetic fallback — {doc_type})",
            bbox=None,
            policy=policy,
            low_ocr=False,
        ))
    return out


# ---------------------------------------------------------------------------
# Route by document type
# ---------------------------------------------------------------------------

_DOC_EXTRACTORS = {
    "fnol":           _extract_fnol,
    "policy":         _extract_policy,
    "repair_estimate": _extract_estimate,
    "police_report":  _extract_police_report,
    "vendor_master":  _extract_vendor_master,
}


def _ocr_pdf_text(pdf_path: Path) -> str:
    """
    OCR fallback for scanned/image-only PDFs (REQ-050). Extracts embedded page
    images and runs tesseract on them. Returns "" if OCR is unavailable or the
    PDF has no images — born-digital PDFs never reach here.
    """
    from backend.tools.ocr import ocr_image_bytes

    try:
        reader = pypdf.PdfReader(str(pdf_path))
        texts: list[str] = []
        for page in reader.pages:
            for image in getattr(page, "images", []) or []:
                text, _conf = ocr_image_bytes(image.data)
                if text:
                    texts.append(text)
        return " ".join(texts)
    except Exception:
        return ""


def extract_document(
    pdf_path: Path,
    doc_type: str,
    policy: PolicyPack,
    low_ocr: bool = False,
) -> list[ExtractedField]:
    """
    Extract fields from a single PDF file.
    Returns a list of ExtractedField with confidence + bbox. REQ-013–015.
    Falls back to empty list on read error (caller should use synthetic fallback).
    """
    extractor = _DOC_EXTRACTORS.get(doc_type)
    if extractor is None:
        return []

    try:
        chunks = _read_chunks(pdf_path)
        raw = _raw(chunks)

        # REQ-050: if the page is born-digital this is non-empty. If it is a
        # scanned image (no embedded text), fall back to real OCR so the same
        # field/confidence/bbox logic can still run on the recognized text.
        if not raw.strip():
            ocr_raw = _ocr_pdf_text(pdf_path)
            if ocr_raw.strip():
                chunks = [
                    _Chunk(
                        text=ocr_raw,
                        bbox=BoundingBox(page=1, x1=0.0, y1=0.0, x2=612.0, y2=792.0),
                    )
                ]
                raw = ocr_raw
                # OCR'd text is inherently less certain than born-digital.
                low_ocr = True

        return extractor(chunks, raw, pdf_path.name, policy, low_ocr)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Bundle extractor — REQ-016: aggregate across ALL documents
# ---------------------------------------------------------------------------

def extract_bundle(
    claims_dir: Path,
    policy: PolicyPack,
) -> dict[str, ExtractedField]:
    """
    Extract and aggregate fields from all documents in a claims bundle.
    When the same field appears in multiple documents, keep the one with
    the highest confidence (cross-document consistency). REQ-016.

    Returns: dict[field_name → ExtractedField] (deduplicated, best confidence).
    """
    manifest_path = claims_dir / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.yaml not found in {claims_dir}")

    with open(manifest_path, encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)

    low_ocr = manifest.get("flags", {}).get("low_ocr_confidence", False)
    documents: list[dict] = manifest.get("documents", [])

    aggregated: dict[str, ExtractedField] = {}

    for doc_ref in documents:
        if not doc_ref.get("processable", True):
            continue

        doc_type = doc_ref["type"]
        pdf_path = claims_dir / doc_ref["file"]

        if pdf_path.exists():
            fields = extract_document(pdf_path, doc_type, policy, low_ocr)
        else:
            # REQ-013: synthetic-data fallback
            fields = _synthetic_fallback(manifest, doc_type, policy)

        # Merge: keep higher confidence when field already seen
        for ef in fields:
            existing = aggregated.get(ef.field_name)
            if existing is None or ef.confidence > existing.confidence:
                aggregated[ef.field_name] = ef

    return aggregated


# ---------------------------------------------------------------------------
# Policy loader helper (used by every agent)
# ---------------------------------------------------------------------------

def load_policy(policy_path: Path) -> PolicyPack:
    """Load and validate config/policy.yaml. REQ-034–037."""
    with open(policy_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return PolicyPack(**data)
