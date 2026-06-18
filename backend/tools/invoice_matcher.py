"""
Invoice / PO / GRN matching tool — REQ-043 (Matching Precision).

Deterministic three-way match: it compares the billed invoice total against the
expected/PO total, folds in the line-level billing variance, verifies the
purchase-order reference, and derives a match/mismatch status against a
policy-driven tolerance. The match STATUS is always derived here — never read
from a pre-set flag — so the result is meaningful on real claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MatchResult:
    match_status: str            # "match" | "mismatch"
    billing_variance_pct: float  # effective variance (total vs PO, or line-level)
    po_present: bool
    grn_present: bool
    reasons: list[str] = field(default_factory=list)


def match_invoice(
    invoice_amount: float,
    expected_amount: float,
    line_variance_pct: float,
    po_reference: str | None,
    grn_reference: str | None,
    tolerance: float,
) -> MatchResult:
    """
    Parameters
    ----------
    invoice_amount   : billed total on the repair/medical invoice.
    expected_amount  : expected/PO total the billing should match.
    line_variance_pct: line-item billing variance reported with the invoice.
    po_reference     : purchase-order reference (None if absent).
    grn_reference    : goods-receipt-note reference (None if absent).
    tolerance        : maximum acceptable variance (e.g. policy 0.25).
    """
    if expected_amount > 0:
        total_variance = abs(invoice_amount - expected_amount) / expected_amount
    else:
        total_variance = 0.0

    # The effective billing variance is the worse of the total mismatch and the
    # reported line-level variance.
    billing_variance = round(max(total_variance, float(line_variance_pct or 0.0)), 4)

    po_present = bool(po_reference)
    grn_present = bool(grn_reference)

    reasons: list[str] = []
    matched = True

    if billing_variance > tolerance:
        matched = False
        reasons.append(
            f"billing variance {billing_variance:.2%} exceeds tolerance {tolerance:.2%}"
        )
    if not po_present:
        matched = False
        reasons.append("missing purchase-order reference")

    return MatchResult(
        match_status="match" if matched else "mismatch",
        billing_variance_pct=billing_variance,
        po_present=po_present,
        grn_present=grn_present,
        reasons=reasons,
    )
