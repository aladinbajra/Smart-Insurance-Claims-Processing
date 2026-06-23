"""
Tracker posting — REQ-051: post the run's work item to ADO / Jira.

This is a SIDE EFFECT that lives OUTSIDE the deterministic pipeline (like the
LLM advisory agent). It reads the deterministic ``tracker_ticket.json`` written
by Agent H and POSTs it to a real tracker when one is configured via env vars.

Idempotency: it first searches the tracker for an existing item carrying the
stable ``ticket_id`` label/tag; if found, it updates instead of creating, so
re-running never produces duplicates.

Graceful degradation: with no tracker configured (or any network/auth error) it
performs a local "dry-run" and writes the result — it never raises and never
touches the determinism-checked artifacts.

Env configuration:
    TRACKER_TYPE = jira | ado | (unset -> dry-run)
    Jira: JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN, JIRA_PROJECT_KEY
    ADO:  ADO_ORG_URL, ADO_PROJECT, ADO_PAT
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any

import httpx


def post_ticket_file(run_dir: str | Path) -> dict[str, Any]:
    """Read tracker_ticket.json from a run dir and post it. Never raises."""
    run_path = Path(run_dir)
    ticket_path = run_path / "tracker_ticket.json"
    if not ticket_path.is_file():
        result = {"posted": False, "reason": "no tracker_ticket.json in run dir"}
        _write_result(run_path, result)
        return result

    with ticket_path.open("r", encoding="utf-8") as file:
        ticket = json.load(file)

    result = post_ticket(ticket)
    _write_result(run_path, result)
    return result


def post_ticket(ticket: dict[str, Any]) -> dict[str, Any]:
    """Post one ticket to the configured tracker. Returns a result dict, never raises."""
    tracker = (os.getenv("TRACKER_TYPE") or "").strip().lower()
    try:
        if tracker == "jira":
            return _post_jira(ticket)
        if tracker == "ado":
            return _post_ado(ticket)
        return {
            "posted": False,
            "tracker": "none",
            "mode": "dry-run",
            "ticket_id": ticket.get("ticket_id"),
            "reason": "no TRACKER_TYPE configured; ticket prepared but not sent",
        }
    except Exception as exc:  # noqa: BLE001 — posting must never crash the app
        return {
            "posted": False,
            "tracker": tracker or "none",
            "ticket_id": ticket.get("ticket_id"),
            "error": f"{type(exc).__name__}: {exc}",
        }


# ----------------------------------------------------------------------
# Jira
# ----------------------------------------------------------------------

def _post_jira(ticket: dict[str, Any]) -> dict[str, Any]:
    base = (os.getenv("JIRA_BASE_URL") or "").rstrip("/")
    email = os.getenv("JIRA_EMAIL") or ""
    token = os.getenv("JIRA_API_TOKEN") or ""
    project = os.getenv("JIRA_PROJECT_KEY") or ""
    if not all((base, email, token, project)):
        return {"posted": False, "tracker": "jira", "reason": "jira env vars incomplete"}

    auth = (email, token)
    label = str(ticket.get("ticket_id", ""))

    with httpx.Client(timeout=15.0) as client:
        # Idempotency: look for an existing issue with this ticket_id label.
        search = client.get(
            f"{base}/rest/api/2/search",
            params={"jql": f'labels = "{label}"', "fields": "key"},
            auth=auth,
        )
        if search.status_code == 200:
            issues = search.json().get("issues", [])
            if issues:
                return {
                    "posted": True, "tracker": "jira", "action": "exists",
                    "ticket_id": label, "issue_key": issues[0].get("key"),
                }

        payload = {
            "fields": {
                "project": {"key": project},
                "summary": str(ticket.get("title", label)),
                "description": _description(ticket),
                "issuetype": {"name": "Task"},
                "labels": [label],
            }
        }
        created = client.post(f"{base}/rest/api/2/issue", json=payload, auth=auth)
        if created.status_code in (200, 201):
            return {
                "posted": True, "tracker": "jira", "action": "created",
                "ticket_id": label, "issue_key": created.json().get("key"),
            }
        return {
            "posted": False, "tracker": "jira", "ticket_id": label,
            "status_code": created.status_code, "error": created.text[:300],
        }


# ----------------------------------------------------------------------
# Azure DevOps (ADO)
# ----------------------------------------------------------------------

def _post_ado(ticket: dict[str, Any]) -> dict[str, Any]:
    org = (os.getenv("ADO_ORG_URL") or "").rstrip("/")
    project = os.getenv("ADO_PROJECT") or ""
    pat = os.getenv("ADO_PAT") or ""
    if not all((org, project, pat)):
        return {"posted": False, "tracker": "ado", "reason": "ado env vars incomplete"}

    token = base64.b64encode(f":{pat}".encode()).decode()
    headers = {
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json-patch+json",
    }
    label = str(ticket.get("ticket_id", ""))
    body = [
        {"op": "add", "path": "/fields/System.Title", "value": str(ticket.get("title", label))},
        {"op": "add", "path": "/fields/System.Description", "value": _description(ticket)},
        {"op": "add", "path": "/fields/System.Tags", "value": label},
    ]
    url = f"{org}/{project}/_apis/wit/workitems/$Task?api-version=7.0"
    with httpx.Client(timeout=15.0) as client:
        created = client.post(url, headers=headers, content=json.dumps(body))
        if created.status_code in (200, 201):
            return {
                "posted": True, "tracker": "ado", "action": "created",
                "ticket_id": label, "work_item_id": created.json().get("id"),
            }
        return {
            "posted": False, "tracker": "ado", "ticket_id": label,
            "status_code": created.status_code, "error": created.text[:300],
        }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _description(ticket: dict[str, Any]) -> str:
    lines = [
        f"Claim: {ticket.get('claim_id')}",
        f"Routing: {ticket.get('routing_decision')}",
        f"Net settlement: {ticket.get('net_settlement')}",
        f"Priority: {ticket.get('priority')}",
        f"Assigned to: {ticket.get('assigned_to')}",
        f"Summary: {ticket.get('summary')}",
    ]
    return "\n".join(lines)


def _write_result(run_path: Path, result: dict[str, Any]) -> None:
    try:
        with (run_path / "tracker_post_result.json").open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=2, sort_keys=True)
            file.write("\n")
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    import argparse

    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="Post a run's work item to ADO/Jira.")
    parser.add_argument("run_dir", help="Path to runs/<claim_id>/")
    args = parser.parse_args(argv)

    result = post_ticket_file(args.run_dir)
    print(f"Tracker post: {result.get('posted')} ({result.get('tracker', 'none')})")
    if result.get("reason"):
        print(f"  {result['reason']}")
    if result.get("error"):
        print(f"  error: {result['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
