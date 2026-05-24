"""Dataset loading and the label space for `_detect_intent`."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# The current 6-label space (Loyd case brief, Section 3).
LABELS = ["schedule", "reschedule", "cancel", "query_agenda", "block_calendar", "none"]

# Verbatim definitions from the brief — fed into the predictor prompt. Refining label
# descriptions measurably separates semantically close classes (arXiv 2412.15603).
LABEL_DEFINITIONS = {
    "schedule": "User or third party wants to set up a new meeting.",
    "reschedule": "An existing meeting needs to be moved.",
    "cancel": "An existing meeting should be cancelled.",
    "query_agenda": "User is asking what's on their own calendar.",
    "block_calendar": "User wants to block time on their own calendar.",
    "none": "No scheduling intent detected.",
}

LABEL_FIT_VALUES = {"clean", "ambiguous", "misfit"}


@dataclass(frozen=True)
class Email:
    """One labeled inbound message. Mirrors a record in data/emails.jsonl."""

    id: str
    channel: str
    from_: str | None
    to: list[str]
    cc: list[str]
    loyd_addressed: str          # to | cc | none (indirect path) | direct (iMessage)
    subject: str | None
    thread_so_far: str | None
    body: str
    gold_label: str              # least-wrong label within the 6-label space
    label_fit: str               # clean | ambiguous | misfit
    alt_label: str | None        # the defensible alternative, for ambiguous rows
    proposed_label: str | None   # what it would be under an expanded space (documentation)
    notes: str

    @property
    def word_count(self) -> int:
        return len(self.body.split())


def load_emails(path: str | Path) -> list[Email]:
    """Load and validate the JSONL evaluation set."""
    emails: list[Email] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        emails.append(
            Email(
                id=r["id"],
                channel=r["channel"],
                from_=r.get("from"),
                to=r.get("to", []),
                cc=r.get("cc", []),
                loyd_addressed=r["loyd_addressed"],
                subject=r.get("subject"),
                thread_so_far=r.get("thread_so_far"),
                body=r["body"],
                gold_label=r["gold_label"],
                label_fit=r["label_fit"],
                alt_label=r.get("alt_label"),
                proposed_label=r.get("proposed_label"),
                notes=r.get("notes", ""),
            )
        )
    for e in emails:
        assert e.gold_label in LABELS, f"{e.id}: invalid gold_label {e.gold_label!r}"
        assert e.label_fit in LABEL_FIT_VALUES, f"{e.id}: invalid label_fit {e.label_fit!r}"
    return emails
