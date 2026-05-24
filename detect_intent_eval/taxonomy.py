"""Failure-mode taxonomy for `_detect_intent`.

Goes beyond correct-vs-incorrect: every prediction is tagged with the structural failure
modes it exhibits. The taxonomy is grounded in the real 25 emails (see dataset-notes.md)
and is designed to be extended with a measured inter-annotator agreement check (the MAST
method, arXiv 2503.13657).
"""

from __future__ import annotations

from .dataset import Email
from .predictor import Prediction

FAILURE_MODES = {
    "F1": ("Label-space misfit", "Input fits no label in the current 6-label space."),
    "F2": ("Meeting-state confusion", "schedule vs reschedule decided without checking whether a meeting is booked."),
    "F3": ("Negation / constraint miss", "An explicit 'don't' instruction was ignored."),
    "F4": ("Decoy vocabulary", "Meeting-shaped words with no real intent triggered an intent label."),
    "F5": ("Indirect-path miss", "cc-carryover / accidental forward not recognised as `none`."),
    "F6": ("Soft-intent miscalibration", "Vague/ambiguous message answered with high confidence."),
    "F7": ("Thin-context overreach", "Very terse message answered with high confidence."),
    "F8": ("Output / format failure", "Model output was malformed / unparseable."),
    "F9": ("Context blindness", "Subject and/or thread state disambiguated the intent, but the model weighted only the body and got it wrong."),
}

# Curated negation slice — see dataset-notes.md. With 25 emails this is a hand list;
# at scale it becomes a detector. Kept explicit so the heuristic is auditable.
NEGATION_IDS = {"E18"}
_MEETING_LABELS = {"schedule", "reschedule"}

# F9 proxy: an email is "context-rich" if the harness passed the model material
# beyond the body — a non-empty thread_so_far OR a reply subject ("Re: ..."). On
# context-rich emails, a wrong prediction is evidence the model ignored what it was
# given. This is a coarse signal; at scale I'd replace it with a per-email
# `context_required` annotation set during labelling.
def _is_context_rich(email: Email) -> bool:
    if email.thread_so_far:
        return True
    subj = (email.subject or "").lstrip().lower()
    return subj.startswith("re:")


def is_acceptable(email: Email, pred_label: str) -> bool:
    """Whether a predicted label counts as correct, respecting the label_fit tier.

    `ambiguous` rows accept either the gold or the documented alternative; `clean` and
    `misfit` rows accept only the (least-wrong) gold.
    """
    if email.label_fit == "ambiguous":
        return pred_label in {email.gold_label, email.alt_label}
    return pred_label == email.gold_label


def tag_failures(email: Email, pred: Prediction) -> list[str]:
    """Failure-mode codes in play for this (email, prediction). Empty = clean pass.

    Some codes are input properties (F1) and some are model behaviours (F2, F8); both
    are surfaced so the report can show *where* the classifier struggles structurally,
    not just *that* it was wrong.
    """
    tags: list[str] = []
    acceptable = is_acceptable(email, pred.label)

    if not pred.parse_ok:
        tags.append("F8")
    if email.label_fit == "misfit":
        tags.append("F1")  # input property — the report shows confidence on these
    if (
        not acceptable
        and email.gold_label in _MEETING_LABELS
        and pred.label in _MEETING_LABELS
    ):
        tags.append("F2")
    if email.id in NEGATION_IDS and not acceptable:
        tags.append("F3")
    if email.gold_label == "none" and pred.label != "none":
        tags.append("F4")
    if email.loyd_addressed == "none" and pred.label != "none":
        tags.append("F5")
    if email.label_fit == "ambiguous" and pred.confidence > 0.9:
        tags.append("F6")  # overconfident on a soft case — failure even if label is OK
    if email.word_count <= 5 and pred.confidence > 0.9:
        tags.append("F7")
    if not acceptable and _is_context_rich(email):
        tags.append("F9")
    return tags
