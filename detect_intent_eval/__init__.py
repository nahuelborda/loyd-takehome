"""Evaluation pipeline for Loyd's `_detect_intent` intent classifier.

Architecture (see docs/design-document.md):

    emails.jsonl -> Predictor -> predictions -> Scorers -> Reporter -> report

The Predictor and Scorer interfaces are deliberately small and stage-agnostic so the
same harness extends to `_parse_fields` and `_generate_email`.
"""

__version__ = "0.1.0"
