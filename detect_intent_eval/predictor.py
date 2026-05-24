"""Zero-shot LLM predictor for `_detect_intent`.

Mirrors production: a single LLM call returning a structured {label, confidence,
rationale}. Model-agnostic — works against OpenAI or any OpenAI-compatible endpoint.
Predictions are cached on disk so re-runs are free and deterministic.

Two prompt modes:
  - direct    (default)      — the model returns JSON only; fast, no reasoning.
  - reasoning (--reasoning)  — the model reasons step by step, then emits the JSON last.
    Used to A/B whether letting the model deliberate fixes the drop-to-`none` errors.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .dataset import LABELS, LABEL_DEFINITIONS, Email

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """\
You are the intent classifier (`_detect_intent`) for Loyd, an AI scheduling agent for \
the entertainment industry. Every inbound message is delegated to you first. Classify \
the message into exactly one intent label.

Return ONLY a JSON object with three keys:
  "label": one of the allowed labels (exact string),
  "confidence": a float in [0, 1] — your probability that the label is correct,
  "rationale": one short sentence.

Be honest with the confidence: lower it when the message is ambiguous, terse, or fits \
no label well."""

REASONING_SYSTEM_PROMPT = """\
You are the intent classifier (`_detect_intent`) for Loyd, an AI scheduling agent for \
the entertainment industry. Every inbound message is delegated to you first. Classify \
the message into exactly one intent label.

Think before you answer. Reason step by step: weigh the thread state, who the message \
is from and to, and whether a terse in-thread reply still carries the thread's intent.

After reasoning, output — on the final line — ONLY a JSON object with three keys:
  "label": one of the allowed labels (exact string),
  "confidence": a float in [0, 1] — your probability that the label is correct,
  "rationale": one short sentence.

Be honest with the confidence: lower it when the message is ambiguous, terse, or fits \
no label well."""


def _format_message(email: Email) -> str:
    """The message as the agent would see it — channel, headers, thread state, body."""
    parts = [f"Channel: {email.channel}"]
    if email.from_:
        parts.append(f"From: {email.from_}")
    if email.to:
        parts.append(f"To: {', '.join(email.to)}")
    if email.cc:
        parts.append(f"Cc: {', '.join(email.cc)}")
    if email.subject:
        parts.append(f"Subject: {email.subject}")
    if email.thread_so_far:
        parts.append(f"Thread so far: {email.thread_so_far}")
    parts.append(f"Body:\n{email.body}")
    return "\n".join(parts)


def _build_user_prompt(email: Email, rng: random.Random, reasoning: bool = False) -> str:
    # Randomise label order to neutralise position bias (arXiv 2406.07001).
    labels = LABELS[:]
    rng.shuffle(labels)
    defs = "\n".join(f"- {lab}: {LABEL_DEFINITIONS[lab]}" for lab in labels)
    closing = (
        "Reason step by step, then give the JSON object as the last line."
        if reasoning
        else "Classify this message. Respond with the JSON object only."
    )
    return (
        f"Allowed intent labels:\n{defs}\n\n"
        f"--- INBOUND MESSAGE ---\n{_format_message(email)}\n--- END MESSAGE ---\n\n"
        f"{closing}"
    )


@dataclass
class Prediction:
    email_id: str
    label: str
    confidence: float
    rationale: str
    parse_ok: bool        # False => malformed output (failure mode F8)
    raw_response: str

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_response(text: str) -> tuple[str, float, str, bool]:
    """Defensively parse the model's JSON. Returns (label, confidence, rationale, ok).

    Handles a plain JSON reply and a reasoning-then-JSON reply — in the latter the JSON
    is taken from the last `{...}` object in the text. `ok` is False whenever the output
    was malformed: that is a real, tracked failure mode (F8), never silently accepted.
    """
    obj = None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            obj = parsed
    except (json.JSONDecodeError, TypeError):
        pass
    if obj is None:
        # Last flat {...} object — robust to reasoning prose appearing before the JSON.
        for span in reversed(re.findall(r"\{[^{}]*\}", text or "", re.DOTALL)):
            try:
                cand = json.loads(span)
            except json.JSONDecodeError:
                continue
            if isinstance(cand, dict) and "label" in cand:
                obj = cand
                break
    if not isinstance(obj, dict):
        return ("none", 0.0, "unparseable response", False)

    label = str(obj.get("label", "")).strip()
    ok = label in LABELS
    try:
        conf = float(obj.get("confidence"))
    except (TypeError, ValueError):
        conf, ok = 0.0, False
    conf = min(1.0, max(0.0, conf))
    rationale = str(obj.get("rationale", ""))
    if label not in LABELS:
        label = "none"  # least-harmful fallback when the label is unusable
    return (label, conf, rationale, ok)


class ZeroShotPredictor:
    """Zero-shot intent predictor over an OpenAI-compatible chat API."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        cache_dir: str | Path = "runs/cache",
        temperature: float = 0.0,
        reasoning: bool = False,
        no_think: bool = False,
    ):
        from openai import OpenAI  # imported lazily so the rest of the package needs no SDK

        self.model = model
        self.temperature = temperature
        # reasoning=True asks the model to deliberate (chain-of-thought) before the JSON.
        self.reasoning = reasoning
        # no_think=True asks the server to disable model-level thinking (Qwen3 chat
        # template kwarg). Forwarded as extra body so OpenAI-only servers ignore it.
        self.no_think = no_think
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "not-needed"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None,
        )
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, email: Email) -> str:
        # Direct mode keeps its original key (no tag) so existing caches stay valid.
        mode = "|reason" if self.reasoning else ""
        if self.no_think:
            mode += "|nothink"
        blob = f"{self.model}{mode}|{PROMPT_VERSION}|{email.id}|{_format_message(email)}"
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _create(self, messages: list[dict]):
        """Chat completion call. Retries without `temperature` for models that reject
        it — newer models (e.g. Claude Opus 4.7) have deprecated the parameter. The
        on-disk cache, not temperature, is what makes re-runs reproducible.
        """
        kwargs: dict = {"model": self.model, "messages": messages}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.no_think:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        try:
            return self.client.chat.completions.create(**kwargs)
        except Exception as exc:  # provider-specific parameter rejection
            if "temperature" in kwargs and "temperature" in str(exc).lower():
                kwargs.pop("temperature")
                return self.client.chat.completions.create(**kwargs)
            raise

    def predict(self, email: Email, use_cache: bool = True) -> Prediction:
        cache_file = self.cache_dir / f"{self._cache_key(email)}.json"
        if use_cache and cache_file.exists():
            return Prediction(**json.loads(cache_file.read_text()))

        rng = random.Random(email.id)  # reproducible label order per email
        system = REASONING_SYSTEM_PROMPT if self.reasoning else SYSTEM_PROMPT
        resp = self._create(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": _build_user_prompt(email, rng, self.reasoning)},
            ]
        )
        raw = resp.choices[0].message.content or ""
        label, conf, rationale, ok = _parse_response(raw)
        pred = Prediction(email.id, label, conf, rationale, ok, raw)
        cache_file.write_text(json.dumps(pred.to_dict(), indent=2))
        return pred

    def predict_all(self, emails: list[Email], use_cache: bool = True) -> list[Prediction]:
        return [self.predict(e, use_cache) for e in emails]
