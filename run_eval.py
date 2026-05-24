#!/usr/bin/env python3
"""Run the `_detect_intent` evaluation pipeline end to end.

    python run_eval.py --model gpt-4.1
    python run_eval.py --model qwen --base-url http://localhost:11434/v1

Predictions are cached under runs/cache/, so re-runs are free. Outputs go to results/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from detect_intent_eval import scorers
from detect_intent_eval.dataset import load_emails
from detect_intent_eval.predictor import ZeroShotPredictor
from detect_intent_eval.report import build_report


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Loyd's _detect_intent classifier.")
    ap.add_argument("--data", default="data/emails.jsonl")
    ap.add_argument("--model", default="gpt-4.1", help="model name (OpenAI-compatible)")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL")
    ap.add_argument("--out", default="results", help="output directory")
    ap.add_argument("--no-cache", action="store_true", help="ignore cached predictions")
    ap.add_argument("--reasoning", action="store_true",
                    help="prompt the model to reason step by step before the JSON")
    ap.add_argument("--no-think", action="store_true",
                    help="ask the server to disable model-level thinking "
                         "(Qwen3 chat-template kwarg; ignored by other models)")
    args = ap.parse_args()

    emails = load_emails(args.data)
    print(f"Loaded {len(emails)} emails from {args.data}")

    predictor = ZeroShotPredictor(
        model=args.model, base_url=args.base_url,
        reasoning=args.reasoning, no_think=args.no_think,
    )
    preds = predictor.predict_all(emails, use_cache=not args.no_cache)
    mode = "reasoning" if args.reasoning else ("direct/no-think" if args.no_think else "direct")
    print(f"Produced {len(preds)} predictions with model '{args.model}' ({mode} prompt)")

    cm = scorers.class_metrics(emails, preds)
    none_res = scorers.none_oos_metrics(emails, preds)
    cal = scorers.calibration_metrics(emails, preds)
    amb = scorers.ambiguity_metrics(emails, preds)
    route = scorers.routing_curve(emails, preds)

    report = build_report(emails, preds, args.model, cm, none_res, cal, amb, route)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(report, encoding="utf-8")
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p.to_dict()) + "\n")
    print(f"Wrote {out / 'report.md'} and {out / 'predictions.jsonl'}")

    asb = cal["autosend_band"]
    print("\nSummary:")
    print(f"  clean accuracy         : {cm['clean_accuracy']}")
    print(f"  macro-F1 (all 25)      : {cm['macro_f1']:.3f}")
    print(f"  ECE                    : {cal['ece']:.3f}")
    print(f"  >0.95 band             : n={asb['n']}  precision={asb['precision']}")
    parse_fail = sum(1 for p in preds if not p.parse_ok)
    print(f"  parse failures (F8)    : {parse_fail}")


if __name__ == "__main__":
    main()
