"""Scorers for the `_detect_intent` eval.

Each scorer is a pure function with a docstring stating what it CLAIMS to measure and
an HONESTY CHECK — how it could mislead, and what is done about it. This is the brief's
explicit grading question: "is the scorer measuring what it claims to measure?"
"""

from __future__ import annotations

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .dataset import LABELS, Email
from .predictor import Prediction


def class_metrics(emails: list[Email], preds: list[Prediction]) -> dict:
    """CLAIMS: which labels the classifier confuses with which — not one aggregate.

    HONESTY CHECK: the dataset is severely imbalanced, so a single accuracy number would
    hide rare classes. Macro-F1 (every class weighted equally) is the headline; micro-F1
    equals accuracy and is reported only for contrast. `clean_accuracy` (exact match over
    `label_fit == clean` rows) is the cleanest signal — ambiguous/misfit rows have soft
    golds and are scored honestly by `ambiguity_metrics`.
    """
    gold = [e.gold_label for e in emails]
    pred = [p.label for p in preds]

    cm = confusion_matrix(gold, pred, labels=LABELS)
    p, r, f1, support = precision_recall_fscore_support(
        gold, pred, labels=LABELS, zero_division=0
    )
    _, _, micro_f1, _ = precision_recall_fscore_support(
        gold, pred, labels=LABELS, average="micro", zero_division=0
    )
    accuracy = sum(g == pp for g, pp in zip(gold, pred)) / len(gold)

    clean = [(e, pp) for e, pp in zip(emails, pred) if e.label_fit == "clean"]
    clean_acc = (
        sum(e.gold_label == pp for e, pp in clean) / len(clean) if clean else None
    )
    return {
        "labels": LABELS,
        "confusion_matrix": cm.tolist(),
        "per_class": [
            {
                "label": LABELS[i],
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(LABELS))
        ],
        "macro_f1": float(f1.mean()),
        "micro_f1": float(micro_f1),
        "accuracy": accuracy,
        "clean_accuracy": clean_acc,
        "n_clean": len(clean),
        "n_total": len(emails),
    }


def none_oos_metrics(emails: list[Email], preds: list[Prediction]) -> dict:
    """CLAIMS: can the classifier REJECT? Out-of-scope detection is a distinct skill
    from in-scope sorting (arXiv 1909.02027).

    HONESTY CHECK: precision and recall on `none` are different harms and must never be
    averaged. Low `none` recall = junk acted on (a newsletter handled as a request);
    low `none` precision = a real request silently dropped to `none`. Also sliced by
    `loyd_addressed` to separate indirect-path noise.
    """
    gold = [e.gold_label for e in emails]
    pred = [p.label for p in preds]
    tp = sum(g == "none" and pp == "none" for g, pp in zip(gold, pred))
    fp = sum(g != "none" and pp == "none" for g, pp in zip(gold, pred))
    fn = sum(g == "none" and pp != "none" for g, pp in zip(gold, pred))

    by_addr: dict[str, dict] = {}
    for e, pp in zip(emails, pred):
        b = by_addr.setdefault(
            e.loyd_addressed,
            {"n": 0, "gold_none": 0, "pred_none_correct": 0, "false_intent": 0},
        )
        b["n"] += 1
        if e.gold_label == "none":
            b["gold_none"] += 1
            if pp == "none":
                b["pred_none_correct"] += 1
            else:
                b["false_intent"] += 1
    return {
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "recall": tp / (tp + fn) if (tp + fn) else None,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "by_addressing": by_addr,
    }


_BINS = ((0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 0.95), (0.95, 1.0001))


def calibration_metrics(emails: list[Email], preds: list[Prediction], bins=_BINS) -> dict:
    """CLAIMS: does the confidence score mean what it says — is a 0.9 really ~90%?

    HONESTY CHECK: ECE over 25 points is unstable, so bin counts are always reported
    alongside it. The decision-relevant figure is the >0.95-band precision: that band
    auto-sends with no human review, so every error in it ships unreviewed.
    """
    rows = [(p.confidence, p.label == e.gold_label) for e, p in zip(emails, preds)]
    n = len(rows)

    bin_stats, ece = [], 0.0
    for lo, hi in bins:
        members = [(c, ok) for c, ok in rows if lo <= c < hi]
        if not members:
            bin_stats.append({"range": [lo, hi], "n": 0, "mean_conf": None, "accuracy": None})
            continue
        acc = sum(ok for _, ok in members) / len(members)
        mean_conf = sum(c for c, _ in members) / len(members)
        ece += (len(members) / n) * abs(acc - mean_conf)
        bin_stats.append(
            {"range": [lo, hi], "n": len(members), "mean_conf": mean_conf, "accuracy": acc}
        )

    autosend = [(c, ok) for c, ok in rows if c > 0.95]
    return {
        "bins": bin_stats,
        "ece": ece,
        "autosend_band": {
            "n": len(autosend),
            "share": len(autosend) / n if n else 0.0,
            "precision": sum(ok for _, ok in autosend) / len(autosend) if autosend else None,
        },
    }


def ambiguity_metrics(emails: list[Email], preds: list[Prediction]) -> dict:
    """CLAIMS: on ambiguous inputs, does the model land in the defensible set AND stay
    appropriately unsure?

    HONESTY CHECK: a model that is *confidently* correct on an ambiguous item is still
    miscalibrated against human uncertainty — it got lucky. The confidence gap (clean vs
    ambiguous) catches this even when the label is "right".
    """
    by_fit: dict[str, list] = {"clean": [], "ambiguous": [], "misfit": []}
    for e, p in zip(emails, preds):
        by_fit[e.label_fit].append((e, p))

    def mean_conf(rows):
        return sum(p.confidence for _, p in rows) / len(rows) if rows else None

    amb = by_fit["ambiguous"]
    amb_hits = sum(p.label in {e.gold_label, e.alt_label} for e, p in amb)
    mis = by_fit["misfit"]

    conf_clean, conf_amb = mean_conf(by_fit["clean"]), mean_conf(amb)
    return {
        "mean_confidence": {
            "clean": conf_clean,
            "ambiguous": conf_amb,
            "misfit": mean_conf(mis),
        },
        "confidence_gap_clean_minus_ambiguous": (
            conf_clean - conf_amb if (conf_clean is not None and conf_amb is not None) else None
        ),
        "ambiguous_acceptable_set_rate": amb_hits / len(amb) if amb else None,
        "n_ambiguous": len(amb),
        "misfit_least_wrong_matches": sum(p.label == e.gold_label for e, p in mis),
        "n_misfit": len(mis),
        "misfit_high_confidence_count": sum(p.confidence > 0.95 for _, p in mis),
    }


def routing_curve(
    emails: list[Email],
    preds: list[Prediction],
    thresholds=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
) -> dict:
    """CLAIMS: at a confidence threshold, what HITL load and what residual escaped-error
    rate result?

    HONESTY CHECK: this measures a PRODUCT tradeoff, not model quality. It is reported as
    a curve, never a single score, so it cannot be mistaken for "the classifier is X%
    good". It exists to inform where Loyd sets the routing gate.
    """
    rows = [(p.confidence, p.label == e.gold_label) for e, p in zip(emails, preds)]
    n = len(rows)
    curve = []
    for t in thresholds:
        auto = [(c, ok) for c, ok in rows if c >= t]
        routed = [(c, ok) for c, ok in rows if c < t]
        escaped = sum(1 for _, ok in auto if not ok)
        curve.append(
            {
                "threshold": t,
                "hitl_volume": len(routed) / n if n else 0.0,
                "auto_n": len(auto),
                "autosend_precision": (
                    sum(ok for _, ok in auto) / len(auto) if auto else None
                ),
                "escaped_errors": escaped,
                "escaped_error_rate": escaped / len(auto) if auto else None,
            }
        )
    return {"curve": curve}
