"""Unit tests for the scorers, predictor parsing, and failure tagging.

The brief grades whether each scorer measures what it claims — these tests pin that
down on small synthetic inputs where the expected answer is obvious by hand.
"""

from detect_intent_eval import scorers
from detect_intent_eval.dataset import Email
from detect_intent_eval.predictor import Prediction, _parse_response
from detect_intent_eval.taxonomy import is_acceptable, tag_failures


def mk_email(id="E1", gold="schedule", fit="clean", alt=None, addr="to",
             body="this is a normal length email body with enough words",
             subject="s", thread_so_far=None):
    return Email(id=id, channel="email", from_="a@b.com", to=["loyd@loyd.ai"], cc=[],
                 loyd_addressed=addr, subject=subject, thread_so_far=thread_so_far,
                 body=body, gold_label=gold, label_fit=fit, alt_label=alt,
                 proposed_label=None, notes="")


def mk_pred(id="E1", label="schedule", conf=0.9, parse_ok=True):
    return Prediction(id, label, conf, "rationale", parse_ok, "{}")


# ---- predictor parsing -------------------------------------------------------

def test_parse_clean_json():
    label, conf, _, ok = _parse_response('{"label":"cancel","confidence":0.8,"rationale":"x"}')
    assert (label, conf, ok) == ("cancel", 0.8, True)


def test_parse_malformed_is_flagged():
    label, conf, _, ok = _parse_response("not json at all")
    assert ok is False and label == "none"  # F8: never silently accepted


def test_parse_embedded_json():
    label, _, _, ok = _parse_response('sure: {"label":"none","confidence":0.5,"rationale":"y"}')
    assert label == "none" and ok is True


def test_parse_unknown_label_falls_back():
    label, _, _, ok = _parse_response('{"label":"Schedule!","confidence":0.9,"rationale":"z"}')
    assert label == "none" and ok is False


def test_parse_reasoning_then_json():
    # reasoning-mode reply: prose first, the JSON object on the final line
    text = ('Let me think — a deferral inside an active scheduling thread still carries '
            'the thread intent.\n\n{"label":"schedule","confidence":0.62,"rationale":"x"}')
    label, conf, _, ok = _parse_response(text)
    assert label == "schedule" and conf == 0.62 and ok is True


# ---- class metrics -----------------------------------------------------------

def test_class_metrics_perfect():
    emails = [mk_email("E1", "schedule"), mk_email("E2", "cancel")]
    preds = [mk_pred("E1", "schedule"), mk_pred("E2", "cancel")]
    r = scorers.class_metrics(emails, preds)
    assert r["accuracy"] == 1.0 and r["clean_accuracy"] == 1.0
    # F1 is 1.0 for the two classes present; macro-F1 averages over all 6 labels, so
    # absent classes (zero support) pull it below 1.0 — expected, documented behaviour.
    f1_by_label = {c["label"]: c["f1"] for c in r["per_class"]}
    assert f1_by_label["schedule"] == 1.0 and f1_by_label["cancel"] == 1.0


# ---- none / OOS --------------------------------------------------------------

def test_none_oos_separates_precision_and_recall():
    # E1: true none predicted as intent (fn). E2: true intent predicted none (fp).
    emails = [mk_email("E1", "none"), mk_email("E2", "schedule")]
    preds = [mk_pred("E1", "schedule"), mk_pred("E2", "none")]
    r = scorers.none_oos_metrics(emails, preds)
    assert r["fn"] == 1 and r["fp"] == 1 and r["tp"] == 0


# ---- calibration -------------------------------------------------------------

def test_calibration_autosend_band_precision():
    emails = [mk_email("E1", "schedule"), mk_email("E2", "schedule")]
    preds = [mk_pred("E1", "schedule", 0.99), mk_pred("E2", "cancel", 0.99)]  # 1 wrong
    r = scorers.calibration_metrics(emails, preds)
    assert r["autosend_band"]["n"] == 2 and r["autosend_band"]["precision"] == 0.5


# ---- ambiguity ---------------------------------------------------------------

def test_ambiguity_accepts_alt_label():
    e = mk_email("E1", "schedule", fit="ambiguous", alt="none")
    r = scorers.ambiguity_metrics([e], [mk_pred("E1", "none", 0.6)])  # matches alt
    assert r["ambiguous_acceptable_set_rate"] == 1.0


def test_ambiguity_confidence_gap():
    clean = mk_email("E1", "schedule", fit="clean")
    amb = mk_email("E2", "schedule", fit="ambiguous", alt="none")
    r = scorers.ambiguity_metrics(
        [clean, amb], [mk_pred("E1", "schedule", 0.9), mk_pred("E2", "schedule", 0.6)]
    )
    assert abs(r["confidence_gap_clean_minus_ambiguous"] - 0.3) < 1e-9


# ---- routing curve -----------------------------------------------------------

def test_routing_curve_hitl_volume_monotonic():
    emails = [mk_email(f"E{i}", "schedule") for i in range(4)]
    preds = [mk_pred("E0", "schedule", 0.4), mk_pred("E1", "schedule", 0.6),
             mk_pred("E2", "schedule", 0.8), mk_pred("E3", "schedule", 0.99)]
    r = scorers.routing_curve(emails, preds, thresholds=(0.5, 0.9))
    vols = [c["hitl_volume"] for c in r["curve"]]
    assert vols[0] <= vols[1]  # a higher threshold routes at least as much to HITL


# ---- acceptability + failure tagging ----------------------------------------

def test_is_acceptable_ambiguous_accepts_both():
    e = mk_email("E1", "schedule", fit="ambiguous", alt="none")
    assert is_acceptable(e, "none") and is_acceptable(e, "schedule")
    assert not is_acceptable(e, "cancel")


def test_tag_failures_clean_pass_is_empty():
    e = mk_email("E1", "schedule", fit="clean")
    assert tag_failures(e, mk_pred("E1", "schedule", 0.9)) == []


def test_tag_failures_indirect_path_decoy():
    e = mk_email("E1", gold="none", fit="clean", addr="none")
    tags = tag_failures(e, mk_pred("E1", "schedule", 0.9))
    assert "F4" in tags and "F5" in tags  # decoy vocabulary + indirect-path miss


def test_tag_failures_misfit_always_f1():
    e = mk_email("E1", "reschedule", fit="misfit")
    assert "F1" in tag_failures(e, mk_pred("E1", "reschedule", 0.5))


def test_tag_failures_parse_failure_is_f8():
    e = mk_email("E1", "schedule", fit="clean")
    assert "F8" in tag_failures(e, mk_pred("E1", "none", 0.0, parse_ok=False))


def test_tag_failures_context_blind_thread_is_f9():
    # Thread state present; model dropped to none and got it wrong -> F9.
    e = mk_email("E1", "schedule", fit="clean",
                 thread_so_far="Loyd proposed times for a Reed x Caleb meeting.")
    assert "F9" in tag_failures(e, mk_pred("E1", "none", 0.6))


def test_tag_failures_context_blind_reply_subject_is_f9():
    # No thread, but reply subject signals an ongoing exchange.
    e = mk_email("E1", "schedule", fit="clean", subject="Re: lock with reed - wed 3pm")
    assert "F9" in tag_failures(e, mk_pred("E1", "none", 0.5))


def test_tag_failures_no_context_no_f9():
    # Body-only email, no thread, no Re: subject -> F9 must not fire even on a miss.
    e = mk_email("E1", "schedule", fit="clean", subject="quick one", thread_so_far=None)
    assert "F9" not in tag_failures(e, mk_pred("E1", "none", 0.5))


def test_tag_failures_correct_prediction_no_f9():
    # Context-rich but the model got it right -> F9 must not fire.
    e = mk_email("E1", "schedule", fit="clean",
                 thread_so_far="Loyd proposed times for a Reed x Caleb meeting.")
    assert "F9" not in tag_failures(e, mk_pred("E1", "schedule", 0.9))
