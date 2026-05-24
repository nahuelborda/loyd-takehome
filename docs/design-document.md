# Design Document — Evaluation Pipeline for `_detect_intent`

**Author:** Nahuel Borda · **Date:** 2026-05-21 · **Take-home:** Loyd, Applied AI Engineer

The executable prototype lives in [`../`](../) (predictions, scorers, and the results
interpretation).

---

## TL;DR

This document designs an evaluation pipeline for `_detect_intent` — Loyd's top-of-funnel
intent classifier (6 labels: `schedule`, `reschedule`, `cancel`, `query_agenda`,
`block_calendar`, `none`). My design rests on three claims, and each one changes what the
eval measures:

1. **The 6-label space is a hypothesis under test, not ground truth.** In a realistic
   25-message sample, **48% of messages are ambiguous or fit no label cleanly**. An
   accuracy score computed against those labels measures the wrong thing. The eval makes
   *label-fit* a first-class signal and turns misfits into label-space proposals.

2. **Confidence calibration is the highest-stakes measurement.** Loyd confirmed
   `_detect_intent` emits a confidence score that drives a live **HITL routing gate** and
   a **>0.95 auto-send bypass**. A miscalibrated score either floods the human queue or
   ships unreviewed errors to outside parties. The eval measures calibration directly,
   with the >0.95 band as its critical slice.

3. **Failures must be attributed to `_detect_intent` specifically.** It sits upstream of
   the whole agent; production success signals (a meeting booked) conflate every stage.
   The design separates *stage-attributable* signal from *end-to-end* signal.

The pipeline itself is a small hand-rolled Python harness — at 25 emails, a
framework's run management and dashboards cost more than they return — built around a
`Predictor → Scorer` contract that generalizes to `_parse_fields` and `_generate_email`.

---

## 1. Scope and context

`_detect_intent` is the first LLM call in Loyd's pipeline: every inbound message hits it,
and its output routes everything downstream. An error here is a first-domino error.

**In scope:** evaluation of `_detect_intent` as a single-label classifier over the
current 6-label space, on the **Gmail** channel.

**Out of scope (deliberately):**
- *iMessage* — Loyd confirmed it is not live yet and is constrained. The dataset's two
  iMessage records (E04, E05) are retained as a forward-looking channel slice but do not
  drive headline metrics.
- *Implementing* evals for `_parse_fields`, the validate-answer layer, and
  `_generate_email` — §11 covers how the approach extends to them, as the brief requests.

**The eval's purpose** is not a leaderboard score. With 25 emails it *cannot* be a
benchmark. I designed it as a **calibration instrument**: a repeatable pipeline that says *where* the
classifier fails, *why*, and *how confidently it was wrong* — and that composes with the
rest of the agent.

## 2. Design principles

Five principles, each one a constraint the rest of the document answers to:

1. **Measure the label space, not just performance within it.** If the taxonomy is wrong,
   accuracy against it is a comfortable illusion.
2. **Treat uncertainty as signal, not noise.** Genuinely ambiguous messages are facts to
   be measured, not errors to be eliminated. The research on soft-label training confirms
   what's intuitive: when humans genuinely disagree on a label, forcing a single gold
   label throws away real signal about where the boundary is hard.
3. **No single number.** Report per-class metrics, a confusion matrix, and calibration —
   never collapse to one accuracy figure that hides which failures occur.
4. **Every scorer states what it claims to measure** — and how it could be fooled. The
   brief grades exactly this.
5. **Compose with the pipeline.** The harness has a stage-agnostic contract; failure
   reporting is attributable to `_detect_intent`, not to "the agent."

## 3. The label space is a product decision

This is the central reframe, and the brief invites it: *"Defining the intent label space
is a product decision as much as engineering."*

The 25-email dataset was labeled on three
tiers of **label-fit**:

| `label_fit` | n | meaning |
|---|---|---|
| `clean` | 13 | exactly one label is correct |
| `ambiguous` | 7 | two labels are each defensible |
| `misfit` | 5 | no label in the 6-space is actually right |

**48% of messages are not clean single-label cases.** That is not noise in the data — it
is the shape of the real inbound stream. Five messages have no correct answer at all:

| ID | What it actually is | Least-wrong label | Missing label |
|---|---|---|---|
| E11 | "is our 3pm still on?" — meeting-status question | `query_agenda` | `meeting_query` |
| E19 | "is the meeting Zoom or in person?" | `query_agenda` | `meeting_query` |
| E17 | "drop Ben from the meeting" — attendee change | `reschedule` | `modify_attendees` |
| E18 | "tell the team I'll be late, don't move anything" | `none` | `notify_attendees` |
| E22 | "remind me to send the deal Friday" — task reminder | `block_calendar` | `set_reminder` |

**Consequence for the eval:** a classifier forced to pick a 6-label answer for E17 *will*
pick one, and a naive harness will score it right or wrong against an arbitrary
least-wrong gold. Both outcomes are misleading. Instead the eval:

- scores `clean` rows with exact match;
- scores `ambiguous` rows against an **acceptable set** (`gold ∪ alt_label`);
- scores `misfit` rows on a different question entirely — *did the model signal low
  confidence and route to HITL?* A misfit that is high-confidence auto-sent is the worst
  possible outcome; a misfit that is flagged uncertain is a success.

Misfit and ambiguous messages are also a **product input**: they feed a proposed v1 label
space — `modify_attendees`, `meeting_query`, `notify_attendees`, `set_reminder` — and a
recurring review process (§9). The eval does not just grade the classifier; it tells
Product the taxonomy needs to grow.

## 4. Eval strategy

### 4.1 Architecture

A hand-rolled Python harness with four components and one contract:

```
emails.jsonl ──▶ Predictor ──▶ predictions ──▶ Scorers ──▶ Reporter ──▶ report
                (message →                    (preds + gold
                 label, confidence,            → metrics)
                 rationale)
```

- **Dataset loader** — reads `data/emails.jsonl` (one labeled record per message).
- **Predictor** — `message → {label, confidence, rationale}`. Swappable: zero-shot LLM,
  few-shot, or a fine-tuned classifier, all behind one interface.
- **Scorers** — `(predictions, gold) → metrics`. Each scorer is independent and
  inspectable (§5).
- **Reporter** — emits an interpretable report: confusion matrix, per-class table,
  calibration table, per-email results, and failure-tag rollup.

The `Predictor`/`Scorer` split is the whole reason for a hand-rolled harness over a
framework — it is small, fully legible, and the same contract carries to the other
pipeline stages (§11).

### 4.2 Prediction method (the v0 prototype)

The brief permits any defensible method. The prototype uses a **zero-shot LLM call** that
mirrors production (`_detect_intent` is GPT-4.1):

- Structured output: `{label, confidence ∈ [0,1], rationale}`.
- The **label definitions** are in the prompt — sharpening how `reschedule` and `cancel`
  are described measurably reduces confusion between semantically close classes.
- Label order is **randomized per call** to neutralize position bias — models give more
  weight to whichever label appears first in the list, so shuffling neutralizes it.
- `confidence` is a verbalized score — the model states its confidence in words, then
  I convert that to a number. Whether it's reliable depends heavily on how you ask, so
  the prompt is designed for it and the score is then *checked*, not trusted (§6).

Zero-shot is the right v0 baseline — with good label descriptions a zero-shot LLM is
surprisingly competitive — and it isolates the eval design from model-training concerns.
The harness can later swap in few-shot or a fine-tuned baseline behind the same interface;
a fine-tuned discriminative model sets the realistic ceiling once enough labeled data exists.

### 4.3 Handling ambiguity and abstention

I did **not** add a 7th "abstain" label. Loyd's classifier already has the right
mechanism: a confidence score plus HITL routing below threshold. The eval evaluates
*that* — low confidence **is** the abstain signal. This keeps the eval faithful to the
production system rather than testing a classifier that does not exist.

The acceptable-set scoring for `ambiguous` rows, and the confidence-based scoring for
`misfit` rows, are described in §3 and §6.

## 5. Scorer design

Five scorers. Each is stated as a **claim** (what it measures) and a **honesty check**
(how it could mislead, and what I do about it). This section answers the brief's
explicit question — *"is the scorer measuring what it claims to measure?"*

### 5.1 Per-class precision / recall / F1 + confusion matrix
- **Claims:** *which* labels the classifier confuses with *which* — not an aggregate.
- **Reports:** a 6×6 confusion matrix; per-class P/R/F1; **macro-F1** (every class weighted
  equally) and **micro-F1** (weighted by frequency).
- **Honesty check:** the dataset is severely imbalanced (`schedule` 10, `cancel` 1). Micro
  metrics would be dominated by `schedule` and a single-accuracy number would hide
  `cancel` entirely. Macro-F1 is the headline; with `cancel` at n=1 it is high-variance,
  so the report states n per class and never presents a class metric below ~5 examples
  without that caveat.

### 5.2 `none` / out-of-scope scorer
- **Claims:** can the classifier *reject*? Out-of-scope detection is a distinct skill from
  in-scope sorting — classifiers that do well at routing known intents reliably fail on
  unknown ones — and the rejection rate degrades as the label space grows (more labels
  means more nearby attractors that pull edge cases in).
- **Reports:** precision and recall on `none` **separately**.
- **Honesty check:** precision and recall on `none` are different harms and must not be
  averaged. Low `none` **recall** = the agent acts on junk — a newsletter (E06) or an
  e-sign bot (E23) handled as a real request. Low `none` **precision** = a genuine
  request is silently dropped to `none` and the meeting never happens. The eval slices
  `none` by `loyd_addressed` to separate "junk Loyd was cc'd on" from "junk addressed to
  Loyd."

### 5.3 Calibration scorer
- **Claims:** does the confidence score mean what it says — is a 0.9 really ~90% likely
  correct? See §6 for why this is the highest-stakes scorer.
- **Reports:** a reliability diagram and Expected Calibration Error (ECE); empirical
  accuracy within the **>0.95 auto-send band** and within the **HITL-routed band**.
- **Honesty check:** ECE over only 25 points is unstable, so the scorer reports it with a
  bootstrap confidence interval and treats the **>0.95-band precision** as the primary,
  decision-relevant figure rather than a global ECE number.

### 5.4 Ambiguity-aware (acceptable-set) scorer
- **Claims:** on genuinely ambiguous inputs, does the model land in the *defensible* set,
  and is it *appropriately unsure*?
- **Reports:** acceptable-set hit rate on `ambiguous` rows (`gold ∪ alt_label`); and the
  **confidence gap** — mean confidence on `clean` rows minus mean confidence on
  `ambiguous` rows.
- **Honesty check:** a model that is *confidently* correct on ambiguous items is still
  miscalibrated against human uncertainty — it just got lucky. The confidence gap catches
  this: if confidence does **not** drop on the `ambiguous` tier, the model is
  overconfident exactly where humans disagree, and the scorer flags it as a failure even
  when the label is "right."

### 5.5 Routing / operating-point scorer
- **Claims:** at a given confidence threshold, what is the HITL load and what is the
  residual error rate that *escapes* review?
- **Reports:** a threshold sweep → an operating curve of (HITL volume, escaped-error rate,
  auto-send precision); a recommended threshold.
- **Honesty check:** this scorer measures a *product* tradeoff, not model quality — it is
  reported as a curve, not a score, so it cannot be mistaken for "the classifier is X%
  good." It exists to inform where Loyd sets the gate.

## 6. Calibration under uncertainty

This is the centerpiece, because Loyd's answers made it concrete. I treat `_detect_intent`'s
confidence score not as a diagnostic — it is **load-bearing**. It drives two live gates:

- **below threshold →** the message routes to the HITL Approval Gate (the user reviews);
- **above 0.95 (no conflicts) →** the draft **auto-sends with no human review at all**.

So calibration error has two distinct, asymmetric costs:

| Failure | Mechanism | Cost |
|---|---|---|
| Over-confident | wrong answer scores >0.95 | error **auto-sends to an outside party** — unguarded |
| Under-confident | correct answer scores low | needless HITL load; erodes trust in the agent |

I treat the **>0.95 band precision** as the single most important number. Every
error in that band is an error Loyd ships blind. A global accuracy of 90% is irrelevant
if the >0.95 band is only 96% precise and 30% of traffic lands there.

**What the eval measures for calibration:**
1. **Reliability** — binned confidence vs empirical accuracy (reliability diagram, ECE).
2. **The auto-send band** — precision at confidence >0.95; this gates the bypass.
3. **The HITL band** — of routed messages, how many were genuinely wrong (good routing)
   vs correct-but-unsure (avoidable load).
4. **The confidence gap** (§5.4) — confidence should be *lower* on the `ambiguous` and
   `misfit` tiers than on `clean`. A classifier whose confidence is flat across tiers is
   miscalibrated against human uncertainty, regardless of its accuracy.
5. **An operating-point sweep** (§5.5) — to recommend where the routing threshold sits.

Once the dataset is large enough to compute a calibration set, I'd replace the single
threshold I picked by reading the data with a smarter mechanism: instead of routing on
a single confidence cutoff, return a small set of plausible labels per email with a
mathematical guarantee that the right label is in that set at a chosen confidence level
(the conformal prediction approach — this is a guarantee independent of the data
distribution, not a heuristic). The clarification question that follows from "here are
the two most likely intents — which did you mean?" maps directly onto Loyd's existing
Approval Gate. The abstain path Loyd already has via HITL routing is the same idea;
conformal prediction just makes the guarantee rigorous.

## 7. Failure-mode taxonomy

Beyond "correct vs incorrect," I tag every wrong (or unsafely-confident)
prediction with a failure mode. The taxonomy is grounded in the real 25 emails — each
category has worked examples — and is built/extended with a measured inter-annotator
agreement check. I'd follow the MAST template: build the taxonomy from real failure
traces, then have two annotators label them independently and compute agreement (MAST
reports κ = 0.88 on their multi-agent taxonomy — that's the bar for "the categories are
actually coherent").

| # | Failure mode | Definition | Example IDs | How the eval detects it |
|---|---|---|---|---|
| F1 | **Label-space misfit** | input fits no label; model forced to pick | E11, E17, E18, E19, E22 | `label_fit=misfit` rows; high confidence here is the alarm |
| F2 | **Meeting-state confusion** | `schedule` vs `reschedule` decided without checking whether a meeting is booked | E02↔E16 | confusion matrix `schedule`↔`reschedule`; sliced by `thread_so_far` |
| F3 | **Negation / constraint miss** | an explicit "don't" is ignored | E18 ("don't move anything") | curated negation slice |
| F4 | **Decoy vocabulary** | meeting-shaped words with no intent | E06, E24, E25 | false-positive intent on `gold=none` |
| F5 | **Indirect-path miss** | cc-carryover / accidental forward not recognized as `none` | E06, E23, E24, E25 | `none` errors sliced by `loyd_addressed` |
| F6 | **Soft-intent miscalibration** | vague social "let's connect" forced to a hard label with high confidence | E07, E09, E13 | confidence gap on `ambiguous` rows |
| F7 | **Thin-context overreach** | terse message answered confidently without enough information | E12 ("next week?") | confidence vs message length |
| F8 | **Output / format failure** | model reasoned correctly but emitted malformed output | (stress-tested) | schema-validation failures, tracked separately |
| F9 | **Context blindness** | subject and/or thread state disambiguates the intent, but the model weights only the body and gets it wrong | E11, E14, E19, E20 | wrong predictions on emails where `thread_so_far` is non-empty or subject starts with `Re:` (`taxonomy.py:_is_context_rich`) — coarse proxy; an ablation that strips context and compares is the strictly-better test |

F8 matters more than it looks: in a 2025 error-analysis study of LLM classification
failures (FLARE), **70.8% of failures were parsing issues, not reasoning errors** — the
model got the logic right but emitted output the parser choked on. Conflating the two
would have me tuning prompts when the real fix is output handling — so F8 is always
counted on its own axis.

Each taxonomy category is also a **dashboard slice**: the eval reports failure counts per
category over time, so a regression shows up as "F2 doubled," not just "F1 dropped."

## 8. Dataset — sourcing and growth beyond 25

25 emails is v0 calibration material. A credible eval set needs to grow along four axes,
deliberately:

1. **Stratified production sampling (Gmail).** Sample to cover all 6 labels, every
   `loyd_addressed` stratum (`to`/`cc`/`none`), and both thread states (fresh /
   mid-thread). Random sampling would drown rare classes — `cancel` and `block_calendar`
   must be over-sampled relative to their natural rate.
2. **Mine the `none` bucket and HITL Rejects.** This is where new intents and the hardest
   cases live. Cluster `none`-labeled messages and rejected drafts to surface candidate
   labels — an LLM can mine the `none` bucket for candidate new intent categories from
   minimal labeled data. This is the concrete pipeline behind the §3 label-space proposals.
3. **Targeted synthetic generation for rare classes.** `cancel` at n=1 cannot be evaluated
   meaningfully. Generate synthetic `cancel`/`block_calendar` messages — but raw LLM
   generation is insufficient; it needs a refinement step for utility and diversity (the
   generate-then-refine pattern: raw LLM output is too repetitive and needs a second pass
   to improve coverage). Synthetic data is marked as such and never mixed silently with
   production data.
4. **Multi-annotator labeling with measured agreement.** Every new batch gets ≥2
   annotators; I'd compute Krippendorff's α and *keep* it — not just as a QA gate but as
   a standing record of where humans disagree (α handles missing data and unequal annotator
   pools; Cohen's κ is sensitive to class imbalance and will quietly change which labels
   look reliable). Disagreement is preserved as the `ambiguous` tier, not voted away:
   when annotators genuinely disagree, that's signal about where the boundary is hard,
   not noise to be eliminated.

**Hygiene:** the eval set is versioned; a held-out slice is rotated to resist
contamination as prompts and models iterate. Target: a few hundred stratified, versioned,
multi-annotated examples — the point at which I'd re-survey the framework landscape.

## 9. Production feedback loop

The brief asks how a feedback loop from production back into eval would work. Loyd's Q4
answer gives the raw signals; my job here is to use them *without fooling myself*.

**Available production signals:**

| Signal | Source | Attribution |
|---|---|---|
| HITL **Approve / Edit / Reject** | Approval Gate (user's email reply) | **stage-ish** — Edit/Reject ≈ the user correcting the agent |
| Booking **success / failure** | calendar outcome | **end-to-end** — conflates all stages |
| Explanatory outcome | Loyd's "no time found / meeting not needed / needs cancelling" message | mixed — but informative |

**The core nuance:** booking success is an *end-to-end* signal. A booking can fail for
reasons that have nothing to do with intent (no mutual slot, a downstream parse error). So
I treat booking success as a **noisy proxy** for `_detect_intent` correctness — not a direct label. The cleanest *stage-attributable* signal is **HITL Edit/Reject of the
draft** — and even that needs a lightweight attribution step, because `_detect_intent`
sits upstream of the Draft Generator: a Reject means *something* upstream was wrong, not
necessarily the intent.

**The loop:**

```
log every _detect_intent call ─▶ join to HITL outcome + booking outcome
        │                                      │
        │                          triage: low-confidence, Rejected,
        │                          or annotator-disagreement cases
        │                                      │
        └──────────◀── versioned eval set ◀── sample + label (≥2 annotators)
```

1. **Log** every call: input message, predicted label, confidence, model/prompt version,
   message id.
2. **Join** to the HITL outcome and the booking outcome.
3. **Triage** for labeling — prioritize the cases that teach the most: confidence near the
   routing threshold, HITL-Rejected drafts, and messages where the explanatory outcome
   ("meeting doesn't need to happen") contradicts a non-`none` prediction. That last one
   is a near-free `none`-recall signal.
4. **Attribute** — for Rejected drafts, a quick check (rule or LLM-assisted) of whether
   the intent was the cause, so `_parse_fields` errors are not blamed on `_detect_intent`.
5. **Feed back** — confirmed errors and new ambiguous/misfit cases enter the next eval-set
   version. Misfits also feed the §3 label-space review.

This closes the loop *and* keeps the eval honest about which stage it is measuring.

## 10. The distillation flywheel — running a cheaper model in production safely

§9's feedback loop collects labelled production data. Add one stage and it becomes a
*training* loop — the mechanism for running `_detect_intent` on a cheap, fast model
without giving up accuracy. This matters because `_detect_intent` is a high-volume
top-of-funnel call: the reasoning-first prompt that fixes the drop-to-`none` failure
(see [`../results/interpretation.md`](../results/interpretation.md)) costs a longer
completion on *every* message. Distillation is how to keep the accuracy and drop the cost.

**The shape.** A strong **teacher** — Opus with the reasoning prompt, the configuration
the A/B validated — runs *offline* (latency is free there; a Batch API is built for it)
and generates labels *and rationales* for a large corpus. A cheap, fast **student** is
fine-tuned on that corpus and serves production traffic.

**The trap — and why the eval is load-bearing.** Teacher output is *not* automatically
good training data. The v0 eval proved the teacher itself is wrong on a structured ~16%
of cases (drop-to-`none`), and the Sonnet and Llama-4 cross-checks showed the failure is
model-general.
Distil from raw teacher output and the student inherits the teacher's bugs — and if the
eval's gold is also teacher-derived, the eval will *reward* the student for reproducing
them. That is the same-model caveat, baked structurally into a training loop.

So the eval is not a step *after* distillation — it is the **QA gate inside** it:

```
teacher (Opus + reasoning prompt) ─▶ candidate label + rationale
          │
   eval + HITL filter ──▶ high-confidence clean cases → training data
          │               low-conf / ambiguous / misfit → human review
          ▼                (the teacher's rationale makes review fast)
   curated corpus ─▶ fine-tune cheap student ─▶ eval validates student
          ▲                                          │
          └──────── HITL Edit/Reject corrections ◀─── production
```

**Two non-negotiables**, both already evidenced by this take-home:
1. *Fix the teacher before distilling.* The reasoning-prompt A/B is exactly that — the
   teacher must be the corrected configuration, or the student learns the bug.
2. *The student's test gold must be independent of the teacher* (human-labelled), or the
   eval cannot see a bug the student inherited.

**Where this sits.** It is a roadmap item, not v0 — fine-tuning on 25 emails is not viable;
the teacher must first generate a corpus of hundreds to thousands (§8). Trace storage and
dataset curation would live in an observability layer (e.g. Langfuse); standing that up
now would be premature infra — the same v0 discipline that led me to a hand-rolled harness.
The point for v0 is that the eval is *designed to be that QA gate* when the time comes.

## 11. Extending to the other LLM stages

The `Predictor → Scorer` contract is the same for every stage; only the scorer changes.

- **`_parse_fields`** (GPT-4o, ~16 structured fields). Eval = **per-field** scoring:
  exact match for categoricals, normalized match for dates/durations/time-zones, set
  match for attendees. Still largely deterministic — the *same harness*, with a field-set
  scorer instead of a label scorer. Field-level confidence gets the same calibration
  treatment as §6.
- **Validate-answer layer** (the downstream check the brief mentions on p.2 — see open
  question Q6). Eval = seed known errors and measure the layer's precision/recall at
  catching them; it is itself a classifier.
- **`_generate_email`** (GPT-5.3, open-ended text). Cannot be exact-matched — this is
  where the harness needs an **LLM-as-judge** scorer with a rigorous rubric. The
  requirements are clear: precise per-class criteria dominate reliability more than
  chain-of-thought reasoning (so write the rubric carefully, don't just ask the judge to
  "think step by step"); judges carry measurable biases — position bias, length preference,
  label-frequency skew — that must be audited; and a judge must be validated against human
  agreement patterns, not just correlation with one ground-truth rater. This is the
  one stage where a framework or a vetted scorer library earns its keep.

**The composition point:** errors compound down the pipeline — a wrong intent yields a
wrong draft yields a wasted HITL review. Stage-isolated evals miss this. The design
recommends, alongside the per-stage evals, a small **end-to-end slice** that runs the full
pipeline and measures cascade behavior — with stage attribution, so a failure is traced to
its origin rather than blamed on the last stage that touched it.

## 12. Assumptions, risks, and open questions

The brief asks for assumptions to be documented.

**Assumptions made:**
- The v0 predictor is zero-shot; production parity is approximate, not exact (I don't
  have Loyd's actual `_detect_intent` prompt).
- `gold_label`s are v0, single-annotator (I labeled them by reading every email, with
  Claude as a labeling assistant); the `ambiguous`/`misfit` rows are intentionally
  contestable.
- I treat confidence as a usable signal; if production confidence is poorly calibrated,
  §6's measurements are exactly what will reveal it.

**Risks:**
- **n=25.** No metric here is a production estimate. The pipeline is the deliverable; the
  numbers are calibration. §8 is the mitigation.
- **Verbalized confidence** — asking the model to state its confidence in words, then
  converting that to a number — can be unreliable if the prompt doesn't elicit it
  carefully. §4.2 designs the prompt for it and §6 verifies the resulting scores rather
  than assuming they're well-calibrated.

**Open questions for the Loyd team:**
- The real production intent mix (to weight the eval set).
- Which misclassifications hurt most (to weight the scorers).
- What the "validate-answer layer" actually is (brief, p.2).
- Prompt/model change cadence (to set the regression-test cadence).
- Whether anything filters messages before `_detect_intent`.

## 13. What v0 delivered, and the roadmap

**v0 (this take-home) delivered** — all complete; full results in
[`../results/interpretation.md`](../results/interpretation.md):
- the labeled 25-email dataset with the three-tier `label_fit` annotation;
- the hand-rolled harness — `Predictor → Scorers → Reporter`;
- a zero-shot predictor producing `{label, confidence, rationale}` for all 25 emails;
- the five scorers of §5 and the failure-tagging of §7;
- an interpretable report and a written interpretation.

**What the run found.** I localised the classifier's failures to a single mode —
real requests collapsed to `none` — traced it to a prompt that suppressed reasoning, and
a reasoning-first prompt A/B confirmed the fix (clean accuracy 0.846 → 1.000). It also
showed what a prompt fix *cannot* do: the misfit E19 needs a label, not a better prompt —
the reasoning-limited vs label-space-limited split, made empirical.

**Roadmap:**
1. Grow the dataset per §8 to a few hundred stratified, multi-annotated examples.
2. Stand up the §9 feedback loop; make HITL Edit/Reject the continuous labeling source.
3. Decide the reasoning-prompt tradeoff — latency on every call vs the §10 distillation
   flywheel — then replace the threshold I picked by reading the data with a conformal
   prediction calibration set (§6).
4. Run the §3 label-space review and ship a v1 taxonomy — I'd start with
   `modify_attendees` and `meeting_query` based on what the dataset already surfaces.
5. Extend the `Scorer` contract to the other stages (§11).

The through-line: **a small, legible eval that runs on every change beats a comprehensive
one that runs rarely.** v0 is built to be that small, legible thing — and to grow without
being rewritten.
