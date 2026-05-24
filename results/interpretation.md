# Results Interpretation — v0 eval of `_detect_intent`

**Model:** `claude-opus-4-7` · **Dataset:** 25 emails · **Date:** 2026-05-21
· Metrics: [`report.md`](report.md) · Raw predictions: [`predictions.jsonl`](predictions.jsonl)

## The one-sentence finding

The eval pinned the v0 classifier's failures to a single mode — **every email it got
wrong, it got wrong by predicting `none`** — and the harness then traced it to a prompt
that suppressed the model's reasoning. With a reasoning-first prompt, **clean accuracy
goes from 11/13 to 13/13.** Measure → localise → fix → validate, in one harness.

## 1. Every error is the same error: a real request dropped to `none`

After I scored honestly (ambiguous rows credited for either defensible label), **4 of 25
predictions are wrong — and all 4 are a genuine intent collapsed to `none`:**

| ID | gold | predicted | confidence | what it is |
|----|------|-----------|------------|------------|
| E14 | schedule | `none` | 0.78 | "Let me check with my team, get back to you tomorrow" — a deferral *inside an active scheduling thread* |
| E20 | schedule | `none` | 0.82 | "Locked in. See you Wednesday." — a slot *confirmation* |
| E11 | query_agenda | `none` | 0.62 | "is this still on?" — a status question about a booked meeting |
| E19 | query_agenda | `none` | 0.78 | "is the meeting Zoom or in person?" — a question about a meeting |

The pattern looked structural to me, not random. The model classifies the **message body in
isolation**; when a message carries no standalone, explicit request, it defaults to
`none`. E14 and E20 are the sharpest cases, both *clean* `schedule` rows. E14 ("get back to you
tomorrow") is a deferral inside an active scheduling thread — the model had the
`thread_so_far` state and ignored it. E20 ("Locked in. See you Wednesday.") has no thread
provided at all — its intent lives in the subject line ("Re: lock with reed — wed 3pm"),
which the model also ignored. The shared failure is **context blindness**: the model
reads the body and nothing else.

This shows up in two scorer views that agree:
- **`none` precision = 0.636** (recall = 1.000). The model never acted on junk — it only
  ever *under*-fires. It errs conservative: it drops real work rather than inventing work.
- **`query_agenda` recall = 0.33** (1 of 3) and **`schedule` recall = 0.70**. Both are
  dragged down by the same `→ none` drops.

For a scheduling agent this is the failure that matters most: a silently dropped meeting
request is **invisible** — no error, no reply, the user's meeting just never gets booked.

An independent model reproduces exactly these drops — see **Cross-check** below.

## 2. The taxonomy missed it — and that is the taxonomy working

Only **F1** fired (the 5 label-space misfits, by construction). F2–F8 are all zero —
including no tag on E14 or E20, the two genuine clean errors. The run surfaced a failure
mode the taxonomy **does not yet have a code for**.

That is the taxonomy behaving as designed: it is a living artifact, and a real run is how
you discover the missing category. **Action: add F9 — "context blindness": the model
classifies the body alone, ignoring intent-bearing context outside it — the thread state
(E14) or the subject line (E20)** — and re-tag. E14 and E20 are its first members.

## 3. Calibration: strong where it is hard, one weak pocket

What I found encouraging is calibration on uncertainty — the axis the brief
calls out:

- **Confidence gap (clean − ambiguous) = 0.241** (clean 0.898, ambiguous 0.657). The
  model is *appropriately less sure* on ambiguous inputs.
- **Misfit mean confidence 0.644, and zero misfits answered above 0.95.** The model never
  confidently auto-sends something that fits no label — the worst-case outcome did not
  occur.
- **Ambiguous acceptable-set hit rate = 7/7.** Every genuinely ambiguous email landed on
  the gold label or the documented alternative.

The weak pocket is narrow and specific: the **0.70–0.90 confidence bin** holds 8 emails
at mean confidence 0.78 but only **62.5% accuracy**. The model's mistakes are made at
*moderate* confidence (~0.78–0.82), not low confidence — a wrong answer that does not
look uncertain. ECE is 0.116, but with n=25 the bin-level story above is more reliable
than the single ECE number.

## 4. What this run says about the routing threshold

> **Scope caveat first:** every number in this section is from *my* zero-shot prompt,
> not Loyd's production `_detect_intent` prompt (which I don't have). The methodology
> and the harness transfer directly; the specific threshold number does not, because
> the model's confidence distribution depends on the prompt. To get a defensible
> threshold for production, point this harness at the real prompt — the routing
> scorer outputs the curve in one command.

What the run shows on **this** prompt:

- **The >0.95 auto-send band: 8 emails (32% of traffic), precision 1.000.** Nothing
  wrong shipped at that threshold in this run. That's the single most decision-relevant
  number, because that band has no human review.
- **The routing curve shows where errors escape.** Errors escape into auto-send at every
  threshold **≤ 0.82**: at 0.70, three errors escape (17.6%); at 0.80, one; only at
  **≥ 0.90** do zero errors escape. The model's mistakes on this prompt cluster at
  ~0.78–0.82, which is what sets that floor.

The defensible takeaway is methodological, not prescriptive: **the >0.95-band precision
and the routing curve are the two numbers that gate auto-send risk and HITL load
respectively** — both should be measured continuously on the production prompt and
re-checked on every prompt or model change. The exact threshold falls out of that
measurement; on my prompt it lands at ≥ 0.90.

## 5. What went right — and the tradeoff inside it

Decoy vocabulary (F4) and indirect-path noise (F5): **zero failures.** All four
indirect-path emails — the newsletter (E06), the e-sign bot (E23), the festival
auto-reply (E24), the social "talk soon" note (E25) — were correctly `none`. Industry
jargon ("co-fi", "committee meets in 4–6 weeks") did not trigger a false intent.

This is the *same conservative bias* that causes the E14/E20 drops. It is one disposition
with two faces: robust against junk, fragile on terse real requests. Naming it as a
single tradeoff — not a separate win and loss — is the accurate reading.

## Cross-check — three runs reproduce the core failure

The same-model caveat — predictor and dataset labeler are both Claude Opus 4.7 — is the
obvious risk for clean-case agreement. Three cross-checks were run to test it (Sonnet
within family for a tier comparison; Llama-4 Scout and Qwen 3.5 122B as independent
model families):

- **`llama4-scout-17b` (Meta) — independent family #1.** Output in
  [`crosscheck-llama4/`](crosscheck-llama4/). Clean accuracy **0.846 — identical to
  Opus**, macro-F1 0.821, ECE 0.052. A non-Claude model, predicting blind, lands on the
  same clean accuracy and the same headline failure: **E14 dropped to `none` at 0.80
  confidence**.
- **`qwen3.5-122b-a10b-abliterated` (Alibaba, no-think) — independent family #2.**
  Output in [`crosscheck-qwen/`](crosscheck-qwen/). Clean accuracy **0.846 — identical
  to Opus and Llama-4**, macro-F1 0.841, ECE 0.119. Same E14 → `none` miss. A second
  independent family agreeing on the headline number and the headline failure is the
  strongest answer I can give to the same-model concern at this scale.
- **`claude-sonnet-4-6` (within-family tier comparison).** Output in
  [`crosscheck-sonnet/`](crosscheck-sonnet/). Clean accuracy 0.769, ECE 0.197 — weaker
  overall, and crucially makes its errors at *higher* confidence (E14 wrong at 0.87),
  which makes the case for measuring (and probably raising) the routing threshold stronger,
  not weaker — but the exact number still depends on the production prompt's calibration.

Three findings sharpen because of the cross-checks:

- **E14 is the universal failure.** All four models — Opus, Sonnet, Llama-4, Qwen — drop
  *"Let me check with my team and get back to you tomorrow"* to `none`. This is the
  drop-to-`none` failure at its purest, and it is not model-specific.
- **F2 (meeting-state confusion) is empirically validated.** Both Sonnet and Llama-4 trip
  the E16 reschedule decoy — exactly the failure the design doc's E02 ↔ E16 pair was
  built to expose, and the taxonomy tagged it automatically. Opus avoided it; two other
  models did not. That is the case for keeping rare-but-anticipated modes in the taxonomy.
- **E20 is recoverable from context.** Llama-4 and Qwen both classified E20 correctly by
  reading the subject line; Opus and Sonnet both ignored it. Not every "context
  blindness" case is universal — E20 is a softer instance of the same failure than E14.

## The fix, tested — letting the model reason

The v0 prompt told the model to *"Return ONLY a JSON object."* That instruction
suppressed the model's reasoning — and the suppression *was* the bug. Re-running with a
**reasoning-first prompt** (reason step by step, then emit the JSON last;
`run_eval.py --reasoning`, output in [`opus-reasoning/`](opus-reasoning/)) confirms it:

| Metric | v0 direct prompt | reasoning prompt |
|---|---|---|
| Clean accuracy | 0.846 (11/13) | **1.000 (13/13)** |
| Macro-F1 (all 25) | 0.817 | **0.917** |
| Acceptable-set errors | 4 | **2** |
| ECE | 0.116 | 0.105 |

It fixed exactly the predicted cases: **E14, E20, and E11** all flip to correct — the
model now reasons that "a terse in-thread reply still carries the thread's intent"
instead of defaulting to `none`. (One ambiguous case, E21, shifts the other way — a
low-confidence 0.55 call that routes to HITL regardless; net acceptable-set errors 4 → 2.)

Two honest qualifications:
- **It does not fix E19.** E19 is a *misfit* — a question about a meeting, with no
  correct label in the 6-label space. No prompt fixes a missing label. This is the clean
  empirical split the design doc predicted: a reasoning prompt fixes *reasoning-limited*
  errors; *label-space-limited* errors need the v1 taxonomy, not a better prompt.
- **It is not free.** A reasoning prompt means a longer completion on every call — real
  latency and cost for a top-of-funnel, high-volume classifier. The eval surfaces the
  tradeoff; whether to pay it — or distil the reasoning model's outputs into a cheap
  direct-prompt student — is a product call.

This is the eval doing its whole job in one loop: it **measured** the failure,
**localised** it to one mode, **guided** a fix, **validated** the fix, and **bounded**
what the fix can and cannot do.

## Caveats

- **Same-model caveat — substantially mitigated (see Cross-check above).** The predictor
  (`claude-opus-4-7`) is the same model that produced the dataset's gold labels, so
  clean-case agreement could in principle be inflated. The independent-family
  `llama4-scout-17b` cross-check substantially answers this: a non-Claude model,
  predicting blind, lands on the same clean accuracy (0.846) and the same headline error
  (E14 → `none`). The production fix remains structural — labeler and predictor should be
  different models, with human-labeled gold for the ambiguous tier.
- **n = 25.** This is calibration material, not a benchmark. Per-class metrics below ~5
  support (`cancel` n=1, `reschedule`/`block_calendar` n=2) are directional only.

## What I would do next

1. **Add failure mode F9** (in-thread reply treated as standalone) and re-tag.
2. **Grow the eval set** (design doc §8) with deliberate emphasis on in-thread replies
   (deferrals, confirmations) and `query_agenda` — recall 0.33 on n=3 is not yet trustworthy.
3. **Re-run the routing scorer against the production `_detect_intent` prompt** to set
   a defensible HITL routing threshold. The harness emits the full curve in one command;
   on my zero-shot prompt the floor lands at ≥ 0.90, but that number is prompt-dependent.
4. **The reasoning-prompt fix is validated** (see "The fix, tested") — the open decision
   is the production tradeoff: pay the reasoning latency on every call, or distil the
   reasoning model's outputs into a cheap direct-prompt student (design doc §10).
5. **Wire the production feedback loop** (design doc §9): E14/E20-type drops would surface
   as HITL `Edit`/`Reject` events — the cheapest continuous source of exactly these labels.
6. **Split labeler and predictor** so the same-model caveat goes away.
