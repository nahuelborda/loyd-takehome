# Loyd — Applied AI Engineer Take-home

Evaluation pipeline for Loyd's `_detect_intent` LLM call — the top-of-funnel intent
classifier in the scheduling agent.

## Start here — the report

I'm shipping the writeup as a small static site. It's the primary deliverable.

- **Hosted: <https://nahuelborda.github.io/loyd-takehome/>** — no install, just open
  in a browser.
- Local: **[`site/index.html`](site/index.html)** — clone the repo and open the file.
  Five linked pages in the top nav: **Overview · Design · Results & Interpretation ·
  Discoveries · Conclusions**.
- Markdown sources of truth: **[`docs/design-document.md`](docs/design-document.md)**
  and **[`results/interpretation.md`](results/interpretation.md)**.

You don't need to run anything to read the report. The code is here if you want to
poke at it.

## Running the code (only if you want to)

The site captures my run. To run the eval yourself you'll need an API key for whichever
model you want to test — the eval is provider-agnostic via the OpenAI-compatible chat
API. Three paths, pick whichever is least friction:

### Docker (zero install beyond Docker)

```bash
export OPENAI_API_KEY=sk-...
docker compose run --rm eval        # defaults to --model gpt-4.1
# pick a model / endpoint:
MODEL=claude-opus-4-7 OPENAI_BASE_URL=https://api.anthropic.com/v1 \
  docker compose run --rm eval
```

Outputs land in `./results/run/` (mounted from your host).

### Makefile (Python + venv on host)

```bash
make install                                   # creates .venv, installs deps
make test                                      # 20 unit tests, no API calls
OPENAI_API_KEY=sk-... make eval MODEL=gpt-4.1  # run the eval
make site                                      # rebuild site/ from results/
```

### Plain Python (no Make, no Docker)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q
OPENAI_API_KEY=sk-... .venv/bin/python run_eval.py --model gpt-4.1 \
  --base-url https://api.openai.com/v1 --out results/run
#   --reasoning   A/B the reasoning-first prompt
#   --no-think    disable Qwen's chat-template thinking mode
```

## Repo layout

```
loyd-takehome/
├── site/                        ★ THE deliverable — open site/index.html
│   ├── index.html · design.html · results.html · discoveries.html
│   ├── style.css · *.svg
│
├── docs/
│   └── design-document.md       ★ primary deliverable
│
├── data/emails.jsonl            · the 25 labeled emails
│
├── detect_intent_eval/          · the eval harness (~700 lines)
│   ├── dataset.py · predictor.py · scorers.py · taxonomy.py · report.py
│
├── results/
│   ├── report.md · interpretation.md · predictions.jsonl   ← v0 (Opus direct)
│   ├── opus-reasoning/                                     ← A/B: reasoning prompt
│   ├── crosscheck-sonnet/ · crosscheck-llama4/ · crosscheck-qwen/
│
├── tests/test_scorers.py        · 20 unit tests
├── tools/build_site.py          · regenerates site/ from results/
├── run_eval.py                  · CLI entrypoint
├── review/*.pdf                 · PDF renderings of each doc
├── Dockerfile · docker-compose.yml · Makefile · requirements.txt
```

## Headline result

v0 run on `claude-opus-4-7`, 25 emails: **clean accuracy 0.846** — every error the
same error, a real request collapsed to `none` because the prompt suppressed the
model's reasoning. A reasoning-first prompt, A/B-tested, lifts clean accuracy to
**1.000** (`run_eval.py --reasoning`, `results/opus-reasoning/`).

Cross-checks corroborate on three independent runs (Sonnet, Llama-4 Scout, Qwen 3.5
122B) — same 0.846 headline, same failure mode. Full reading:
[`results/interpretation.md`](results/interpretation.md).

## Status

| Deliverable | State |
|---|---|
| Design document | ✅ [`docs/design-document.md`](docs/design-document.md) |
| Executable prototype | ✅ `detect_intent_eval/` + `run_eval.py` — 20 tests passing |
| Results + interpretation | ✅ `results/report.md` + [`results/interpretation.md`](results/interpretation.md) |
| Supporting work | tooling study · dataset notes · 45-paper literature review |
| Founding-engineer depth | cross-checks on independent-family models (Llama-4, Qwen 3.5 122B) · the reasoning-prompt A/B fix |
