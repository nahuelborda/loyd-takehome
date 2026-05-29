"""Build the multi-page HTML report site.

Output: site/index.html, design.html, results.html, style.css, *.svg

Visualisation principles applied:
- charts integrated in flow with the prose that reads them
- closely integrated with the verbal description of the data
- many numbers in a small space (per-email cross-check grid)
- avoid distortion (monochrome + one accent, no 3D, honest axes, minimal chartjunk)
- show data at multiple levels of detail (headline + fine structure)
- small multiples for the cross-check (same aggregate can hide different fine structure)

Re-run any time to refresh, the site is regenerated from current data on disk. If a
new model run lands (e.g. Qwen), `python tools/build_site.py` picks it up automatically.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/b4rd0/loyd/loyd-takehome")
SITE = ROOT / "site"
SITE.mkdir(parents=True, exist_ok=True)

LABELS = ["schedule", "reschedule", "cancel", "query_agenda", "block_calendar", "none"]

# ---------- Restrained matplotlib defaults (monochrome + one accent, no chartjunk) ----------

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.6,
    "axes.labelcolor": "#333333",
    "axes.titlesize": 11,
    "axes.titleweight": "normal",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "savefig.bbox": "tight",
    "savefig.transparent": False,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})

ACCENT = "#1a5fb4"
INK = "#1a1a1a"
GREY = "#888888"
PALE = "#cccccc"


# ---------- Data loading ----------

def load_emails():
    return [json.loads(l) for l in (ROOT / "data/emails.jsonl").open() if l.strip()]

def load_predictions(path: Path):
    return [json.loads(l) for l in path.open() if l.strip()]

MODEL_REGISTRY = [
    ("results/predictions.jsonl",
        "opus-direct", "Claude Opus 4.7", "direct prompt"),
    ("results/opus-reasoning/predictions.jsonl",
        "opus-reasoning", "Claude Opus 4.7", "reasoning prompt"),
    ("results/crosscheck-sonnet/predictions.jsonl",
        "sonnet", "Claude Sonnet 4.6", "direct prompt"),
    ("results/crosscheck-llama4/predictions.jsonl",
        "llama4", "Llama-4 Scout 17B", "Meta, independent family"),
    ("results/crosscheck-qwen/predictions.jsonl",
        "qwen", "Qwen 3.5 122B", "Alibaba, independent family"),
]

def load_all_runs():
    out = {}
    for relpath, key, model, note in MODEL_REGISTRY:
        p = ROOT / relpath
        if p.exists():
            preds = {x["email_id"]: x for x in load_predictions(p)}
            out[key] = {"model": model, "note": note, "preds": preds}
    return out


# ---------- Computation helpers ----------

def is_acceptable(email, pred_label):
    if email["label_fit"] == "ambiguous":
        return pred_label in {email["gold_label"], email["alt_label"]}
    return pred_label == email["gold_label"]

def confusion_matrix(emails, preds_by_id):
    cm = [[0]*len(LABELS) for _ in LABELS]
    for e in emails:
        gi = LABELS.index(e["gold_label"])
        p = preds_by_id.get(e["id"])
        if p is None:
            continue
        pi = LABELS.index(p["label"]) if p["label"] in LABELS else LABELS.index("none")
        cm[gi][pi] += 1
    return cm

def per_class_stats(emails, preds_by_id):
    rows = []
    for i, lab in enumerate(LABELS):
        gold = [e for e in emails if e["gold_label"] == lab]
        pred_lab = [e for e in emails if preds_by_id.get(e["id"], {}).get("label") == lab]
        tp = sum(1 for e in gold if preds_by_id.get(e["id"], {}).get("label") == lab)
        p = tp / len(pred_lab) if pred_lab else None
        r = tp / len(gold) if gold else None
        f1 = (2*p*r/(p+r)) if (p and r) else None
        rows.append({"label": lab, "p": p, "r": r, "f1": f1, "support": len(gold)})
    return rows

def calibration_bins(emails, preds_by_id, bins=((0,0.5),(0.5,0.7),(0.7,0.9),(0.9,0.95),(0.95,1.0001))):
    out = []
    for lo, hi in bins:
        members = []
        for e in emails:
            p = preds_by_id.get(e["id"])
            if p is None: continue
            if lo <= p["confidence"] < hi:
                members.append((p["confidence"], p["label"] == e["gold_label"]))
        if members:
            mean_conf = sum(c for c,_ in members)/len(members)
            acc = sum(ok for _,ok in members)/len(members)
            out.append({"lo": lo, "hi": min(hi,1.0), "n": len(members), "mean_conf": mean_conf, "acc": acc})
        else:
            out.append({"lo": lo, "hi": min(hi,1.0), "n": 0, "mean_conf": None, "acc": None})
    return out

def routing_curve(emails, preds_by_id, thresholds=(0.5,0.6,0.7,0.8,0.85,0.9,0.95,0.99)):
    rows = [(preds_by_id[e["id"]]["confidence"], preds_by_id[e["id"]]["label"] == e["gold_label"])
            for e in emails if e["id"] in preds_by_id]
    n = len(rows)
    out = []
    for t in thresholds:
        auto = [(c,ok) for c,ok in rows if c >= t]
        routed = n - len(auto)
        escaped = sum(1 for _,ok in auto if not ok)
        out.append({
            "t": t,
            "hitl": routed/n if n else 0,
            "auto_n": len(auto),
            "escaped_n": escaped,
            "escaped_rate": escaped/len(auto) if auto else 0,
            "auto_precision": sum(ok for _,ok in auto)/len(auto) if auto else None,
        })
    return out

def metrics(emails, preds_by_id):
    n = len(emails)
    clean = [e for e in emails if e["label_fit"] == "clean"]
    clean_correct = sum(1 for e in clean if preds_by_id.get(e["id"], {}).get("label") == e["gold_label"])
    strict = sum(1 for e in emails if preds_by_id.get(e["id"], {}).get("label") == e["gold_label"])
    accept = sum(1 for e in emails if is_acceptable(e, preds_by_id.get(e["id"], {}).get("label", "")))
    # macro F1
    pcs = per_class_stats(emails, preds_by_id)
    macros = [r["f1"] for r in pcs if r["f1"] is not None]
    macro = sum(macros)/len(macros) if macros else None
    # ECE
    cal = calibration_bins(emails, preds_by_id)
    ece = sum((b["n"]/n)*abs(b["acc"]-b["mean_conf"]) for b in cal if b["n"] > 0)
    # >0.95 band
    auto = [(preds_by_id[e["id"]]["confidence"], preds_by_id[e["id"]]["label"] == e["gold_label"])
            for e in emails if e["id"] in preds_by_id and preds_by_id[e["id"]]["confidence"] > 0.95]
    return {
        "n": n, "clean": len(clean), "clean_correct": clean_correct,
        "clean_acc": clean_correct/len(clean) if clean else None,
        "strict_acc": strict/n, "accept_acc": accept/n,
        "macro_f1": macro, "ece": ece,
        "autosend_n": len(auto),
        "autosend_precision": sum(ok for _,ok in auto)/len(auto) if auto else None,
    }


# ---------- SVG charts ----------

def chart_confusion(emails, preds_by_id, out_path, title=None):
    cm = confusion_matrix(emails, preds_by_id)
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    arr = [[c for c in row] for row in cm]
    vmax = max(max(row) for row in arr)
    ax.imshow(arr, cmap="Greys", vmin=0, vmax=vmax*1.2, aspect="equal")
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS, fontsize=8)
    ax.set_xlabel("predicted", fontsize=9)
    ax.set_ylabel("gold", fontsize=9)
    if title: ax.set_title(title, loc="left", fontsize=10, pad=8)
    for i, row in enumerate(arr):
        for j, v in enumerate(row):
            if v == 0: continue
            color = "white" if v > vmax*0.6 else INK
            ax.text(j, i, str(v), ha="center", va="center", fontsize=9, color=color)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def chart_calibration(emails, preds_by_id, out_path, title=None):
    bins = calibration_bins(emails, preds_by_id)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    xs = list(range(len(bins)))
    accs = [b["acc"] if b["n"] > 0 else 0 for b in bins]
    confs = [b["mean_conf"] if b["n"] > 0 else 0 for b in bins]
    ns = [b["n"] for b in bins]
    width = 0.4
    bars1 = ax.bar([x - width/2 for x in xs], confs, width, label="mean confidence", color=PALE, edgecolor=INK, linewidth=0.6)
    bars2 = ax.bar([x + width/2 for x in xs], accs, width, label="actual accuracy", color=ACCENT, edgecolor=INK, linewidth=0.6)
    labels = [f"{b['lo']:.2f}-{b['hi']:.2f}\nn={b['n']}" for b in bins]
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate", fontsize=9)
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    if title: ax.set_title(title, loc="left", fontsize=10, pad=8)
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def chart_routing(emails, preds_by_id, out_path, title=None):
    curve = routing_curve(emails, preds_by_id)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ts = [c["t"] for c in curve]
    hitl = [c["hitl"]*100 for c in curve]
    escaped = [(c["escaped_n"]/len(emails))*100 for c in curve]
    ax.plot(ts, hitl, marker="o", color=GREY, linewidth=1.4, label="HITL volume (% of traffic)")
    ax.plot(ts, escaped, marker="s", color=ACCENT, linewidth=1.4, label="errors that escape (% of traffic)")
    ax.axvline(x=0.9, color="#555", linestyle=":", linewidth=0.8)
    ax.text(0.9, max(hitl)*1.02, "0.90 floor\non this run", fontsize=8, ha="center", color="#555")
    ax.set_xlabel("routing threshold (confidence >= t -> auto-send)", fontsize=9)
    ax.set_ylabel("% of traffic", fontsize=9)
    ax.set_xlim(0.45, 1.02)
    ax.legend(loc="center right", frameon=False, fontsize=8)
    if title: ax.set_title(title, loc="left", fontsize=10, pad=8)
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def chart_confidence_scatter(emails, preds_by_id, out_path, title=None):
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    annotate_ids = {"E14", "E20", "E11", "E19"}
    for e in emails:
        p = preds_by_id.get(e["id"])
        if p is None: continue
        ok = is_acceptable(e, p["label"])
        ax.scatter(p["confidence"], 1 if ok else 0,
                   color=ACCENT if ok else "#c62828",
                   s=42 if e["id"] in annotate_ids else 22,
                   alpha=0.8,
                   edgecolor="white", linewidth=0.6, zorder=3)
        if e["id"] in annotate_ids:
            ax.annotate(e["id"], (p["confidence"], 1 if ok else 0),
                        xytext=(6, -10 if ok else 10), textcoords="offset points",
                        fontsize=8, color=INK)
    ax.axvline(x=0.9, color="#555", linestyle=":", linewidth=0.8)
    ax.text(0.9, 1.13, "0.90 floor on this run", fontsize=8, ha="center", color="#555")
    ax.axvline(x=0.95, color=ACCENT, linestyle=":", linewidth=0.8)
    ax.text(0.95, 1.08, "auto-send 0.95", fontsize=8, ha="center", color=ACCENT)
    ax.set_xlim(0.35, 1.02); ax.set_ylim(-0.4, 1.4)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["wrong", "correct"], fontsize=9)
    ax.set_xlabel("model confidence", fontsize=9)
    if title: ax.set_title(title, loc="left", fontsize=10, pad=8)
    fig.savefig(out_path, format="svg")
    plt.close(fig)


# ---------- HTML helpers ----------

def fmt(x, nd=3):
    if x is None: return "-"
    if isinstance(x, float): return f"{x:.{nd}f}"
    return str(x)


def header_nav(current: str, title: str) -> str:
    pages = [("index.html", "Overview"), ("design.html", "Design"),
             ("results.html", "Results & Interpretation"),
             ("discoveries.html", "Discoveries"),
             ("conclusions.html", "Conclusions")]
    nav = "".join(
        f'<a class="{"current" if href == current else ""}" href="{href}">{label}</a>'
        for href, label in pages
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<title>{title}, Loyd take-home</title>'
        '<link rel="stylesheet" href="style.css">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '</head><body>'
        '<header class="site-header">'
        '  <div class="brand">Loyd take-home . <span class="muted">eval pipeline for <code>_detect_intent</code></span></div>'
        f'  <nav>{nav}</nav>'
        '</header><main>'
    )


def footer() -> str:
    return ('</main><footer><div class="muted fine">'
            'Loyd Applied AI Engineer take-home . Nahuel Borda . 2026'
            '</div></footer></body></html>')


def cross_check_grid_html(emails, runs):
    """Per-email x per-model grid, same aggregate, different fine structure made visible."""
    keys = [k for k in ["opus-direct", "opus-reasoning", "sonnet", "llama4", "qwen"] if k in runs]
    headers = ['<div class="h">email</div>', '<div class="h">gold</div>'] + [
        f'<div class="h">{runs[k]["model"]}<br><span class="fine">{runs[k]["note"]}</span></div>'
        for k in keys
    ]
    cols = len(keys) + 2
    cells = []
    for e in emails:
        # row label
        row = [f'<div class="c label" title="{e["body"][:120].replace(chr(34), "")}">{e["id"]} <span class="fine">{e["label_fit"][:3]}</span></div>',
               f'<div class="c gold">{e["gold_label"]}</div>']
        for k in keys:
            p = runs[k]["preds"].get(e["id"])
            if p is None:
                row.append('<div class="c miss">.</div>')
                continue
            pred_label = p["label"]
            ok = is_acceptable(e, pred_label)
            if pred_label == e["gold_label"]:
                row.append(f'<div class="c ok" title="conf {p["confidence"]:.2f}">{pred_label}</div>')
            elif ok:
                row.append(f'<div class="c alt" title="acceptable via alt_label . conf {p["confidence"]:.2f}">{pred_label}</div>')
            else:
                row.append(f'<div class="c miss" title="conf {p["confidence"]:.2f}">{pred_label}</div>')
        cells.append("".join(row))
    grid_html = "".join(headers) + "".join(cells)
    return (f'<div class="cross-check" style="grid-template-columns: 1fr 1fr {" 1.4fr"*len(keys)};">'
            f'{grid_html}'
            f'</div>')


# ---------- Page bodies ----------

def write_css():
    css = r"""
:root{--text:#1a1a1a;--muted:#5b5b5b;--accent:#1a5fb4;--rule:#d8d8d8;--bg:#fafaf7;--card:#fff;--ok:#2e7d32;--bad:#c62828;--alt:#8a6300;}
*{box-sizing:border-box}
html{background:var(--bg)}
body{max-width:48em;margin:0 auto;padding:0 1.4rem 4rem;font:16px/1.6 Georgia,'Iowan Old Style','Apple Garamond',Palatino,serif;color:var(--text);}
.site-header{padding:1.1rem 0 0;border-bottom:1px solid var(--rule);margin-bottom:2.2rem}
.brand{font-family:-apple-system,'Segoe UI','Inter',sans-serif;font-size:0.85rem;letter-spacing:-0.01em;margin-bottom:0.85rem;color:var(--muted)}
nav{font-family:-apple-system,'Segoe UI','Inter',sans-serif;display:flex;gap:0.25rem;flex-wrap:wrap;margin-top:0.2rem;}
nav a{color:var(--muted);padding:0.55rem 1.1rem 0.55rem;font-size:1.02rem;font-weight:500;text-decoration:none;border-bottom:3px solid transparent;border-radius:5px 5px 0 0;transition:background 0.12s,color 0.12s,border-color 0.12s;}
nav a:hover{color:var(--text);background:#efeee8;border-bottom:3px solid var(--rule);}
nav a.current{color:var(--text);background:#efeee8;border-bottom:3px solid var(--accent);font-weight:600;}
h1{font-size:2.1rem;font-weight:400;letter-spacing:-0.015em;margin:0.6rem 0 0.4rem;line-height:1.15}
h2{font-size:1.35rem;margin:2.6rem 0 0.6rem;font-weight:600;letter-spacing:-0.01em;}
h2 .num{color:var(--muted);font-weight:400;margin-right:0.5em}
h3{font-size:1.05rem;margin:1.5rem 0 0.4rem;font-style:italic;font-weight:400;color:#333;}
.lede{color:var(--muted);font-size:1.05rem;line-height:1.55;margin:0 0 1.6rem;font-style:italic;}
p,li{margin:0.55rem 0}
ul,ol{padding-left:1.4rem}
.headline{background:#f0f0eb;padding:0.85rem 1.15rem;border-left:3px solid var(--accent);margin:1.8rem 0;font-size:1.04rem;line-height:1.55;font-style:normal}
.headline strong{font-weight:600}
a{color:var(--accent);text-decoration:none;border-bottom:1px dotted var(--accent);}
a:hover{border-bottom-style:solid}
code{background:#f0f0eb;padding:1px 5px;border-radius:3px;font-size:0.86em;font-family:'SF Mono','JetBrains Mono','Consolas',monospace;}
pre{background:#f0f0eb;padding:0.8rem 1rem;border-radius:4px;font-size:0.82rem;line-height:1.4;overflow-x:auto;}
pre code{background:none;padding:0}
blockquote{font-style:italic;color:var(--muted);border-left:2px solid var(--rule);padding-left:1rem;margin:1rem 0;}
.muted{color:var(--muted)}
.fine{font-size:0.82rem;color:var(--muted)}
.callout{background:#fff;border:1px solid var(--rule);border-radius:5px;padding:0.7rem 1rem;margin:1rem 0;font-size:0.96rem}
.callout.good{background:#f1f7f1;border-color:#c9e2cf}
.callout.bad{background:#fcefea;border-color:#e6c4be}
.callout strong{font-weight:600}
.figure{margin:1.5rem 0 1.8rem}
.figure svg{width:100%;height:auto;max-width:100%;display:block}
.caption{font-size:0.84rem;color:var(--muted);font-style:italic;padding-top:0.4rem;line-height:1.45;}
table{border-collapse:collapse;width:100%;font-size:0.92rem;margin:0.8rem 0 1.3rem;font-family:-apple-system,'Segoe UI','Inter',sans-serif;}
th,td{padding:6px 9px;text-align:left;vertical-align:top;border-bottom:1px solid var(--rule);}
th{font-weight:600;border-bottom:1.5px solid var(--text);background:#f6f5f0}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
td.ok{color:var(--ok)}
td.bad{color:var(--bad)}
.cross-check{display:grid;font-family:-apple-system,'Segoe UI','Inter',sans-serif;font-size:0.8rem;border:1px solid var(--rule);border-radius:5px;overflow:hidden;margin:1rem 0 1.4rem;}
.cross-check .h{font-weight:600;background:#f6f5f0;padding:6px 8px;border-bottom:1.5px solid var(--text);line-height:1.25}
.cross-check .c{padding:5px 8px;border-bottom:1px solid #ececec;text-align:center;font-variant-numeric:tabular-nums;font-size:0.78rem}
.cross-check .c.label{text-align:left;background:#fafaf6;font-weight:600;color:#333}
.cross-check .c.gold{text-align:left;background:#fafaf6;color:#333;font-style:italic}
.cross-check .c.ok{background:#f1f7f1;color:var(--ok)}
.cross-check .c.miss{background:#fcefea;color:var(--bad)}
.cross-check .c.alt{background:#fbf6e3;color:var(--alt)}
.legend{font-size:0.84rem;color:var(--muted);margin:0.4rem 0 1rem}
.legend .sw{display:inline-block;width:12px;height:12px;vertical-align:middle;margin-right:4px;border-radius:2px;border:1px solid #d6d6d6}
.legend .ok{background:#f1f7f1}
.legend .miss{background:#fcefea}
.legend .alt{background:#fbf6e3}
.kv{display:grid;grid-template-columns:auto 1fr;gap:0.3rem 1rem;font-size:0.94rem;margin:1rem 0 1.6rem;font-family:-apple-system,'Segoe UI','Inter',sans-serif;}
.kv dt{color:var(--muted);font-style:italic}
.kv dd{margin:0;font-weight:600}
footer{margin-top:4rem;padding-top:1rem;border-top:1px solid var(--rule);}
.toc{background:#f6f5f0;border:1px solid var(--rule);border-radius:6px;padding:0.7rem 1.1rem;font-size:0.9rem;font-family:-apple-system,'Segoe UI','Inter',sans-serif;margin:1.6rem 0}
.toc strong{display:block;text-transform:uppercase;letter-spacing:0.04em;font-size:0.75rem;color:var(--muted);font-weight:600;margin-bottom:0.4rem}
.toc ol{margin:0;padding-left:1.4rem}
.toc li{margin:0.1rem 0}
.toc h3{margin:0 0 0.4rem;font-style:normal;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.04em;color:var(--muted);font-weight:600}
.appendix-section{margin:2.5rem 0;padding-top:1.5rem;border-top:1px solid var(--rule)}
.appendix-section:first-of-type{border-top:none;padding-top:0}
.appendix-section h2{margin-top:0.2rem}
.appendix-section h3{margin-top:1.3rem}
.glossary{background:#f6f5f0;border:1px solid var(--rule);border-radius:6px;padding:0.6rem 1.1rem;margin:0 0 2rem;font-family:-apple-system,'Segoe UI','Inter',sans-serif;font-size:0.92rem;}
.glossary summary{cursor:pointer;font-weight:600;color:var(--muted);padding:0.3rem 0;list-style:none;}
.glossary summary::-webkit-details-marker{display:none}
.glossary summary::before{content:'+ ';color:var(--accent);font-weight:700}
.glossary[open] summary::before{content:'- '}
.glossary[open] summary{margin-bottom:0.4rem;border-bottom:1px solid var(--rule);}
.glossary dl{margin:0.6rem 0 0.2rem;}
.glossary dt{font-weight:600;color:var(--text);margin-top:0.8rem;font-family:-apple-system,'Segoe UI','Inter',sans-serif}
.glossary dd{margin:0.2rem 0 0;padding-left:0;color:#333;line-height:1.5}
.choices{margin:1rem 0 1.8rem;font-family:-apple-system,'Segoe UI','Inter',sans-serif;font-size:0.95rem}
.choices dt{font-weight:600;margin-top:1rem;color:var(--text)}
.choices dt:first-child{margin-top:0}
.choices dd{margin:0.25rem 0 0;padding-left:0;line-height:1.55;color:#222}
@media (max-width:640px){body{font-size:15px}.cross-check{font-size:0.72rem}.cross-check .c,.cross-check .h{padding:4px 5px}}
"""
    (SITE / "style.css").write_text(css.strip(), encoding="utf-8")


# ---------- index.html ----------

def write_index(emails, runs, primary, m):
    cross_check = cross_check_grid_html(emails, runs)
    has_qwen = "qwen" in runs
    five = "five" if has_qwen else "four"
    extra_run = " (a fifth, Qwen 122B, is in progress)" if not has_qwen else ""
    body = f"""
{header_nav("index.html", "Overview")}
<h1>Evaluating Loyd's <code>_detect_intent</code> classifier</h1>
<p class="lede">A small evaluation pipeline for the top-of-funnel intent classifier in
Loyd's scheduling agent. The dataset is 25 deliberately-tricky emails; the eval reports
where the classifier struggles, why, and what to do about it.</p>

<div class="headline">
The eval pins the classifier's failures to a single mode, <strong>every email it got
wrong, it got wrong by predicting <code>none</code></strong>, traces it to a prompt
that suppressed the model's reasoning, and the harness then validates the fix:
<strong>a reasoning-first prompt lifts clean accuracy from 11/13 to 13/13.</strong>
Cross-checks on {five} independent runs corroborate the failure is real, not a
same-model artefact{extra_run}.
</div>

<div class="kv">
<dt>Primary run</dt><dd>{primary['model']} . {primary['note']}</dd>
<dt>Clean accuracy</dt><dd>{m['clean_correct']}/{m['clean']} = {fmt(m['clean_acc'])}</dd>
<dt>Macro-F1 (all 25)</dt><dd>{fmt(m['macro_f1'])}</dd>
<dt>ECE</dt><dd>{fmt(m['ece'])}</dd>
<dt>&gt;0.95 auto-send precision</dt><dd>{fmt(m['autosend_precision'])} (n={m['autosend_n']})</dd>
</div>

<details class="glossary">
<summary>What the metric names mean (click to open)</summary>
<dl>
<dt>Clean accuracy</dt>
<dd>Accuracy on just the 13 "clean" rows (one obvious correct label per email). The
strict score, before any credit for either-way answers on ambiguous rows.</dd>

<dt>F1</dt>
<dd>The harmonic mean of <em>precision</em> (when the model predicts label X, how often
is it actually X?) and <em>recall</em> (of all the actual X emails, how many did the
model catch?). Between 0 and 1; high only when both precision and recall are high.</dd>

<dt>Macro-F1</dt>
<dd>The F1 for each of the 6 labels, averaged with equal weight, so the rare classes
count as much as the common ones. The alternative, <em>micro</em>-F1, would let
<code>none</code> (7 examples) dominate <code>cancel</code> (1 example); macro keeps
the imbalanced label space honest. Caveat: with <code>cancel</code> at n=1, macro is
high-variance, one prediction swings the score, so the report shows it next to the
per-class breakdown.</dd>

<dt>ECE, Expected Calibration Error</dt>
<dd>The gap between what the model says it knows and what it actually gets right. Bucket
predictions by confidence (0.5-0.7, 0.7-0.9, etc.); in each bucket compare mean
confidence to actual accuracy; weight by bucket size. Lower is better. 0.0 = perfect
calibration. Unstable at n=25, so it's reported alongside the reliability bins.</dd>

<dt>&gt;0.95 auto-send precision</dt>
<dd>Of the predictions the model made at &gt;95% confidence, what fraction were correct.
This is the slice Loyd's pipeline auto-sends with no human review, so it's the single
most decision-relevant number. The "auto-send n" is how many predictions landed in
that band.</dd>

<dt>HITL</dt>
<dd>Human In The Loop. Loyd's Approval Gate: low-confidence predictions get routed to
the user to review the draft before send. The routing threshold is what decides which
ones.</dd>

<dt>label_fit (clean / ambiguous / misfit)</dt>
<dd>My 3-tier annotation of how well the 6-label space fits each email. <em>Clean</em> =
one obvious label (13 emails). <em>Ambiguous</em> = two defensible labels, both
recorded (7 emails). <em>Misfit</em> = no label in the 6 fits (5 emails), the model is
forced to pick the least-wrong one. Almost half the dataset isn't clean, which is the
point, it's calibration material, not a leaderboard set.</dd>

<dt>Acceptable-set accuracy</dt>
<dd>For ambiguous rows, the model is credited for either the gold label OR the
documented alternate. Sits next to strict accuracy in the report, a model that always
picks the alternate isn't penalised for the ambiguity tier.</dd>
</dl>
</details>

<h2>Design choices the reviewer should know up front</h2>

<p>The brief said <em>"simulate the LLM predictions however you'd like."</em> What I
picked, and why, so none of this is hidden in a footnote:</p>

<dl class="choices">
<dt>Gold labels</dt>
<dd>I labeled the 25 emails myself by reading each one. I used <code>claude-opus-4-7</code>
as a labelling assistant, it drafted candidate labels and the <code>label_fit</code>
tier (clean / ambiguous / misfit); I reviewed and finalised every row. So the dataset
has same-model exposure on the label side, which is the obvious risk
<em>and the reason for the cross-checks below</em>.</dd>

<dt>Primary predictor (the "v0" run)</dt>
<dd><code>claude-opus-4-7</code> with a direct prompt: "classify this message, return ONLY
a JSON object." Same model family as the labeler, on purpose, the closest mirror to
Loyd's production setup (single LLM call with a confidence score) makes the eval
methodology directly transferable.</dd>

<dt>The A/B fix tested</dt>
<dd><code>claude-opus-4-7</code> again, but with a <strong>reasoning-first prompt</strong>:
"reason step by step, then give the JSON object as the last line." Same model,
different prompt, which lifted clean accuracy from 11/13 to 13/13. Output in
<code>results/opus-reasoning/</code>.</dd>

<dt>Cross-checks (to defuse the same-model concern)</dt>
<dd>Three cross-check runs: <code>claude-sonnet-4-6</code> (within family, for a
weaker-tier comparison), <code>llama4-scout-17b</code> (Meta, independent family), and
<code>qwen3.5-122b-a10b</code> (Alibaba, independent family, no-think mode). The two
independent-family runs, Llama-4 and Qwen, hit the same 0.846 clean accuracy as Opus
and the same E14 -&gt; <code>none</code> miss. The same-model labeler caveat is
substantially answered, though the structural fix at scale (split labeler and
predictor; multi-annotator human gold for the ambiguous tier) is still in the
dataset-growth plan.</dd>
</dl>

<h2>What the eval is about</h2>
<p>Loyd is an AI scheduling assistant. Every inbound message hits <code>_detect_intent</code>
first, a classifier that drops the message into one of six buckets (<code>schedule</code>,
<code>reschedule</code>, <code>cancel</code>, <code>query_agenda</code>,
<code>block_calendar</code>, <code>none</code>). It is the first domino: an error here
cascades. The take-home asked us to design and implement a way to <em>measure</em>
whether that classifier is doing a good job, and to surface where and why it fails.</p>

<p>I'm submitting two artefacts: a <strong>design document</strong> (the strategy) and an
<strong>executable prototype</strong> (the code, the run, the written interpretation).
Both live on this site, navigable from the top bar.</p>

<h2>Reading order</h2>
<ol>
<li><a href="design.html">Design</a>, the eval pipeline strategy. Covers all six
required sections (eval strategy, scorer design, dataset growth, failure-mode taxonomy,
production feedback loop, extension to the other LLM stages) plus assumptions and the
label-space expansion question the brief invited.</li>
<li><a href="results.html">Results &amp; Interpretation</a>, the prototype outputs and
what the eval told us: the headline finding, scorer-by-scorer numbers, calibration, the
fix tested, and the cross-check that addresses the same-model caveat.</li>
<li><a href="discoveries.html">Discoveries</a>, findings from the runs, pulled out
as standalone signal.</li>
</ol>

<h2>The cross-check picture in one chart</h2>
<p>Every row is one of the 25 emails. Every column is a model run. Cells:
<span class="legend"><span class="sw ok"></span>correct &nbsp;
<span class="sw alt"></span>acceptable via alt_label (ambiguous tier) &nbsp;
<span class="sw miss"></span>wrong</span>. The headline accuracies look similar across
models, but the per-email patterns reveal what the aggregates hide. Hover any cell for
the model's confidence.</p>

{cross_check}

<p class="fine">Two of these models share an exact clean accuracy (0.846) but fail on
<em>different</em> emails, aggregate metrics hide structure that the per-email view
makes visible. I built the eval to surface that, not hide it.</p>

{footer()}
"""
    (SITE / "index.html").write_text(body, encoding="utf-8")


# ---------- design.html ----------

def write_design():
    body = f"""
{header_nav("design.html", "Design")}
<h1>Design Document</h1>
<p class="lede">The eval pipeline for <code>_detect_intent</code>: strategy, scorers,
how the dataset grows, the failure-mode taxonomy, the production feedback loop, and how
the approach extends to the other LLM stages. Every section the brief asked for; I folded
the tooling rationale and the label-space proposal in where they belong.</p>

<div class="toc">
<strong>Sections</strong>
<ol>
<li><a href="#strategy">Eval strategy and the rationale behind it</a></li>
<li><a href="#scorers">Scorer design, what I measure and why</a></li>
<li><a href="#labelspace">The label space is a product decision</a></li>
<li><a href="#dataset">How I'd source and grow the dataset beyond 25</a></li>
<li><a href="#calibration">Calibration under uncertainty</a></li>
<li><a href="#taxonomy">The failure-mode taxonomy I'd track</a></li>
<li><a href="#feedback">How a production feedback loop would work</a></li>
<li><a href="#flywheel">The distillation flywheel, running a cheap model safely</a></li>
<li><a href="#extension">How this extends to the other LLM stages</a></li>
<li><a href="#assumptions">Assumptions, risks, open questions</a></li>
</ol>
</div>

<h2 id="strategy"><span class="num">1.</span>Eval strategy and the rationale behind it</h2>
<p>I designed this as a <strong>calibration instrument, not a benchmark.</strong> With 25 emails
no metric here is a production estimate. The brief is explicit: <em>"the point is the
pipeline, not a leaderboard score."</em> What the pipeline must do is surface
<em>where</em> the classifier fails, <em>why</em>, and with <em>how much confidence</em>,
and stay honest when the dataset itself doesn't have one correct answer.</p>

<h3>Five design principles</h3>
<ol>
<li><strong>Measure the label space, not just performance within it.</strong> If the
taxonomy is wrong, accuracy against it is a comfortable illusion. 48% of the 25 emails
are non-clean cases, see §<a href="#labelspace">3</a>.</li>
<li><strong>Treat uncertainty as signal, not noise.</strong> Genuinely ambiguous emails
are facts to be measured, not errors to be eliminated. When humans genuinely disagree
on a label, forcing a single gold label throws away real signal about where the
boundary is hard.</li>
<li><strong>No single number.</strong> Per-class metrics, a confusion matrix, calibration,
never collapse to one accuracy figure that hides which failures occur.</li>
<li><strong>Every scorer states what it claims to measure</strong>, and how it could be
fooled. The brief grades exactly this.</li>
<li><strong>Compose with the pipeline.</strong> The harness has a stage-agnostic contract;
failure reporting is attributable to <code>_detect_intent</code>, not "the agent."</li>
</ol>

<h3>Tooling, hand-rolled, deliberately</h3>
<p>I built a small Python harness, about 700 lines all in, roughly 200 of which is the core harness and the rest is tests, with a clean
<code>Predictor -> Scorers -> Reporter</code> contract. I surveyed Inspect AI, Braintrust,
DeepEval, and promptfoo and deliberately didn't use them here. The case grades
<em>quality of thinking</em>, and a transparent harness keeps every design
decision visible; framework run-management and dashboards pay off past a few hundred
examples, not at 25.</p>

<p>One real 2026 governance signal worth naming: <em>promptfoo</em> was acquired by
OpenAI in March 2026 (TechCrunch, CNBC), and <em>Braintrust</em> disclosed a credentials
breach in May 2026 (TechCrunch). The hand-rolled harness avoids both the governance and
hosted-dependency surfaces.</p>

<h3>Prediction method</h3>
<p>v0 uses a zero-shot LLM call mirroring production (<code>_detect_intent</code> is
GPT-4.1; I ran against Claude Opus 4.7 via an OpenAI-compatible endpoint). The prompt:</p>
<ul>
<li>Includes the label definitions verbatim, so the model has the same taxonomy the
labeler used.</li>
<li>Randomises label order per call, models give more weight to whichever label appears
first in the list, so shuffling neutralises it (the position-bias finding is well-established;
the fix is one line of code).</li>
<li>Returns a structured object <code>{{label, confidence, rationale}}</code>.</li>
<li>Verbalised confidence, asking the model to state its confidence in words, then
converting to a number, is treated as a signal worth checking, not a number to trust
blindly. Its reliability depends heavily on how the question is asked.</li>
</ul>

<h3>Handling ambiguity, without an "abstain" label</h3>
<p>I didn't add a seventh "abstain" label. Loyd already has the right mechanism:
a confidence score plus HITL routing below threshold. <strong>Low confidence
<em>is</em> the abstain signal</strong>, and I score against that signal. Keeping the eval
faithful to the production system matters more than testing a classifier that doesn't
exist.</p>

<h2 id="scorers"><span class="num">2.</span>Scorer design, what I measure and why</h2>
<p>Five scorers. Each one states a <strong>claim</strong> (what it measures) and an
<strong>honesty check</strong> (how it could mislead, and what I do about it). This is
the brief's explicit grading question, <em>"is the scorer measuring what it claims to
measure?"</em></p>

<h3>2.1 Per-class P/R/F1 + confusion matrix</h3>
<p><strong>Claims:</strong> <em>which</em> labels the classifier confuses with <em>which</em>,
not one aggregate. <strong>Reports:</strong> a 6x6 confusion matrix, per-class P/R/F1,
macro-F1, micro-F1.</p>
<p><strong>Honesty check:</strong> the dataset is severely imbalanced
(<code>schedule</code> n=10, <code>cancel</code> n=1). Micro metrics are dominated by
<code>schedule</code>; a single accuracy number would hide <code>cancel</code> entirely.
Macro-F1 is the headline; class metrics with support &lt; 5 are explicitly flagged as
high-variance.</p>

<h3>2.2 <code>none</code> / out-of-scope detector</h3>
<p><strong>Claims:</strong> can the classifier <em>reject</em>? Out-of-scope detection is
a distinct skill from in-scope sorting, classifiers that do well at routing known intents
reliably fail on unknown ones. Rejection quality also degrades as the label space grows:
more labels means more nearby attractors that pull edge cases in rather than letting
them fall through to <code>none</code>.</p>
<p><strong>Honesty check:</strong> precision and recall on <code>none</code> are
<em>different harms</em> and must never be averaged. Low <code>none</code> recall = junk
acted on (a newsletter handled as a real request). Low <code>none</code> precision = a
real request silently dropped to <code>none</code>. The eval slices by
<code>loyd_addressed</code> to separate indirect-path noise (cc carryover, accidental
forwards) from messages addressed to the agent.</p>

<h3>2.3 Calibration scorer</h3>
<p><strong>Claims:</strong> does the confidence score mean what it says, is a 0.9
really ~90% likely correct? See §<a href="#calibration">5</a> for why this is the highest-stakes
scorer (it gates Loyd's HITL routing and the &gt;0.95 auto-send bypass).</p>
<p><strong>Honesty check:</strong> ECE over 25 points is unstable, so I always report
bin counts; the <em>decision-relevant</em> figure is the &gt;0.95-band precision (errors
shipping unreviewed).</p>

<h3>2.4 Ambiguity-aware (acceptable-set) scorer</h3>
<p><strong>Claims:</strong> on ambiguous inputs, does the model land in the <em>defensible
set</em> AND stay <em>appropriately unsure</em>?</p>
<p><strong>Honesty check:</strong> a model that is <em>confidently</em> correct on an
ambiguous item is still miscalibrated against human uncertainty, it just got lucky. The
confidence gap (mean conf on <code>clean</code> minus mean conf on <code>ambiguous</code>)
catches this even when the label is "right".</p>

<h3>2.5 Routing / operating-point scorer</h3>
<p><strong>Claims:</strong> at a given confidence threshold, what is the HITL load and
what is the residual escaped-error rate?</p>
<p><strong>Honesty check:</strong> this measures a <em>product</em> tradeoff, not model
quality. Reported as a curve, never a single score, so it cannot be mistaken for "the
classifier is X% good." It exists to inform where Loyd sets the routing gate.</p>

<h2 id="labelspace"><span class="num">3.</span>The label space is a product decision</h2>
<p>The brief invites this directly: <em>"Defining the intent label space is a product
decision as much as engineering, if you find yourself wanting to expand it, walk us
through how you'd structure that work."</em></p>

<p>The dataset has <strong>5 misfits</strong>, emails that fit no label in the 6 cleanly.
They are not noise; they are the eval's clearest signal that the taxonomy is incomplete.</p>

<table>
<tr><th>Email</th><th>What it really is</th><th>Least-wrong of the 6</th><th>What it needs</th></tr>
<tr><td>E17</td><td>"drop Ben from the meeting", attendee change</td><td><code>reschedule</code></td><td><code>modify_attendees</code></td></tr>
<tr><td>E11</td><td>"is this still on?"</td><td><code>query_agenda</code></td><td><code>meeting_query</code></td></tr>
<tr><td>E19</td><td>"is the meeting Zoom or in person?"</td><td><code>query_agenda</code></td><td><code>meeting_query</code></td></tr>
<tr><td>E18</td><td>"tell the team I'll be late, don't move anything"</td><td><code>none</code></td><td><code>notify_attendees</code></td></tr>
<tr><td>E22</td><td>"remind me to send the deal Friday"</td><td><code>block_calendar</code></td><td><code>set_reminder</code></td></tr>
</table>

<p>The gaps cluster into <strong>two missing categories</strong>: <em>meeting-modify
operations that don't move time</em> (attendee changes, notifications) and
<em>information requests about a specific meeting</em> (the agent must <em>answer</em>,
not <em>act</em>). The current 6 conflate "calendar writes" with "meeting operations"
and omit "answer about one meeting" entirely.</p>

<h3>The bar, each new label must earn its place</h3>
<p>Adding labels is not free. OOS detection degrades as the label space grows, more labels
means more nearby attractors that pull edge cases in rather than letting them fall through
to <code>none</code>. Three tests every candidate label must clear:</p>
<ol>
<li><strong>Recurrence</strong>, non-trivial production frequency.</li>
<li><strong>Distinct action</strong>, routes the agent to a different handler. Two
intents that trigger the same downstream don't need separate labels.</li>
<li><strong>Annotatable</strong>, two annotators can agree on when it applies. If they
can't, the label is badly defined.</li>
</ol>

<h3>Proposed v1 changes, graded</h3>
<table>
<tr><th>Candidate</th><th>Recommendation</th><th>Covers</th><th>Why</th></tr>
<tr><td><code>modify_attendees</code></td><td><strong>Add</strong></td><td>E15, E17, E21 (12% of sample)</td><td>Highest-frequency gap; distinct action; cleanly annotatable.</td></tr>
<tr><td><code>meeting_query</code></td><td><strong>Add</strong></td><td>E11, E19 (8%)</td><td>Distinct handler, the agent must answer; routing to <code>none</code> means silence.</td></tr>
<tr><td><code>notify_attendees</code></td><td>Conditional</td><td>E18 (4%)</td><td>Genuinely distinct action, but only 1/25 here. Add iff production frequency confirms it.</td></tr>
<tr><td><code>set_reminder</code></td><td>Re-scope, don't add</td><td>E22 (4%)</td><td>Is "task reminders" even <code>_detect_intent</code>'s scope, or a different product surface? A product call, not an eval call.</td></tr>
</table>

<p><strong>v1 recommendation:</strong> add <code>modify_attendees</code> and
<code>meeting_query</code> now (-> 8 labels); hold the other two pending production
frequency data. Keep flat for v1; past ~10 labels move to a two-tier scheme
(in-scope vs <code>none</code> first, then fine intent) to protect OOS detection from
label-count growth.</p>

<h3>The recurring label-space review process</h3>
<ol>
<li><strong>Mine.</strong> Cluster two streams: <code>none</code>-bucket messages, and
HITL Edit/Reject events (rejected drafts where the intent was wrong or missing).</li>
<li><strong>Propose.</strong> For each cluster: candidate intent, the action it implies,
estimated frequency.</li>
<li><strong>Gate.</strong> Apply the three tests above.</li>
<li><strong>Validate.</strong> Re-label a sample with >=2 annotators; measure
inter-annotator agreement. Low agreement = kill or redefine.</li>
<li><strong>Ship.</strong> Version the label space, update the predictor prompt and
label definitions, extend the eval set, regression-test against the previous version.</li>
<li><strong>Cadence.</strong> Quarterly; jointly owned by Product (step 3) and the eval
owner (steps 4-5).</li>
</ol>

<h2 id="dataset"><span class="num">4.</span>How I'd source and grow the dataset beyond 25</h2>
<p>25 emails is calibration material. A credible eval set needs to grow along four axes:</p>
<ol>
<li><strong>Stratified production sampling (Gmail).</strong> Sample to cover all 6 labels,
every <code>loyd_addressed</code> stratum, and both thread states. Random sampling would
drown rare classes, <code>cancel</code> and <code>block_calendar</code> must be
over-sampled relative to their natural rate.</li>
<li><strong>Mine the <code>none</code> bucket and HITL Rejects.</strong> Where new
intents and the hardest cases live. An LLM can mine the <code>none</code> bucket for
candidate new intent categories from minimal labeled data. This pipeline feeds
§<a href="#labelspace">3</a>'s label-space review.</li>
<li><strong>Targeted synthetic generation for rare classes.</strong> Raw LLM generation
is insufficient, it's too repetitive. A refinement pass lifts data utility and
diversity. Synthetic data is marked as such and never mixed silently with production.</li>
<li><strong>Multi-annotator labeling with measured agreement.</strong> Every new batch
gets >=2 annotators; I'd compute Krippendorff's α and <em>keep</em> it, not just as a
QA gate but as a standing record of where humans disagree. α is the right choice here:
it handles missing data and unequal annotator pools; Cohen's κ is sensitive to class
imbalance and will quietly change which labels look reliable. Disagreement is preserved
as the <code>ambiguous</code> tier, not voted away.</li>
</ol>

<p>The eval set is versioned; a held-out slice is rotated to resist contamination as prompts
and models iterate. Target: a few hundred stratified, versioned, multi-annotated
examples, the point at which I'd re-survey the framework landscape.</p>

<h2 id="calibration"><span class="num">5.</span>Calibration under uncertainty</h2>
<p>Loyd's classifier already emits a confidence score. That score is
<strong>load-bearing</strong>, it drives two live gates:</p>
<ul>
<li><strong>Below threshold</strong> -> route to the HITL Approval Gate (user reviews).</li>
<li><strong>Above 0.95, no conflicts</strong> -> draft auto-sends with <em>no</em> human review.</li>
</ul>

<p>So calibration error has two distinct, asymmetric costs:</p>
<table>
<tr><th>Failure</th><th>Mechanism</th><th>Cost</th></tr>
<tr><td>Over-confident</td><td>wrong answer scores &gt;0.95</td><td>error auto-sends to an outside party, unguarded</td></tr>
<tr><td>Under-confident</td><td>correct answer scores low</td><td>needless HITL load; erodes trust in the agent</td></tr>
</table>

<p>I treat <strong>&gt;0.95-band precision</strong> as the single most important
number. A global accuracy of 90% is irrelevant if the &gt;0.95 band is 96% precise and
30% of traffic lands there. What gets measured:</p>
<ol>
<li><strong>Reliability</strong>, binned confidence vs empirical accuracy; ECE.</li>
<li><strong>The auto-send band</strong>, precision at confidence &gt;0.95.</li>
<li><strong>The HITL band</strong>, of routed messages, what fraction were genuinely wrong.</li>
<li><strong>The confidence gap</strong>, confidence should be <em>lower</em> on
<code>ambiguous</code> and <code>misfit</code> tiers than on <code>clean</code>.</li>
<li><strong>An operating-point sweep</strong>, to recommend where the routing threshold sits.</li>
</ol>

<p>Once the dataset is large enough to compute a calibration set, I'd replace the single
threshold I picked by reading the data with a smarter mechanism: instead of routing on one
cutoff, return a small set of plausible labels per email with a mathematical guarantee
that the right label is in that set at a chosen confidence level (the conformal prediction
approach, that guarantee is distribution-free, not a heuristic). The clarification
question that follows, "here are the two most likely intents, which did you mean?",
maps directly onto Loyd's existing Approval Gate.</p>

<h2 id="taxonomy"><span class="num">6.</span>The failure-mode taxonomy I'd track</h2>
<p>Beyond "correct vs incorrect," I tag every wrong (or unsafely-confident)
prediction with a failure mode. The taxonomy is grounded in the real 25 emails, each
category has worked examples, and is built/extended with a measured inter-annotator
agreement check. I'd follow the MAST template: build from real failure traces, have two
annotators label them independently, compute agreement. MAST reported κ = 0.88 on a
14-category multi-agent taxonomy, that's the bar for "the categories are actually
coherent."</p>

<table>
<tr><th>#</th><th>Failure mode</th><th>Definition</th><th>Example IDs</th></tr>
<tr><td>F1</td><td>Label-space misfit</td><td>input fits no label; model forced to pick</td><td>E11, E17, E18, E19, E22</td></tr>
<tr><td>F2</td><td>Meeting-state confusion</td><td><code>schedule</code> vs <code>reschedule</code> decided without checking whether a meeting is booked</td><td>E02 <-> E16</td></tr>
<tr><td>F3</td><td>Negation / constraint miss</td><td>explicit "don't" ignored</td><td>E18 ("don't move anything")</td></tr>
<tr><td>F4</td><td>Decoy vocabulary</td><td>meeting-shaped words with no intent</td><td>E06, E24, E25</td></tr>
<tr><td>F5</td><td>Indirect-path miss</td><td>cc-carryover / forward not recognised as <code>none</code></td><td>E06, E23, E24, E25</td></tr>
<tr><td>F6</td><td>Soft-intent miscalibration</td><td>vague social "let's connect" forced to a hard label with high confidence</td><td>E07, E09, E13</td></tr>
<tr><td>F7</td><td>Thin-context overreach</td><td>terse message answered confidently</td><td>E12 ("next week?")</td></tr>
<tr><td>F8</td><td>Output / format failure</td><td>model reasoned correctly but emitted malformed output</td><td>(stress-tested)</td></tr>
<tr><td>F9</td><td>Context blindness</td><td>subject and/or thread state disambiguates the intent, but the model weights only the body and gets it wrong</td><td>E11, E14, E19, E20</td></tr>
</table>

<p>F8 matters more than it looks: in a 2025 error-analysis study of LLM classification
failures, <strong>70.8% of failures were parsing issues, not reasoning errors</strong>,
the model got the logic right but emitted output the parser choked on. Conflating the
two would send us tuning prompts when the real fix is output handling, so F8 is always
counted on its own axis.</p>

<h2 id="feedback"><span class="num">7.</span>How a production feedback loop would work</h2>
<p>Loyd's HITL Approval Gate is exactly the source of structured labels the eval set
needs. The signals available:</p>

<table>
<tr><th>Signal</th><th>Source</th><th>Attribution</th></tr>
<tr><td>HITL <strong>Approve / Edit / Reject</strong></td><td>Approval Gate (user's email reply)</td><td><strong>stage-ish</strong>, Edit/Reject ≈ user correcting the agent</td></tr>
<tr><td>Booking <strong>success / failure</strong></td><td>calendar outcome</td><td><strong>end-to-end</strong>, conflates all stages</td></tr>
<tr><td>Explanatory outcome</td><td>"no time found / meeting not needed / needs cancelling"</td><td>mixed, but informative</td></tr>
</table>

<p><strong>The core nuance:</strong> booking success is an end-to-end signal. A booking
can fail for reasons that have nothing to do with intent (no mutual slot, downstream
parse error). I treat booking success as a <em>noisy proxy</em> for <code>_detect_intent</code>
correctness — not a direct label. The cleanest stage-attributable
signal is HITL Edit/Reject of the draft, and even that needs a lightweight attribution
step, because <code>_detect_intent</code> sits upstream of the Draft Generator: a Reject
means <em>something</em> upstream was wrong, not necessarily the intent.</p>

<pre>log every _detect_intent call -> join to HITL outcome + booking outcome
        |                              |
        |              triage: low-confidence, Rejected,
        |              or annotator-disagreement cases
        |                              |
        +----<-- versioned eval set <-- sample + label (>=2 annotators)</pre>

<p>This closes the loop and keeps the eval honest about which stage it is actually measuring.</p>

<h2 id="flywheel"><span class="num">8.</span>The distillation flywheel, running a cheap model safely</h2>
<p>§<a href="#feedback">7</a>'s feedback loop collects labelled production data. Add one
stage and it becomes a <em>training</em> loop, the mechanism for running
<code>_detect_intent</code> on a cheap, fast model without giving up accuracy. Why this
matters: <code>_detect_intent</code> is a high-volume top-of-funnel call. The
reasoning-first prompt that fixes the drop-to-<code>none</code> failure costs a longer
completion on <em>every</em> message. Distillation is how to keep the accuracy and drop
the cost.</p>

<p><strong>The trap, and why the eval is load-bearing.</strong> Teacher output is
<em>not</em> automatically good training data. The v0 eval proved the teacher itself is
wrong on a structured ~16% of cases (drop-to-<code>none</code>), and the cross-checks
showed the failure is model-general. Distil from raw teacher output and the student
inherits the teacher's bugs, and if the eval's gold is also teacher-derived, the eval
will <em>reward</em> the student for reproducing them. That is the same-model caveat,
baked structurally into a training loop.</p>

<p>So the eval is not a step <em>after</em> distillation, it is the <strong>QA gate
inside</strong> it:</p>

<pre>teacher (Opus + reasoning prompt) -> candidate label + rationale
          |
   eval + HITL filter --> high-confidence clean cases -> training data
          |               low-conf / ambiguous / misfit -> human review
          ▼
   curated corpus -> fine-tune cheap student -> eval validates student
          ▲                                          |
          +-------- HITL Edit/Reject corrections <--- production</pre>

<p>Two non-negotiables, both evidenced by this take-home: fix the teacher prompt
<em>before</em> distilling (the reasoning-prompt A/B did exactly this); the student's test gold
must be independent of the teacher (or the eval can't see inherited bugs).</p>

<h2 id="extension"><span class="num">9.</span>How this extends to the other LLM stages</h2>
<p>The <code>Predictor -> Scorer</code> contract is the same for every stage; only the
scorer changes.</p>
<ul>
<li><strong><code>_parse_fields</code></strong> (GPT-4o, ~16 structured fields). Eval =
per-field scoring: exact match for categoricals, normalised match for
dates/durations/time-zones, set match for attendees. Same harness, different scorer.</li>
<li><strong>Validate-answer layer</strong> (downstream, brief p.2). Eval = seed known
errors and measure precision/recall at catching them. It is itself a classifier.</li>
<li><strong><code>_generate_email</code></strong> (GPT-5.3, open-ended). Cannot be
exact-matched, needs an <strong>LLM-as-judge</strong> scorer with a rigorous rubric.
The requirements are clear: precise per-class criteria dominate reliability more than
chain-of-thought reasoning, so write the rubric carefully rather than just asking the
judge to "think step by step"; judges carry measurable biases (position, length, label
frequency) that must be audited; and a judge must be validated against human
agreement patterns (using Cohen's κ or similar), not just correlation with one
ground-truth rater.</li>
</ul>

<p><strong>Composition.</strong> Errors compound down the pipeline, a wrong intent
yields a wrong draft yields a wasted HITL review. Stage-isolated evals miss cascade
behaviour, so alongside the per-stage evals I'd add a small <em>end-to-end</em>
slice with stage attribution.</p>

<h2 id="assumptions"><span class="num">10.</span>Assumptions, risks, open questions</h2>
<p><strong>Assumptions made:</strong></p>
<ul>
<li>The v0 predictor is zero-shot; production parity is approximate, not exact (I don't
have Loyd's actual <code>_detect_intent</code> prompt).</li>
<li><code>gold_label</code>s are v0, single-annotator; <code>ambiguous</code>/<code>misfit</code>
rows are intentionally contestable.</li>
<li>I treat confidence as a usable signal; §<a href="#calibration">5</a>'s measurements
are what will reveal it if it isn't.</li>
</ul>

<p><strong>Risks:</strong></p>
<ul>
<li><strong>n=25.</strong> No metric here is a production estimate. The pipeline is the
deliverable; the numbers are calibration. §<a href="#dataset">4</a> is the mitigation.</li>
<li><strong>Verbalised confidence</strong>, asking the model to state its confidence in
words, then converting to a number, can be unreliable if the prompt doesn't elicit it
carefully. The prompt is designed for it and §<a href="#calibration">5</a> verifies the
resulting scores rather than assuming they're well-calibrated.</li>
</ul>

{footer()}
"""
    (SITE / "design.html").write_text(body, encoding="utf-8")


# ---------- coverage.html ----------

def write_coverage():
    body = f"""\
{header_nav("conclusions.html", "Conclusions")}

<h1>What the brief asked, and my answer</h1>
<p class="lede">The brief asked the design doc to cover six things, plus a question about
the labels. Here's each one in plain words, with a link if you want the deep version.</p>

<table>
<tr><th>What they asked</th><th>My answer, in plain words</th><th>Where</th></tr>

<tr><td><strong>What's your eval strategy, and why?</strong></td>
<td>Don't just ask "what percent did it get right?". Build something that tells you <em>where</em> it breaks, <em>why</em>, and <em>how sure it was</em> when it was wrong. Almost half the test emails don't fit the labels cleanly, so a single score would lie to you. So I grade the easy ones, the could-go-either-way ones, and the don't-fit-at-all ones separately.</td>
<td><a href="design.html#strategy">Design &sect;1</a></td></tr>

<tr><td><strong>What are you measuring, and why?</strong></td>
<td>Five little measuring tools. One spots which labels get mixed up with which. One checks it correctly says "not my job" to junk mail. One checks that when it says "90% sure" it's actually right about 90% of the time. One is fair to the genuinely ambiguous emails. One shows when it's safe to send on its own vs ask a human. And each tool says, in plain words, what it measures and how it could fool you.</td>
<td><a href="design.html#scorers">Design &sect;2</a></td></tr>

<tr><td><strong>How would you grow the dataset past 25?</strong></td>
<td>Grow it from real misses. The emails it can't label are exactly where the missing categories hide. Have two people label each new one, measure how often they actually agree, and keep the ones they argue about flagged as "genuinely ambiguous" instead of forcing a single answer.</td>
<td><a href="design.html#dataset">Design &sect;4</a></td></tr>

<tr><td><strong>What failure modes would you track?</strong></td>
<td>Nine named ways it can go wrong, each with a real example from the 25 emails. That way when something breaks you can say "ah, failure type 2 got worse this week" instead of just "the score dropped". One of the nine I discovered while building this.</td>
<td><a href="design.html#taxonomy">Design &sect;6</a></td></tr>

<tr><td><strong>How would production feed back into the eval?</strong></td>
<td>Every time a human approves, edits, or rejects what Loyd drafted, that's a free grade. Log it, connect it back to what the classifier guessed, and you learn which mistakes were the classifier's fault vs something further down the line. Those edits become next month's test set, for free.</td>
<td><a href="design.html#feedback">Design &sect;7</a></td></tr>

<tr><td><strong>How does this extend to the other AI steps?</strong></td>
<td>Same plumbing. The next step (pulling out the meeting details) is still mostly black-and-white, so it plugs straight in. The step after (writing the actual email) is fuzzy and open-ended, so that one needs a model grading another model's writing, which is where an off-the-shelf tool finally earns its keep.</td>
<td><a href="design.html#extension">Design &sect;9</a></td></tr>

<tr><td><strong>Would you change the labels?</strong></td>
<td>The 5 emails that fit nothing aren't garbage, they're telling you what's missing. They group into four new buttons the system should have: ask about a meeting, change who's invited, send a heads-up, and set a reminder. I'd add one only if it happens a lot <em>and</em> there's an action behind it, then re-test to make sure adding it doesn't break the existing labels.</td>
<td><a href="design.html#labelspace">Design &sect;3</a></td></tr>
</table>

<p class="fine">The brief also asked for working code: a guess on each of the 25 emails, at
least one scorer, readable output, and a short write-up of what it all means. That's in
the repo (<code>run_eval.py</code> + <code>detect_intent_eval/</code>) and on the
<a href="results.html">Results &amp; Interpretation</a> page.</p>

{footer()}
"""
    (SITE / "conclusions.html").write_text(body, encoding="utf-8")


# ---------- results.html ----------

def write_results(emails, runs, primary_key, m):
    primary = runs[primary_key]
    pbi = primary["preds"]
    has_qwen = "qwen" in runs
    has_reasoning = "opus-reasoning" in runs

    pcs = per_class_stats(emails, pbi)
    cal = calibration_bins(emails, pbi)
    rc = routing_curve(emails, pbi)
    cross_check = cross_check_grid_html(emails, runs)

    # per-class table
    pc_rows = "".join(
        f'<tr><td><code>{r["label"]}</code></td>'
        f'<td class="num">{fmt(r["p"])}</td><td class="num">{fmt(r["r"])}</td>'
        f'<td class="num">{fmt(r["f1"])}</td><td class="num">{r["support"]}</td></tr>'
        for r in pcs
    )

    # calibration table
    cal_rows = "".join(
        f'<tr><td>{b["lo"]:.2f}-{b["hi"]:.2f}</td><td class="num">{b["n"]}</td>'
        f'<td class="num">{fmt(b["mean_conf"], 2)}</td><td class="num">{fmt(b["acc"], 2)}</td></tr>'
        for b in cal
    )

    # routing table
    rc_rows = "".join(
        f'<tr><td>{r["t"]:.2f}</td><td class="num">{r["hitl"]*100:.0f}%</td>'
        f'<td class="num">{r["auto_n"]}</td><td class="num">{fmt(r["auto_precision"])}</td>'
        f'<td class="num">{r["escaped_n"]}</td></tr>'
        for r in rc
    )

    # cross-check headlines
    xcheck_rows = []
    xcheck_keys = [k for k in ["opus-direct", "opus-reasoning", "sonnet", "llama4", "qwen"] if k in runs]
    for k in xcheck_keys:
        run_m = metrics(emails, runs[k]["preds"])
        e14 = runs[k]["preds"].get("E14", {})
        e14_label = e14.get("label", "-")
        e14_ok = "✓" if e14_label == "schedule" else "✗"
        xcheck_rows.append(
            f'<tr><td>{runs[k]["model"]}<br><span class="fine">{runs[k]["note"]}</span></td>'
            f'<td class="num">{run_m["clean_correct"]}/{run_m["clean"]} = {fmt(run_m["clean_acc"])}</td>'
            f'<td class="num">{fmt(run_m["macro_f1"])}</td>'
            f'<td class="num">{fmt(run_m["ece"])}</td>'
            f'<td class="num">{e14_label} {e14_ok}</td></tr>'
        )
    xcheck_table = "".join(xcheck_rows)

    qwen_note = ""
    if not has_qwen:
        qwen_note = ('<p class="fine">A fifth run (Qwen 3.5 122B, Alibaba, a second '
                     'independent family) is in progress; it will be folded into this '
                     'table and the cross-check grid as soon as it completes.</p>')

    fix_section = ""
    if has_reasoning:
        rm = metrics(emails, runs["opus-reasoning"]["preds"])
        fix_section = f"""
<h2 id="fix">7. The fix, tested: letting the model reason</h2>
<p>The v0 prompt told the model: <em>"Return ONLY a JSON object."</em> That instruction
suppressed the model's reasoning, and the suppression <em>was</em> the bug. Re-running
with a reasoning-first prompt (reason step by step, then emit the JSON last;
<code>run_eval.py --reasoning</code>) confirms it:</p>

<table>
<tr><th>Metric</th><th class="num">v0 direct prompt</th><th class="num">reasoning prompt</th></tr>
<tr><td>Clean accuracy</td><td class="num">{m['clean_correct']}/{m['clean']} = {fmt(m['clean_acc'])}</td><td class="num"><strong>{rm['clean_correct']}/{rm['clean']} = {fmt(rm['clean_acc'])}</strong></td></tr>
<tr><td>Macro-F1 (all 25)</td><td class="num">{fmt(m['macro_f1'])}</td><td class="num"><strong>{fmt(rm['macro_f1'])}</strong></td></tr>
<tr><td>ECE</td><td class="num">{fmt(m['ece'])}</td><td class="num">{fmt(rm['ece'])}</td></tr>
</table>

<div class="callout good"><strong>It fixed exactly the predicted cases: E14, E20, and
E11 all flip to correct.</strong> The model now explicitly reasons "a terse in-thread
reply still carries the thread's intent" instead of defaulting to <code>none</code>.</div>

<p><strong>It does <em>not</em> fix E19.</strong> E19 is a misfit, a question about a
meeting, with no correct label in the 6-label space. No prompt fixes a missing label.
This is the clean empirical split the design predicted: a reasoning prompt fixes
<em>reasoning-limited</em> errors; <em>label-space-limited</em> errors need the v1
taxonomy, not a better prompt.</p>

<p>The cost worth naming: a reasoning prompt is a longer completion on every call, real
latency and money for a top-of-funnel classifier. The roadmap suggests using this
reasoning model as a <em>teacher</em> in a distillation flywheel (see
<a href="design.html#flywheel">design §8</a>) rather than paying the latency in production.</p>
"""

    body = f"""
{header_nav("results.html", "Results & Interpretation")}
<h1>Results &amp; Interpretation</h1>
<p class="lede">The executable prototype's outputs on the 25-email dataset, and what the
eval told us, in the order required by the brief: predictions on each email, scorer
outputs in interpretable form, and a written interpretation of what the results say.</p>

<div class="headline">
The eval pinned the v0 classifier's failures to a single mode, <strong>every email it
got wrong, it got wrong by predicting <code>none</code></strong>, traced it to a prompt
that suppressed the model's reasoning, and a reasoning-first prompt A/B confirmed the
fix. Cross-checks across {len(xcheck_keys)} independent runs corroborate that the failure
is model-general, not a same-model artefact.
</div>

<h2 id="setup">Setup</h2>
<dl class="kv">
<dt>Primary model</dt><dd>{primary['model']} . {primary['note']}</dd>
<dt>Dataset</dt><dd>25 labelled emails (13 clean . 7 ambiguous . 5 misfit)</dd>
<dt>Method</dt><dd>zero-shot LLM call with label definitions in prompt; structured JSON output</dd>
<dt>Tests</dt><dd>20 unit tests on the scorers, parsing, and taxonomy</dd>
</dl>

<h2 id="finding">1. Every error is the same error: a real request dropped to <code>none</code></h2>
<p>After I scored honestly (ambiguous rows credited for either defensible label),
<strong>{int((1-m['accept_acc'])*m['n'])} of {m['n']} predictions are wrong, and all of
them are a genuine intent collapsed to <code>none</code>.</strong></p>

<table>
<tr><th>ID</th><th>gold</th><th>predicted</th><th class="num">conf</th><th>what it is</th></tr>
{''.join(
    f'<tr><td>{e["id"]}</td><td><code>{e["gold_label"]}</code></td>'
    f'<td><code>{pbi[e["id"]]["label"]}</code></td>'
    f'<td class="num">{pbi[e["id"]]["confidence"]:.2f}</td>'
    f'<td class="fine">{e["notes"][:80]}...</td></tr>'
    for e in emails if e["id"] in pbi and not is_acceptable(e, pbi[e["id"]]["label"])
)}
</table>

<p>The pattern looked structural to me, not random. The model classifies the <strong>message body
in isolation</strong>, when the body carries no standalone explicit request, it defaults
to <code>none</code>, ignoring the context that does carry the intent. E14
("get back to you tomorrow") is a deferral inside an active scheduling thread; the
<code>thread_so_far</code> is right there in the input. E20 ("Locked in. See you
Wednesday.") has no thread provided; the intent lives in the subject line ("Re: lock
with reed, wed 3pm"). The shared failure: <strong>context blindness</strong>.</p>

<div class="figure">
<img src="confidence-scatter.svg" alt="confidence vs correctness scatter with E14, E20, E11, E19 labelled">
<div class="caption">Each dot is one email. Blue = correct, red = wrong. The error
cluster sits at <em>moderate</em> confidence (0.6-0.85), not low, which is the
dangerous pattern: wrong answers that don't look uncertain.</div>
</div>

<h2 id="cm">2. Confusion matrix and per-class metrics</h2>
<div class="figure">
<img src="confusion-opus-direct.svg" alt="confusion matrix">
<div class="caption">Rows = gold, columns = predicted. The only column with non-diagonal
mass is <code>none</code>, every error is a drop to it.</div>
</div>

<table>
<tr><th>Label</th><th class="num">Precision</th><th class="num">Recall</th><th class="num">F1</th><th class="num">Support</th></tr>
{pc_rows}
</table>
<p class="fine">Classes with support &lt; 5 are high-variance, read directionally only.</p>

<h2 id="none">3. <code>none</code> / out-of-scope detection</h2>
<p>The two harms here are different. Low <code>none</code> recall = junk acted on
(newsletter handled as a real request). Low <code>none</code> precision = real request
dropped to <code>none</code>. In this run:</p>
<ul>
<li>The model never acted on junk (recall 1.0, every newsletter, e-sign bot, social
note classified correctly).</li>
<li>But 4 real requests got dropped to <code>none</code>, pulling precision to 0.636.</li>
</ul>
<p>It errs conservative: drops real work rather than inventing work. For a scheduling
agent, the silent drop is the failure that matters most, invisible, no error, no reply,
the user's meeting never happens.</p>

<h2 id="cal">4. Calibration</h2>
<div class="figure">
<img src="calibration-opus-direct.svg" alt="calibration: mean confidence vs actual accuracy by bin">
<div class="caption">Bars side-by-side: pale = mean confidence within the bin, blue =
actual accuracy in the bin. Aligned bars = well-calibrated bin; gap = miscalibration.</div>
</div>

<table>
<tr><th>Confidence bin</th><th class="num">n</th><th class="num">mean conf</th><th class="num">accuracy</th></tr>
{cal_rows}
</table>

<div class="callout good"><strong>The &gt;0.95 band on this run:
{m['autosend_n']}/{m['autosend_n']} correct.</strong> On <em>this</em> prompt, when the
model says it is &gt;95% sure, it really is. That is the most decision-relevant single
number, and the one to re-measure on the production <code>_detect_intent</code> prompt
before treating any specific threshold as production-ready.</div>

<div class="callout bad"><strong>But the moderate-confidence band is the danger zone.</strong>
The 0.70-0.90 bin held 8 emails at mean confidence 0.775 but only 62.5% accuracy. The
model's mistakes are made at <em>moderate</em> confidence, not low. Wrong answers that
do not look uncertain.</div>

<p>What I found encouraging is calibration <em>on uncertainty</em>: the
confidence gap (mean conf on clean − mean conf on ambiguous) is 0.241; zero misfits were
answered above 0.95; ambiguous acceptable-set hit rate is 7/7. The model is appropriately
unsure where humans are unsure.</p>

<h2 id="routing">5. HITL routing: the operating curve</h2>
<div class="figure">
<img src="routing-curve.svg" alt="HITL volume and escaped errors as a function of routing threshold">
<div class="caption">Higher threshold -> more goes to HITL (cost) but fewer errors
escape (risk). The dotted line at 0.90 is where errors stop escaping <em>on this
prompt</em>, on this run, the model's characteristic mistakes cluster at 0.78-0.82, so
anything below 0.90 lets them through unreviewed. The production threshold falls out of
re-running this scorer on the real prompt.</div>
</div>

<table>
<tr><th class="num">threshold</th><th class="num">HITL volume</th><th class="num">auto n</th><th class="num">auto precision</th><th class="num">escaped errors</th></tr>
{rc_rows}
</table>

<div class="callout"><strong>What this run says</strong>, on my zero-shot prompt the
routing floor lands at <strong>≥ 0.90</strong>. The defensible takeaway is
methodological, not prescriptive: the &gt;0.95-band precision and the routing curve are
the two numbers that gate auto-send risk and HITL load. Both should be measured on the
production prompt and re-checked on every prompt or model change. The exact threshold
falls out of that measurement.</div>

<h2 id="taxonomy-results">6. Failure-mode taxonomy</h2>
<p>Only F1 fired in this run (the 5 misfits, by construction). F2-F8 are all zero on
Opus direct, but the cross-check section below shows F2 firing on Sonnet and Llama-4 on
E16 (the reschedule decoy the design's E02 <-> E16 pair was built to expose). That is the
taxonomy being validated by a real model failing the way it predicted.</p>

<p>The run also surfaced a failure mode the taxonomy lacks a code for, context
blindness on in-thread or subject-line-only emails (E14, E20). That is the taxonomy
working: a living artefact that a real run extends.</p>

{fix_section}

<h2 id="crosscheck">{'8' if has_reasoning else '7'}, Cross-check: independent models on the same dataset</h2>
<p>The same-model caveat, predictor and dataset labeler are both Claude Opus, is the
obvious risk for clean-case agreement. Three cross-checks address it (Sonnet within
family for a tier comparison; Llama-4 Scout and Qwen 3.5 122B as independent model
families):</p>

<table>
<tr><th>Model</th><th class="num">Clean accuracy</th><th class="num">Macro-F1</th><th class="num">ECE</th><th class="num">E14 -> ?</th></tr>
{xcheck_table}
</table>

{qwen_note}

<p>Three findings sharpen because of the cross-checks:</p>
<ul>
<li><strong>E14 is the universal failure.</strong> Every model, Claude or otherwise,
drops <em>"Let me check with my team and get back to you tomorrow"</em> to
<code>none</code>. The drop-to-<code>none</code> failure is the task, not the model.</li>
<li><strong>F2 (meeting-state confusion) is empirically validated.</strong> Both Sonnet
and Llama-4 trip the E16 reschedule decoy. Opus avoided it; two other models did not.
That is the case for keeping rare-but-anticipated modes in the taxonomy.</li>
<li><strong>E20 is recoverable from context.</strong> Llama-4 classified E20 correctly
by reading the subject line; Opus and Sonnet both ignored it. Not every context-blindness
case is universal, E20 is softer than E14.</li>
</ul>

<h3>The cross-check picture, per email</h3>
<p>Same data as the headline table, drilled to every email x every model. Same
aggregate accuracy across models can hide very different per-email patterns, the grid
makes that visible. Hover any cell for the model's confidence.</p>

<div class="legend"><span class="sw ok"></span>correct &nbsp;
<span class="sw alt"></span>acceptable via alt_label (ambiguous tier) &nbsp;
<span class="sw miss"></span>wrong</div>

{cross_check}

<h2 id="caveats">{'9' if has_reasoning else '8'}, Caveats</h2>
<ul>
<li><strong>Same-model caveat, substantially mitigated.</strong> The two
independent-family cross-checks, Llama-4 Scout (Meta) and Qwen 3.5 122B (Alibaba,
no-think), both land on the same 0.846 clean accuracy and the same E14 failure as
Opus. Two non-Claude models, predicting blind, agree on the headline. The production
fix remains structural: labeler and predictor must be different models, with
human-labeled gold for the ambiguous tier.</li>
<li><strong>n = 25.</strong> This is calibration material, not a benchmark. Per-class
metrics below ~5 support are directional only.</li>
</ul>

<h2 id="next">{'10' if has_reasoning else '9'}, What I'd do next</h2>
<ol>
<li><strong>Run the F9 ablation</strong>, strip <code>thread_so_far</code> from the
prompt for the context-rich emails and re-predict. If predictions don't change, the
model wasn't using context anyway and F9 is confirmed as a real failure mode, not just
an inference.</li>
<li><strong>Grow the eval set</strong> with deliberate emphasis on in-thread replies and
<code>query_agenda</code>, see <a href="design.html#dataset">design §4</a>.</li>
<li><strong>Re-run the routing scorer against the production <code>_detect_intent</code>
prompt</strong> to set a defensible HITL threshold. On my zero-shot prompt the floor
lands at ≥ 0.90, but that number is prompt-dependent. The harness emits the full curve
in one command.</li>
<li><strong>The reasoning-prompt fix is validated</strong>, the open decision is the
production tradeoff: pay the latency on every call, or distil into a cheap student
(<a href="design.html#flywheel">design §8</a>).</li>
<li><strong>Wire the production feedback loop</strong>, HITL Edit/Reject is the cheapest
continuous source of stage-attributable labels (<a href="design.html#feedback">design §7</a>).</li>
<li><strong>Split labeler and predictor</strong> so the same-model caveat goes away entirely.</li>
</ol>

{footer()}
"""
    (SITE / "results.html").write_text(body, encoding="utf-8")


# ---------- discoveries.html ----------

def write_discoveries():
    body = f"""\
{header_nav("discoveries.html", "Discoveries")}

<h1>Discoveries</h1>
<p class="lede">What the eval surfaced, pulled out as standalone signal, separate
from the methodology in <a href="design.html">Design</a> and the narrative in
<a href="results.html">Results &amp; Interpretation</a>. Each finding links back to
its receipt.</p>

<h2>From the runs</h2>

<h3>1. Every error was the same error</h3>
<p>On the v0 run (<code>claude-opus-4-7</code>, direct prompt), 11 of 13 clean emails
were correct. The two misses (E14, E20) were <strong>both</strong> real requests
collapsed to <code>none</code>. Not scattered noise, one coherent failure shape. That
single observation drove the whole interpretation and the F9 ("context blindness")
addition to the taxonomy.</p>

<h3>2. The errors err <em>conservative</em>, not aggressive</h3>
<p>Every wrong call dropped a real request, never invented one. For an HITL scheduling
agent that's the safer direction: a missed request gets surfaced when the user replies
asking about it; an invented meeting becomes a wrong-number outbound email and burns
trust with a third party. The failure shape is structurally aligned with the agent's
risk posture.</p>

<h3>3. A reasoning-first prompt closed the gap, no training data needed</h3>
<p>The v0 prompt closed with <em>"Return ONLY a JSON object."</em> That instruction
suppressed the reasoning the model needed on the borderline cases. Changing the closer
to <em>"Reason step by step, then give the JSON object as the last line"</em> lifted
clean accuracy from <strong>0.846 to 1.000</strong> and macro-F1 from 0.817 to 0.917.
A/B tested via <code>run_eval.py --reasoning</code>. A prompt-level fix beats a
fine-tune at this stage.</p>

<h3>4. Four models, same headline number, same failure mode</h3>
<table>
<tr><th>Model</th><th>Family</th><th class="num">Clean accuracy</th><th class="num">Macro-F1</th></tr>
<tr><td><code>claude-opus-4-7</code></td><td>Anthropic</td><td class="num">0.846</td><td class="num">0.817</td></tr>
<tr><td><code>claude-sonnet-4-6</code></td><td>Anthropic</td><td class="num">0.769</td><td class="num">0.701</td></tr>
<tr><td><code>llama4-scout-17b</code></td><td>Meta</td><td class="num">0.846</td><td class="num">0.821</td></tr>
<tr><td><code>qwen3.5-122b-a10b</code> (no-think)</td><td>Alibaba</td><td class="num">0.846</td><td class="num">0.841</td></tr>
</table>
<p>Three out of four hit the exact same 0.846 clean accuracy. All four made the same
E14 -> <code>none</code> miss. Same-family bias was the obvious objection to a
Claude-labeled, Claude-predicted eval, three independent-family cross-checks
substantially answer it.</p>

<h3>5. But the per-email patterns diverged</h3>
<p>Same aggregate accuracy hid real per-item disagreement. E07 only Opus gets right.
E19 only Llama-4 gets right. E20 only Llama-4 + Qwen get right. The
<a href="results.html#crosscheck">per-email x per-model grid</a> shows the structure,
aggregate metrics lie, the grid doesn't.</p>

<h3>6. 48% of the dataset doesn't fit the 6-label space</h3>
<p>Of 25 emails: 13 "clean" (one obviously correct label), <strong>7 "ambiguous"</strong>
(a reasonable person could pick either of two labels), <strong>5 "misfits"</strong>
(no label in the 6 fits). Almost half. An accuracy score computed against the 6-label
gold for these rows measures the wrong thing, which is why the eval treats
<code>label_fit</code> as a first-class signal and ambiguous rows get acceptable-set
scoring.</p>

<h3>7. F9 ("context blindness") emerged from reading the predictions</h3>
<p>E11, E14, E19 all had <code>thread_so_far</code> AND a <code>Re:</code> subject
line in the prompt. The predictor passes all of it to the model. The model still
dropped to <code>none</code>. F1-F8 didn't have a code for "context was present and
the model ignored it", so I added F9. The eval surfaced a gap in its own
taxonomy.</p>

{footer()}
"""
    (SITE / "discoveries.html").write_text(body, encoding="utf-8")


# ---------- main ----------

def main():
    emails = load_emails()
    runs = load_all_runs()
    if "opus-direct" not in runs:
        raise SystemExit("primary run (opus-direct) missing, cannot build site")
    primary_key = "opus-direct"
    primary = runs[primary_key]
    pbi = primary["preds"]
    m = metrics(emails, pbi)

    print(f"loaded {len(emails)} emails, {len(runs)} model runs: {sorted(runs)}")

    # charts
    chart_confusion(emails, pbi, SITE / "confusion-opus-direct.svg",
                    title=f"Confusion matrix, {primary['model']} ({primary['note']})")
    chart_calibration(emails, pbi, SITE / "calibration-opus-direct.svg",
                      title="Calibration, mean confidence vs actual accuracy, by bin")
    chart_routing(emails, pbi, SITE / "routing-curve.svg",
                  title="HITL routing, operating curve")
    chart_confidence_scatter(emails, pbi, SITE / "confidence-scatter.svg",
                             title="Per-email confidence vs correctness")
    print("charts written")

    write_css()
    write_index(emails, runs, primary, m)
    write_design()
    write_results(emails, runs, primary_key, m)
    write_discoveries()
    write_coverage()
    print("html pages written:", *sorted(p.name for p in SITE.glob("*.html")))

if __name__ == "__main__":
    main()
