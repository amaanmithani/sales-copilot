"""Render README result tables from committed results/*.json.

python scripts/render_readme.py          # rewrite the block between the markers
python scripts/render_readme.py --check  # exit 1 if README is stale (used in CI)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def ci(d: dict[str, float], pct: bool = False) -> str:
    if pct:
        return f"{d['point'] * 100:.1f}% [{d['lo'] * 100:.1f}, {d['hi'] * 100:.1f}]"
    return f"{d['point']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"


def cue_tables(r: dict[str, Any]) -> list[str]:
    systems = list(r["systems"])
    out = [
        f"**Cue classification** (held-out test set, n = {r['n_test']} utterances; "
        f"trained on n = {r['n_train']}; exact-text overlap train/test = "
        f"{r['exact_text_overlap_train_test']}; {r['bootstrap']}).",
        "",
        "| System | Macro-F1 [95% CI] | Micro-F1 [95% CI] | ms / utterance (batched) |",
        "|---|---|---|---|",
    ]
    for s in systems:
        d = r["systems"][s]
        out.append(
            f"| {s} | {ci(d['macro_f1'])} | {ci(d['micro_f1'])} | {d['ms_per_utterance_batch']} |"
        )
    out += ["", "Per cue type, F1 [95% CI] (support in brackets):", ""]
    out.append("| Cue (support) | " + " | ".join(systems) + " |")
    out.append("|---|" + "---|" * len(systems))
    for lab in r["labels"]:
        sup = r["systems"][systems[0]]["per_label"][lab]["support"]
        cells = []
        for s in systems:
            pl = r["systems"][s]["per_label"][lab]
            p_, r_ = pl["precision"]["point"], pl["recall"]["point"]
            cells.append(f"{ci(pl['f1'])}<br>P {p_:.2f} / R {r_:.2f}")
        out.append(f"| {lab} ({sup}) | " + " | ".join(cells) + " |")
    out += ["", "Paired bootstrap, macro-F1 difference vs the rules baseline:", ""]
    out += [
        "| Comparison | Δ macro-F1 [95% CI] | share of resamples where rules >= |",
        "|---|---|---|",
    ]
    for k, v in r["paired_macro_f1_differences"].items():
        out.append(
            f"| {k} | {v['point']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] | "
            f"{v['p_b_better_or_equal']} |"
        )
    return out


def asr_table(r: dict[str, Any]) -> list[str]:
    out = [
        f"**ASR accuracy and speed** ({r['dataset']}, n = {r['n_utterances']} utterances "
        f"= {r['audio_s'] / 60:.1f} min, sample seed {r['sample_seed']}; {r['machine']}).",
        "",
        "| Model | WER [95% CI] | Errors / ref words | RTF (total) | RTF p50 / p95 per utt | "
        "load avg (1 min) start→end |",
        "|---|---|---|---|---|---|",
    ]
    for run in r["runs"]:
        out.append(
            f"| {run['backend']} (beam {run['beam_size']}, {run['cpu_threads']} threads) | "
            f"{ci(run['wer_ci95'], pct=True)} | {run['errors']} / {run['ref_words']} | "
            f"{run['rtf']:.3f} | {run['rtf_p50']:.3f} / {run['rtf_p95']:.3f} | "
            f"{run['loadavg_1m_start']} → {run['loadavg_1m_end']} |"
        )
    return out


def latency_table(r: dict[str, Any]) -> list[str]:
    out = [
        f"**Streaming end-to-end** ({r['audio']}; chunks {r['chunk_ms']} ms; VAD {r['vad']}). "
        f"Latency = {r['latency_definition']}.",
        "",
        "| ASR | Audio | Streaming RTF | Speech-end→cue p50 / p90 / max (cue segments) | "
        "All segments p50 / p90 | Mean breakdown: endpoint / ASR / classify | Call WER | "
        "load avg start→end |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for run in r["runs"]:
        c, a, b = run["e2e_cue_segments"], run["e2e_all_segments"], run["breakdown_mean_ms"]
        out.append(
            f"| {run['backend']} | {run['audio_s']} s, {run['segments']} segments | "
            f"{run['streaming_rtf']:.3f} | {c['p50_ms']:.0f} / {c['p90_ms']:.0f} / "
            f"{c['max_ms']:.0f} ms (n={c['n']}) | {a['p50_ms']:.0f} / {a['p90_ms']:.0f} ms | "
            f"{b['endpoint_wait']:.0f} / {b['asr']:.0f} / {b['classify']:.0f} ms | "
            f"{run['call_wer'] * 100:.1f}% | {run['loadavg_1m_start']} → {run['loadavg_1m_end']} |"
        )
    out += [
        "",
        "Turn-level cue detection on the synthetic calls (same classifier: "
        f"{r['runs'][0]['classifier']}), ASR transcript vs the gold script text:",
        "",
        "| ASR | On ASR text: P / R / F1 (tp, fp, fn) | On gold text: P / R / F1 (tp, fp, fn) |",
        "|---|---|---|",
    ]
    for run in r["runs"]:
        x, g = run["turn_cues_on_asr"], run["turn_cues_on_gold_text"]
        out.append(
            f"| {run['backend']} | {x['precision']:.2f} / {x['recall']:.2f} / {x['f1']:.2f} "
            f"({x['tp']}, {x['fp']}, {x['fn']}) | {g['precision']:.2f} / {g['recall']:.2f} / "
            f"{g['f1']:.2f} ({g['tp']}, {g['fp']}, {g['fn']}) |"
        )
    return out


def render() -> str:
    parts: list[str] = []
    for name, fn in (
        ("cue_eval.json", cue_tables),
        ("asr_wer.json", asr_table),
        ("latency.json", latency_table),
    ):
        p = ROOT / "results" / name
        if p.exists():
            parts += [*fn(json.loads(p.read_text())), ""]
        else:
            parts += [f"_results/{name} not generated yet._", ""]
    return "\n".join(parts).rstrip() + "\n"


def splice(readme: str, body: str) -> str:
    if START not in readme or END not in readme:
        raise SystemExit("README is missing the RESULTS markers")
    head, rest = readme.split(START, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{START}\n{body}{END}{tail}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    path = ROOT / "README.md"
    current = path.read_text()
    new = splice(current, render())
    if args.check:
        if new != current:
            print("README results block is stale; run scripts/render_readme.py", file=sys.stderr)
            return 1
        print("README results block is up to date")
        return 0
    path.write_text(new)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
