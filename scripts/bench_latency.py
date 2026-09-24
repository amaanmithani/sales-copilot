"""End-to-end streaming benchmark on the SYNTHETIC demo calls -> results/latency.json.

For each ASR size, each call is replayed in real time (100 ms chunks released at capture
time) through endpointing -> faster-whisper -> cue classifier. Reports speech-end -> cue
latency, its breakdown, streaming RTF, call WER and turn-level cue detection on ASR text
versus the same classifier on the gold script text.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import numpy as np

from sales_copilot.asr import FasterWhisperTranscriber
from sales_copilot.audio import iter_chunks, read_wav
from sales_copilot.cues.factory import build_classifier
from sales_copilot.engine import CueEngine
from sales_copilot.labels import CUE_TYPES
from sales_copilot.llm import ResponseRewriter
from sales_copilot.metrics import corpus_wer
from sales_copilot.pipeline import Copilot, SegmentEvent
from sales_copilot.playbook import load_playbook

ROOT = Path(__file__).resolve().parents[1]
CALLS = ("call_a", "call_b")


def pct(xs: list[float]) -> dict[str, float]:
    a = np.asarray(xs) * 1000
    return {
        "n": len(xs),
        "p50_ms": round(float(np.median(a)), 1),
        "p90_ms": round(float(np.quantile(a, 0.9)), 1),
        "max_ms": round(float(a.max()), 1),
        "mean_ms": round(float(a.mean()), 1),
    }


def turn_labels(events: list[SegmentEvent], turns: list[dict[str, object]]) -> list[set[str]]:
    """Union of segment labels for each gold turn, assigning a segment to the turn it
    overlaps most in time."""
    out: list[set[str]] = [set() for _ in turns]
    for ev in events:
        best, best_ov = None, 0.0
        for i, t in enumerate(turns):
            ov = min(ev.end_s, float(t["end_s"])) - max(ev.start_s, float(t["start_s"]))  # type: ignore[arg-type]
            if ov > best_ov:
                best, best_ov = i, ov
        if best is not None:
            out[best] |= set(ev.labels)
    return out


def micro(gold: list[set[str]], pred: list[set[str]]) -> dict[str, float]:
    tp = sum(len(g & p) for g, p in zip(gold, pred, strict=True))
    fp = sum(len(p - g) for g, p in zip(gold, pred, strict=True))
    fn = sum(len(g - p) for g, p in zip(gold, pred, strict=True))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["tiny", "base"])
    ap.add_argument("--classifier", default="minilm")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "results/latency.json"))
    args = ap.parse_args()

    playbook = load_playbook()
    clf = build_classifier(args.classifier, playbook)
    metas = {c: json.loads((ROOT / f"data/demo/{c}.segments.json").read_text()) for c in CALLS}
    runs = []
    for size in args.models:
        asr = FasterWhisperTranscriber(size, cpu_threads=args.threads)
        asr.transcribe(np.zeros(16_000, dtype=np.float32))  # warm-up
        all_ev: list[SegmentEvent] = []
        refs, hyps = [], []
        gold_t: list[set[str]] = []
        pred_asr: list[set[str]] = []
        audio_s = asr_s = 0.0
        load0 = os.getloadavg()[0]
        per_call = {}
        for call in CALLS:
            meta = metas[call]
            engine = CueEngine(clf, playbook, rewriter=ResponseRewriter(env={}))
            cop = Copilot(asr, engine, realtime=True)
            events = list(cop.run(iter_chunks(read_wav(ROOT / f"data/audio/{call}.wav"))))
            all_ev += events
            audio_s += cop.stats.audio_s
            asr_s += cop.stats.asr_s
            turns = meta["segments"]
            refs.append(" ".join(t["text"] for t in turns))
            hyps.append(" ".join(e.text for e in events))
            gold_t += [set(t["labels"]) for t in turns]
            pred_asr += turn_labels(events, turns)
            per_call[call] = {
                "audio_s": round(cop.stats.audio_s, 1),
                "segments": len(events),
                "gold_turns": len(turns),
                "cards": sum(len(e.cards) for e in events),
                "transcript": [e.text for e in events],
            }
        gold_text_pred = []
        for call in CALLS:
            texts = [t["text"] for t in metas[call]["segments"]]
            for sc in clf.predict_scores(texts):
                gold_text_pred.append({c for c in CUE_TYPES if sc[c] >= 0.5})
        wer, errs, words = corpus_wer(refs, hyps)
        cue_ev = [e for e in all_ev if e.cards]
        runs.append(
            {
                "model": size,
                "backend": asr.name,
                "classifier": clf.name,
                "audio_s": round(audio_s, 1),
                "streaming_rtf": round(asr_s / audio_s, 4),
                "segments": len(all_ev),
                "e2e_all_segments": pct([e.e2e_s for e in all_ev]),
                "e2e_cue_segments": pct([e.e2e_s for e in cue_ev]),
                "breakdown_mean_ms": {
                    "endpoint_wait": round(
                        float(np.mean([e.endpoint_wait_s for e in all_ev])) * 1000, 1
                    ),
                    "asr": round(float(np.mean([e.asr_s for e in all_ev])) * 1000, 1),
                    "classify": round(float(np.mean([e.classify_s for e in all_ev])) * 1000, 1),
                },
                "call_wer": round(wer, 4),
                "call_wer_errors_words": [errs, words],
                "turn_cues_on_asr": micro(gold_t, pred_asr),
                "turn_cues_on_gold_text": micro(gold_t, gold_text_pred),
                "loadavg_1m_start": round(load0, 2),
                "loadavg_1m_end": round(os.getloadavg()[0], 2),
                "calls": per_call,
            }
        )
        print(json.dumps({k: v for k, v in runs[-1].items() if k != "calls"}))
        del asr
    out = {
        "audio": "SYNTHETIC: macOS `say` TTS, two voices per call, -60 dBFS hiss (see scripts)",
        "calls": list(CALLS),
        "chunk_ms": 100,
        "vad": "energy, -42 dBFS, 450 ms trailing silence",
        "latency_definition": "wall time from capture of the last voiced sample to cue ready "
        "(endpoint wait + queueing + ASR + classify); excludes UI paint",
        "machine": f"{platform.system()} {platform.machine()}, {os.cpu_count()} cores, shared",
        "runs": runs,
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
