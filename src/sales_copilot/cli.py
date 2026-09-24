"""Command-line entry point: `sales-copilot replay|mic|classify`."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from pathlib import Path

from sales_copilot.asr import FasterWhisperTranscriber, Transcriber
from sales_copilot.audio import Chunk, iter_chunks, iter_mic_chunks, read_wav
from sales_copilot.cues.factory import CLASSIFIERS, build_classifier
from sales_copilot.engine import CueEngine
from sales_copilot.labels import CUE_TYPES
from sales_copilot.pipeline import Copilot, SegmentEvent
from sales_copilot.playbook import load_playbook


def _event_json(ev: SegmentEvent) -> str:
    d = asdict(ev)
    d["cards"] = [asdict(c) for c in ev.cards]
    return json.dumps(d)


def run_stream(
    chunks: Iterable[Chunk],
    transcriber: Transcriber,
    classifier_kind: str,
    realtime: bool,
    tui: bool,
    events_path: str | None,
    title: str,
) -> Copilot:
    playbook = load_playbook()
    engine = CueEngine(build_classifier(classifier_kind, playbook), playbook)
    copilot = Copilot(transcriber, engine, realtime=realtime)
    sink = open(events_path, "w", encoding="utf-8") if events_path else None  # noqa: SIM115
    try:
        if tui:
            from rich.live import Live

            from sales_copilot.tui import CopilotView

            view = CopilotView(title)
            with Live(view.render(), refresh_per_second=8, screen=False) as live:
                for ev in copilot.run(chunks):
                    view.update(ev)
                    live.update(view.render())
                    if sink:
                        sink.write(_event_json(ev) + "\n")
        else:
            for ev in copilot.run(chunks):
                cues = ", ".join(c.title for c in ev.cards)
                print(f"[{ev.start_s:6.1f}s] {ev.text}" + (f"   >> {cues}" if cues else ""))
                if sink:
                    sink.write(_event_json(ev) + "\n")
    finally:
        if sink:
            sink.close()
    s = copilot.stats
    print(
        f"audio {s.audio_s:.1f}s | ASR RTF {s.rtf:.3f} | segments {len(s.segments)} | "
        f"cues {sum(len(e.cards) for e in s.segments)}",
        file=sys.stderr,
    )
    return copilot


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sales-copilot", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_stream_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--asr", default="tiny", help="faster-whisper size (tiny, base, ...)")
        p.add_argument("--threads", type=int, default=4)
        p.add_argument("--classifier", choices=CLASSIFIERS, default="minilm")
        p.add_argument("--no-tui", action="store_true", help="plain line output")
        p.add_argument("--events", help="write per-segment JSONL events here")

    rp = sub.add_parser("replay", help="replay a WAV file as a simulated live call")
    rp.add_argument("wav")
    rp.add_argument("--fast", action="store_true", help="do not pace to real time")
    add_stream_args(rp)

    mp = sub.add_parser("mic", help="live microphone (needs the `mic` extra)")
    add_stream_args(mp)

    cp = sub.add_parser("classify", help="classify text utterances")
    cp.add_argument("text", nargs="+")
    cp.add_argument("--classifier", choices=CLASSIFIERS, default="rules")

    args = ap.parse_args(argv)
    if args.cmd == "classify":
        clf = build_classifier(args.classifier)
        for t, sc in zip(args.text, clf.predict_scores(list(args.text)), strict=True):
            hits = {c: round(sc[c], 3) for c in CUE_TYPES if sc[c] >= 0.5}
            print(json.dumps({"text": t, "cues": hits}))
        return 0

    asr = FasterWhisperTranscriber(args.asr, cpu_threads=args.threads)
    if args.cmd == "replay":
        chunks: Iterable[Chunk] = iter_chunks(read_wav(args.wav), realtime=not args.fast)
        title = f"Replay: {Path(args.wav).name} ({asr.name})"
        realtime = not args.fast
    else:
        chunks, title, realtime = iter_mic_chunks(), f"Microphone ({asr.name})", True
    run_stream(chunks, asr, args.classifier, realtime, not args.no_tui, args.events, title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
