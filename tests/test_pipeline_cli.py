from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sales_copilot import cli
from sales_copilot.audio import iter_chunks, write_wav
from sales_copilot.cues.rules import RuleClassifier
from sales_copilot.engine import CueEngine
from sales_copilot.llm import ResponseRewriter
from sales_copilot.pipeline import Copilot, RunStats
from sales_copilot.playbook import load_playbook
from sales_copilot.tui import CopilotView

from .conftest import FakeClock, StubTranscriber

TEXTS = ["How much does it cost per seat?", "Send me a calendar invite for Tuesday."]


def make_copilot(clock: FakeClock, realtime: bool, cost: float) -> Copilot:
    pb = load_playbook()
    engine = CueEngine(RuleClassifier(pb), pb, rewriter=ResponseRewriter(env={}))
    return Copilot(StubTranscriber(TEXTS, clock, cost), engine, clock=clock, realtime=realtime)


def test_realtime_latency_accounting(clock: FakeClock, two_bursts: np.ndarray) -> None:
    cop = make_copilot(clock, realtime=True, cost=0.2)
    events = list(cop.run(iter_chunks(two_bursts, 100, realtime=True, clock=clock)))
    assert [e.text for e in events] == TEXTS
    assert [c.cue_type for c in events[0].cards] == ["pricing_question"]
    assert [c.cue_type for c in events[1].cards] == ["next_step"]
    assert events[0].labels == ("pricing_question",)
    for e in events:
        # e2e = endpoint wait (450 ms of silence, 100 ms chunk quantisation) + ASR (0.2 s)
        assert 0.2 + 0.45 <= e.e2e_s <= 0.2 + 0.45 + 0.1 + 1e-6
        assert e.asr_s == pytest.approx(0.2)
    assert cop.stats.rtf == pytest.approx(0.4 / cop.stats.audio_s)
    assert cop.stats.wall_s >= cop.stats.audio_s


def test_realtime_backlog_is_included(clock: FakeClock, two_bursts: np.ndarray) -> None:
    # ASR slower than the gap between utterances -> second segment waits in the queue
    cop = make_copilot(clock, realtime=True, cost=2.5)
    events = list(cop.run(iter_chunks(two_bursts, 100, realtime=True, clock=clock)))
    assert events[1].e2e_s > 2.5 + 0.45


def test_fast_mode_uses_virtual_speech_end(clock: FakeClock, two_bursts: np.ndarray) -> None:
    cop = make_copilot(clock, realtime=False, cost=0.3)
    seen = []
    events = list(
        cop.run(iter_chunks(two_bursts, 100, realtime=False, clock=clock), on_event=seen.append)
    )
    assert len(seen) == 2
    for e in events:
        assert 0.3 + 0.45 <= e.e2e_s <= 0.3 + 0.45 + 0.1 + 1e-6


def test_flush_at_end_of_stream(clock: FakeClock) -> None:
    from .conftest import silence, tone

    audio = np.concatenate([silence(0.3), tone(0.8)])  # stream ends mid-speech
    cop = make_copilot(clock, realtime=False, cost=0.1)
    events = list(cop.run(iter_chunks(audio, 100, realtime=False, clock=clock)))
    assert len(events) == 1 and events[0].forced


def test_runstats_empty() -> None:
    assert RunStats().rtf == 0.0


def test_tui_render(clock: FakeClock, two_bursts: np.ndarray) -> None:
    from rich.console import Console

    cop = make_copilot(clock, realtime=False, cost=0.1)
    view = CopilotView("test")
    console = Console(record=True, width=120)
    console.print(view.render())
    for e in cop.run(iter_chunks(two_bursts, 100, realtime=False, clock=clock)):
        view.update(e)
    console.print(view.render())
    out = console.export_text()
    assert "Pricing question" in out and "listening" in out and "cues 2" in out


def test_cli_classify(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["classify", "how much is it?", "hello"]) == 0
    lines = [json.loads(x) for x in capsys.readouterr().out.strip().splitlines()]
    assert lines[0]["cues"] == {"pricing_question": 1.0} and lines[1]["cues"] == {}


@pytest.mark.parametrize("tui", [False, True])
def test_cli_replay_with_stub_asr(
    tmp_path: Path,
    two_bursts: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tui: bool,
) -> None:
    wav = tmp_path / "call.wav"
    write_wav(wav, two_bursts)
    monkeypatch.setattr(cli, "FasterWhisperTranscriber", lambda *a, **k: StubTranscriber(TEXTS))
    events = tmp_path / "ev.jsonl"
    args = ["replay", str(wav), "--fast", "--classifier", "rules", "--events", str(events)]
    if not tui:
        args.append("--no-tui")
    assert cli.main(args) == 0
    rows = [json.loads(x) for x in events.read_text().splitlines()]
    assert [r["text"] for r in rows] == TEXTS
    assert rows[0]["cards"][0]["cue_type"] == "pricing_question"
    captured = capsys.readouterr()
    assert "ASR RTF" in captured.err
    if not tui:
        assert ">> Pricing question" in captured.out
