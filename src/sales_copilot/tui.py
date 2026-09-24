"""Rich terminal UI: live transcript on the left, cue cards on the right."""

from __future__ import annotations

from collections import deque

from rich.console import Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from sales_copilot.engine import CueCard
from sales_copilot.pipeline import SegmentEvent

CUE_STYLE = {
    "objection_price": "bold red",
    "objection_timing": "bold yellow",
    "objection_authority": "bold magenta",
    "objection_status_quo": "bold bright_red",
    "competitor_mention": "bold cyan",
    "pricing_question": "bold green",
    "next_step": "bold blue",
}


class CopilotView:
    def __init__(self, title: str, max_lines: int = 14, max_cards: int = 4) -> None:
        self.title = title
        self.lines: deque[SegmentEvent] = deque(maxlen=max_lines)
        self.cards: deque[tuple[CueCard, float]] = deque(maxlen=max_cards)
        self.n_cards = 0
        self.last_e2e = 0.0

    def update(self, ev: SegmentEvent) -> None:
        self.lines.append(ev)
        self.last_e2e = ev.e2e_s
        for c in ev.cards:
            self.cards.appendleft((c, ev.e2e_s))
            self.n_cards += 1

    def render(self) -> Layout:
        transcript = Text()
        for ev in self.lines:
            transcript.append(f"[{ev.start_s:6.1f}s] ", style="dim")
            transcript.append(ev.text + "\n", style="bold" if ev.cards else "")
        cards = []
        for card, lat in self.cards:
            body = Text()
            body.append(f'"{card.trigger_text}"\n', style="italic dim")
            body.append(card.response)
            if card.battlecard:
                body.append("\n" + card.battlecard, style="cyan")
            cards.append(
                Panel(
                    body,
                    title=Text(card.title, style=CUE_STYLE.get(card.cue_type, "bold")),
                    subtitle=f"score {card.score:.2f} | {lat * 1000:.0f} ms after speech end",
                    border_style=CUE_STYLE.get(card.cue_type, "white").split()[-1],
                )
            )
        footer = Table.grid(expand=True)
        footer.add_row(
            f"segments {len(self.lines)} shown | cues {self.n_cards}",
            f"last speech-end -> cue {self.last_e2e * 1000:.0f} ms",
        )
        layout = Layout()
        layout.split_column(Layout(name="main", ratio=1), Layout(footer, name="foot", size=1))
        layout["main"].split_row(
            Layout(Panel(transcript, title=self.title), ratio=3),
            Layout(
                Panel(Group(*cards) if cards else Text("listening...", style="dim"), title="Cues"),
                ratio=2,
            ),
        )
        return layout
