"""Playbook: suggested responses per cue and competitor battlecards (YAML)."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from sales_copilot.labels import CUE_TYPES


@dataclass(frozen=True)
class Competitor:
    name: str
    aliases: tuple[str, ...]
    battlecard: str


@dataclass(frozen=True)
class Playbook:
    titles: dict[str, str]
    responses: dict[str, tuple[str, ...]]
    competitors: tuple[Competitor, ...] = field(default_factory=tuple)

    def response_for(self, cue_type: str, index: int = 0) -> str:
        options = self.responses.get(cue_type, ())
        if not options:
            return ""
        return options[index % len(options)]

    def title_for(self, cue_type: str) -> str:
        return self.titles.get(cue_type, cue_type.replace("_", " ").title())

    def find_competitors(self, text: str) -> list[Competitor]:
        """Competitors whose name or alias appears as a whole word/phrase in text."""
        import re

        low = text.lower()
        hits: list[Competitor] = []
        for comp in self.competitors:
            for alias in (comp.name.lower(), *comp.aliases):
                if re.search(rf"(?<![\w]){re.escape(alias)}(?![\w])", low):
                    hits.append(comp)
                    break
        return hits


def _parse(raw: dict[str, Any]) -> Playbook:
    cues = raw.get("cues") or {}
    unknown = set(cues) - set(CUE_TYPES)
    if unknown:
        raise ValueError(f"playbook has unknown cue types: {sorted(unknown)}")
    titles = {k: str(v.get("title", k)) for k, v in cues.items()}
    responses = {k: tuple(str(r) for r in v.get("responses", [])) for k, v in cues.items()}
    comps = tuple(
        Competitor(
            name=str(name),
            aliases=tuple(str(a).lower() for a in (spec or {}).get("aliases", [])),
            battlecard=str((spec or {}).get("battlecard", "")),
        )
        for name, spec in (raw.get("competitors") or {}).items()
    )
    return Playbook(titles=titles, responses=responses, competitors=comps)


def load_playbook(path: str | Path | None = None) -> Playbook:
    """Load a playbook YAML; defaults to the packaged playbook."""
    if path is None:
        text = resources.files("sales_copilot").joinpath("data/playbook.yaml").read_text("utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError("playbook must be a YAML mapping")
    return _parse(raw)
