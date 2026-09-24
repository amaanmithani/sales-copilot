"""Baseline: hand-written keyword / regex rules.

The rules were written by looking only at the *training* utterances and general sales
vocabulary; the held-out test file was not consulted while writing them.
"""

from __future__ import annotations

import re

from sales_copilot.cues.base import Scores
from sales_copilot.labels import CUE_TYPES
from sales_copilot.playbook import Playbook, load_playbook

_P: dict[str, list[str]] = {
    "objection_price": [
        r"\btoo (expensive|pricey|high|much)\b",
        r"\b(expensive|pricey|steep|overpriced)\b",
        r"\b(no|don'?t have (the|a)?|out of|over|tight|cut)\s*budget",
        r"\bbudget (is|got|was) (tight|cut|gone)",
        r"\bcan'?t (afford|justify)",
        r"\b(price|pricing|cost|number|quote)\b.{0,30}\b(high|steep|much|too|non.?starter)\b",
        r"\b(cheaper|discount)\b.{0,20}\b(need|hoping|make this work)",
        r"\bhoping for something .{0,15}cheaper",
        r"\b(too rich|a lot of money|big investment|no money|kind of money)\b",
        r"\bcost is the\b",
        r"\bpush back .{0,20}price\b",
    ],
    "objection_timing": [
        r"\b(not|isn'?t) (a )?(good|right) (time|moment)\b",
        r"\bnext (quarter|year|fiscal year)\b",
        r"\b(circle|reach out|follow up|check) (back|again)\b",
        r"\bin (a few|six|two|three|a couple( of)?) months\b",
        r"\bnot (a )?priority\b",
        r"\b(right now|at the moment|for now|yet)\b.{0,10}$",
        r"\b(bandwidth|too busy|too much going on|heads down|fires to put out)\b",
        r"\b(later|after|until) (in )?(the )?(year|new year|summer|spring|migration|launch|"
        r"busy season|holidays)\b",
        r"\bpark this\b|\bback burner\b|\bnot in a rush\b",
        r"\btiming\b",
        r"\bq[1-4]\b",
        r"\bwhen things calm down\b",
        r"\bfrozen on\b|\bnothing new gets approved\b",
    ],
    "objection_authority": [
        r"\b(run|check) (this|it) by\b",
        r"\b(my|our) (manager|boss|director|ceo|cfo|vp|co-?founder|leadership)\b",
        r"\b(sign.?off|approve|approval|final (call|say)|buy-?in)\b",
        r"\b(procurement|purchasing|legal|security|committee|board)\b",
        r"\bnot the one who\b|\bsomeone else decides\b|\bnot my budget\b",
        r"\bcan'?t (sign|commit|approve)\b",
        r"\bdecision (sits|is) with\b|\bowns? (that|the) decision\b",
    ],
    "objection_status_quo": [
        r"\b(happy|comfortable|fine) with (what|how|our|the)\b",
        r"\b(works|working) (fine|for us)\b|\bgood enough\b",
        r"\bin-?house\b|\bhomegrown\b|\bour own (tool|version)\b",
        r"\b(reason|need) to switch\b|\bswitching\b",
        r"\balready (have|pay for|use)\b",
        r"\bdon'?t (really )?have a problem\b|\bnot convinced we need\b",
        r"\bisn'?t broken\b|\bwhy fix\b",
        r"\bchange management\b|\bretrain",
        r"\blocked into\b|\bcurrent (vendor|setup|process|system)\b",
        r"\bnobody (used|is asking|on the team)\b|\badoption\b",
        r"\bmigrat(e|ing)\b.{0,30}\b(painful|hassle|nightmare)\b",
    ],
    "pricing_question": [
        r"\bhow much\b",
        r"\bwhat('?s| is| does)? (the )?(pricing|price|cost)\b",
        r"\b(pricing|priced|charge|bill)\b.{0,40}\?",
        r"\bper (seat|user|rep|month)\b",
        r"\b(free trial|discount|quote|setup fee|fees)\b.{0,40}\?",
        r"\bcost\b.{0,20}\?",
        r"\b(rough|ballpark) (number|cost)\b",
        r"\bcheapest plan\b|\bspecial pricing\b|\bmoney-back\b",
        r"\bpay (monthly|quarterly|annually)\b",
    ],
    "next_step": [
        r"\b(set up|schedule|book|put) .{0,30}\b(demo|call|meeting|time|follow-?up|invite|"
        r"calendar)\b",
        r"\b(send|email|forward|share) (me |over |us )?(the |a )?(contract|proposal|quote|invite|"
        r"calendar|docs?|documentation|references|msa|order form|one-pager|recording)",
        r"\bnext steps?\b",
        r"\b(pilot|trial account|proof of concept|poc|onboarding|kickoff)\b",
        r"\b(move forward|ready to sign|sign(ed)? (this|the))\b",
        r"\b(introduce|connect(ed)?|loop in) (you|our|my)\b",
        r"\breconnect\b|\bregroup\b|\bsecond call\b",
        r"\bon (monday|tuesday|wednesday|thursday|friday)\b",
    ],
}


class RuleClassifier:
    """Deterministic regex baseline; competitor detection uses the playbook's alias list."""

    name = "rules"

    def __init__(self, playbook: Playbook | None = None) -> None:
        self.playbook = playbook or load_playbook()
        self._compiled = {k: [re.compile(p, re.IGNORECASE) for p in v] for k, v in _P.items()}

    def labels_for(self, text: str) -> frozenset[str]:
        found = {k for k, pats in self._compiled.items() if any(p.search(text) for p in pats)}
        if self.playbook.find_competitors(text):
            found.add("competitor_mention")
        return frozenset(found)

    def predict_scores(self, texts: list[str]) -> list[Scores]:
        out: list[Scores] = []
        for t in texts:
            labels = self.labels_for(t)
            out.append({c: 1.0 if c in labels else 0.0 for c in CUE_TYPES})
        return out
