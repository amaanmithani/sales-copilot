from __future__ import annotations

from sales_copilot.cues.rules import RuleClassifier
from sales_copilot.engine import CueEngine
from sales_copilot.llm import ResponseRewriter
from sales_copilot.playbook import load_playbook


def make_engine(**kw: object) -> CueEngine:
    pb = load_playbook()
    return CueEngine(RuleClassifier(pb), pb, rewriter=ResponseRewriter(env={}), **kw)  # type: ignore[arg-type]


def test_engine_emits_cards_with_playbook_responses() -> None:
    eng = make_engine()
    cards = eng.on_segment("That's too expensive, and we're also looking at Gong.", 10.0)
    types = {c.cue_type for c in cards}
    assert types == {"objection_price", "competitor_mention"}
    comp = next(c for c in cards if c.cue_type == "competitor_mention")
    assert comp.competitor == "Gong" and comp.battlecard and "Gong" in comp.title
    price = next(c for c in cards if c.cue_type == "objection_price")
    assert price.response == load_playbook().response_for("objection_price", 0)
    assert len(eng.classify_times_s) == 1


def test_engine_cooldown_and_rotation() -> None:
    eng = make_engine(cooldown_s=20.0)
    assert eng.on_segment("It's too expensive.", 5.0)
    assert eng.on_segment("Really too expensive.", 15.0) == []  # within cooldown
    later = eng.on_segment("Still way too expensive.", 30.0)
    assert later and later[0].response == load_playbook().response_for("objection_price", 1)


def test_engine_competitors_have_separate_cooldowns() -> None:
    eng = make_engine()
    assert eng.on_segment("We use Gong.", 1.0)
    cards = eng.on_segment("And Clari too.", 2.0)
    assert [c.competitor for c in cards] == ["Clari"]


def test_engine_short_segment_uses_previous_context() -> None:
    eng = make_engine()
    assert eng.on_segment("", 1.0) == []
    eng.on_segment("Honestly the price is", 1.0)
    cards = eng.on_segment("too high.", 2.0)
    assert [c.cue_type for c in cards] == ["objection_price"]
    assert cards[0].trigger_text == "too high."


def test_engine_competitor_score_without_named_competitor() -> None:
    class Always:
        name = "always"

        def predict_scores(self, texts: list[str]) -> list[dict[str, float]]:
            return [{"competitor_mention": 0.9} for _ in texts]

    pb = load_playbook()
    eng = CueEngine(Always(), pb, rewriter=ResponseRewriter(env={}))
    cards = eng.on_segment("the other vendor", 1.0)
    assert len(cards) == 1 and cards[0].competitor is None


def test_rewriter_off_by_default() -> None:
    calls: list[str] = []

    def poster(url: str, body: dict[str, object], h: dict[str, str], t: float) -> dict[str, object]:
        calls.append(url)
        return {}

    rw = ResponseRewriter(env={}, poster=poster)
    assert not rw.enabled
    assert rw.rewrite("tip", "utt") == "tip" and calls == []
    # URL alone is not enough; the explicit flag is required
    rw2 = ResponseRewriter(env={"SALES_COPILOT_LLM_BASE_URL": "http://x"}, poster=poster)
    assert not rw2.enabled


def test_rewriter_enabled_success_and_fallbacks() -> None:
    env = {
        "SALES_COPILOT_LLM_REWRITE": "1",
        "SALES_COPILOT_LLM_BASE_URL": "http://localhost:9/v1/",
        "SALES_COPILOT_LLM_API_KEY": "k",
    }
    seen: dict[str, object] = {}

    def ok(url: str, body: dict[str, object], h: dict[str, str], t: float) -> dict[str, object]:
        seen.update(url=url, auth=h.get("Authorization"))
        return {"choices": [{"message": {"content": " Better tip. "}}]}

    rw = ResponseRewriter(env=env, poster=ok)
    assert rw.rewrite("tip", "too pricey") == "Better tip."
    assert seen == {"url": "http://localhost:9/v1/chat/completions", "auth": "Bearer k"}
    assert rw.rewrite("", "x") == ""

    def boom(url: str, body: dict[str, object], h: dict[str, str], t: float) -> dict[str, object]:
        raise TimeoutError

    assert ResponseRewriter(env=env, poster=boom).rewrite("tip", "u") == "tip"

    def empty(url: str, body: dict[str, object], h: dict[str, str], t: float) -> dict[str, object]:
        return {"choices": []}

    assert ResponseRewriter(env=env, poster=empty).rewrite("tip", "u") == "tip"
