from __future__ import annotations

from pathlib import Path

import pytest

from sales_copilot.labels import CUE_TYPES, load_tsv, parse_labels, to_matrix
from sales_copilot.playbook import load_playbook

from .conftest import ROOT


def test_parse_labels() -> None:
    assert parse_labels("none") == frozenset()
    assert parse_labels("next_step; pricing_question") == {"next_step", "pricing_question"}
    with pytest.raises(ValueError):
        parse_labels("objection_weather")


def test_train_and_test_are_disjoint_and_cover_all_labels() -> None:
    train = load_tsv(ROOT / "data/cues/train.tsv")
    test = load_tsv(ROOT / "data/cues/test.tsv")
    assert len(train) >= 200 and len(test) >= 150
    norm = lambda s: s.lower().strip(" .?!,")  # noqa: E731
    assert not {norm(e.text) for e in train} & {norm(e.text) for e in test}
    for split in (train, test):
        for cue in CUE_TYPES:
            assert sum(cue in e.labels for e in split) >= 15, cue
        assert any(not e.labels for e in split)


def test_to_matrix() -> None:
    ex = load_tsv(ROOT / "data/cues/test.tsv")[:3]
    m = to_matrix(ex)
    assert len(m) == 3 and all(len(r) == len(CUE_TYPES) for r in m)


def test_load_tsv_skips_blank(tmp_path: Path) -> None:
    p = tmp_path / "x.tsv"
    p.write_text("labels\ttext\nnone\t\nnext_step\tsend the contract\n")
    ex = load_tsv(p)
    assert len(ex) == 1 and ex[0].labels == {"next_step"}


def test_default_playbook() -> None:
    pb = load_playbook()
    assert set(pb.responses) == set(CUE_TYPES)
    assert pb.response_for("objection_price", 0) != pb.response_for("objection_price", 1)
    assert pb.response_for("objection_price", 2) == pb.response_for("objection_price", 0)
    assert pb.response_for("nonexistent") == ""
    assert pb.title_for("nonexistent_cue") == "Nonexistent Cue"
    assert pb.title_for("next_step") == "Next step detected"


def test_find_competitors_word_boundaries() -> None:
    pb = load_playbook()
    assert [c.name for c in pb.find_competitors("we use Gong and HubSpot")] == ["Gong", "HubSpot"]
    assert [c.name for c in pb.find_competitors("the sales loft team")] == ["Salesloft"]
    assert pb.find_competitors("the gongs rang at hubspots") == []


def test_playbook_validation(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("cues:\n  objection_weather:\n    title: x\n")
    with pytest.raises(ValueError):
        load_playbook(bad)
    notmap = tmp_path / "list.yaml"
    notmap.write_text("- a\n- b\n")
    with pytest.raises(ValueError):
        load_playbook(notmap)
    ok = tmp_path / "ok.yaml"
    ok.write_text(
        "cues:\n  next_step:\n    title: Next\n    responses: [Book it]\n"
        "competitors:\n  Acme:\n    aliases: [acme corp]\n    battlecard: ask\n"
    )
    pb = load_playbook(ok)
    assert pb.response_for("next_step") == "Book it"
    assert pb.find_competitors("we use Acme Corp")[0].name == "Acme"
