"""Runner seed plan: --first-seed sets the seed of the first new run.
Estimate-only mode: no model call, no file written."""

import sys

from sprint_code.rfq_runner import run as R


def plan(monkeypatch, capsys, tmp_path, *extra):
    argv = ["run", "--arm", "R1_pair_talk", "--runs", "3", "--T", "150", "--out-dir", str(tmp_path),
            "--estimate-only", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    assert R.main() == 0
    return capsys.readouterr().out


def test_first_seed_sets_run_ids_and_pairs(monkeypatch, capsys, tmp_path):
    out = plan(monkeypatch, capsys, tmp_path, "--first-seed", "10")
    for k, pair in ((10, "AC"), (11, "BC"), (12, "AB")):
        assert f"_T150_r{k:03d}: pair {pair}" in out
    assert "_r000" not in out
    assert not any(tmp_path.iterdir())


def test_default_still_starts_at_seed_zero(monkeypatch, capsys, tmp_path):
    assert "_T150_r000: pair AB" in plan(monkeypatch, capsys, tmp_path)
