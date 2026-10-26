"""The bin and run-length selection rules in sprint_code/analysis/scripted_sweep.py. Synthetic power tables."""

from sprint_code.analysis.scripted_sweep import BINS, PS, TS, choose


def table(net: dict, cross: dict, power_by_T: dict | None = None) -> list[dict]:
    """Power rows: net[bins] / cross[bins] at T = 150; power_by_T[T] for the chosen bins at p = 0.6."""
    rows = []
    for b in BINS:
        for T in TS:
            for cell in ("S3a", "S3b"):
                for p in PS:
                    v = net[b] if T == 150 or power_by_T is None else power_by_T[T]
                    if power_by_T is not None and p == 0.6:
                        v = power_by_T[T]
                    rows.append({"bins": b, "T": T, "view": "public", "cell": cell, "p": p,
                                 "netinf_correct": v, "cross_correct": cross[b] if cell == "S3a" else 0.0})
    return rows


def test_fewer_bins_win_within_002():
    d = choose(table({"A": 0.80, "B": 0.70, "C": 0.815}, {"A": 0.50, "B": 0.40, "C": 0.60}))
    assert d["bins_bid_dependence"] == "A"          # C leads by 0.015 < 0.02: A has fewer bins
    assert d["bins_cost_responsiveness"] == "C"     # C leads by 0.10


def test_run_length_rule_and_fallback():
    d = choose(table({"A": 0.9, "B": 0.5, "C": 0.5}, {"A": 0.5, "B": 0.5, "C": 0.5},
                     {60: 0.5, 120: 0.85, 150: 0.9, 240: 0.95}))
    assert d["T_main"] == 120 and not d["T_needs_budget_decision"]
    d = choose(table({"A": 0.9, "B": 0.5, "C": 0.5}, {"A": 0.5, "B": 0.5, "C": 0.5},
                     {60: 0.1, 120: 0.2, 150: 0.3, 240: 0.4}))
    assert d["T_main"] == 150
