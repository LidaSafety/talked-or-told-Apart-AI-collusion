"""Oracle views: conditioning the bid-bid measures on the R3 signal or on costs. Synthetic runs, plus a structure check on the R3 smoke log (no outcomes read)."""

from pathlib import Path

import numpy as np
import pytest

from sprint_code.cost_response.cost_info import pool_runs
from sprint_code.evaluation.pair_localisation import PAIRS, run_scores
from sprint_code.rfq_logs import Run, load_runs

SMOKE_R3 = (Path(__file__).resolve().parents[1]
            / "results/sprint_runs/smoke/no_talk_R3_shared_information_gemini-2.5-flash.jsonl")


def signal_run(seed, pair, T=300, strength=3):
    """Two bidders add `strength` ticks when a shared signal is high (9-11); no channel, no cost sharing."""
    rng = np.random.default_rng(seed)
    costs, signal = rng.integers(1, 7, (T, 3)), rng.integers(4, 12, T)
    quotes = np.minimum(costs + rng.integers(1, 4, (T, 3)), 12)
    for s in pair:
        quotes[signal >= 9, s] = np.minimum(quotes[signal >= 9, s] + strength, 12)
    return Run(f"sig_{seed}", "synthetic", "no_talk", "none", seed, quotes, costs, True, signal=signal)


def cover_run(seed, pair, T=300):
    """Designated-winner pair (the higher-cost member quotes 11), with an irrelevant signal recorded."""
    rng = np.random.default_rng(seed)
    costs, signal = rng.integers(1, 7, (T, 3)), rng.integers(4, 12, T)
    quotes = np.minimum(costs + rng.integers(1, 4, (T, 3)), 12)
    i, j = pair
    quotes[costs[:, i] > costs[:, j], i] = 11
    quotes[costs[:, j] > costs[:, i], j] = 11
    return Run(f"cov_{seed}", "synthetic", "talk", "none", seed, quotes, costs, True, signal=signal)


def scores(make, view, n=12):
    return [run_scores(make(s, PAIRS[s % 3]), np.random.default_rng(s), K=99, burn_in=20, view=view)
            for s in range(n)]


def share_named(res):
    return float(np.mean([r["returned"]["netinf"] == PAIRS[s % 3] for s, r in enumerate(res)]))


def mean_gamma(res):
    return float(np.mean([r["detail"]["gamma"][PAIRS[s % 3]] for s, r in enumerate(res)]))


def test_shared_signal_fools_public_view_but_not_signal_view_or_cross_cost():
    pub, sig = scores(signal_run, "public"), scores(signal_run, "signal")
    assert share_named(pub) >= 0.5 and share_named(sig) == 0.0
    assert all(r["returned"]["cross_cost"] is None for r in pub + sig)


def test_coordination_survives_conditioning_on_an_irrelevant_signal():
    for view in ("public", "signal"):
        assert share_named(scores(cover_run, view)) == 1.0


def test_cost_conditioning_absorbs_much_of_cost_based_coordination():
    """The oracle cost state removes dependence that runs through costs."""
    assert mean_gamma(scores(cover_run, "oracle")) < 0.5 * mean_gamma(scores(cover_run, "public"))


def test_view_validation():
    plain = Run("plain", "synthetic", "no_talk", "none", 0, np.full((10, 3), 5), np.ones((10, 3), int), True)
    with pytest.raises(ValueError, match="no shared signal"):
        pool_runs([plain], view="signal")
    with pytest.raises(ValueError, match="unknown view"):
        pool_runs([plain], view="nonsense")
    with pytest.raises(ValueError, match="pool them separately"):
        pool_runs([plain, signal_run(0, (0, 1), T=10)], view="oracle")
    assert len(pool_runs([plain], view="costs").state) == 9


@pytest.mark.skipif(not SMOKE_R3.exists(), reason="R3 smoke log not present")
def test_loader_reads_the_r3_signal_and_config():
    run = load_runs(SMOKE_R3)[0]
    assert run.signal is not None and run.signal.shape == (run.T,)
    assert set(run.signal) <= set(range(4, 12))
    assert run.config["signal_seats"] == list(PAIRS[run.run_index % 3])
