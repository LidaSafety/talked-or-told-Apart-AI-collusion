"""Offline check of bid bins and public-state edges with the RFQ detection API.

Uses only no-talk logs and scripted markets (no pair-talk logs). All synthetic outputs are labelled SCRIPTED.

Run from repo root:
 python -m sprint_code.scripted_controls.bin_state_check \
   --r0-folder external/dealer_market/data/runs/rfq --out results/bin_state_check
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sprint_code.detectors.info import Bins, binned, entropy, occupancy
from sprint_code.evaluation.pair_localisation import run_scores, calibrate_gaps, summarise
from sprint_code.rfq_logs import Run, load_runs

# candidate bid bins, plus one scripted-only candidate
BID_EDGES = [(4, 8), (4, 7), (3, 5, 7), (3, 5)]
STATE_EDGES = [(3,), (4,), (8,)]  # the default edge included as a negative check
COST_EDGES = (3,)
T_VALUES = [30, 60, 120, 150, 240, 480]
BNE = (3, 4, 4, 5, 6, 7)  # independently checked with RFQ src/rfq.py, n=3
PAIRS = [(0, 1), (0, 2), (1, 2)]


def write_rows(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def counted(a, n):
    return [int(np.sum(a == k)) for k in range(n)]


def no_talk_selection(r0_runs, k=99):
    """R0 ONLY. Report observational bin occupancy, balanced states, R0 false alarms.

    Different model runs are calibrated separately for the two gap detectors.
    In-run R0 leave-one-out calibration is a diagnostic, not independent holdout.
    """
    rows = []
    for be in BID_EDGES:
        for se in STATE_EDGES:
            bins = Bins(bid=be, cost=COST_EDGES, state=se)
            bne = binned(np.asarray(BNE), be)
            all_bids = np.vstack([r.quotes[1:] for r in r0_runs])
            all_binned = binned(all_bids, be)
            all_state = binned(np.concatenate([r.price_paid[:-1] for r in r0_runs]), se)
            state_counts = counted(all_state, len(se) + 1)
            bid_counts = counted(all_binned.ravel(), len(be) + 1)
            per_model = {}
            for model in sorted({r.model for r in r0_runs}):
                current = [r for r in r0_runs if r.model == model]
                rng = np.random.default_rng(20261023)
                scores = [run_scores(r, rng, bins=bins, K=k, burn_in=0) for r in current]
                calibrate_gaps(scores, {r.run_id for r in current})
                ss = summarise(scores, {r.run_id: None for r in current})
                per_model[model] = {det: ss[det].get('false_pair') for det in ('netinf', 'cross_cost', 'own_cost', 'level')}
            rows.append({
                'bid_edges': str(be), 'state_edges': str(se), 'r0_runs': len(r0_runs),
                'post_first_round_rows': len(all_state),
                'bne_bid_bin_counts_by_cost': str(counted(bne, len(be)+1)),
                'r0_bid_bin_counts': str(bid_counts), 'r0_state_counts': str(state_counts),
                'r0_state_entropy_bits': round(entropy(all_state), 6),
                'r0_bid_entropy_bits': round(entropy(all_binned.ravel()), 6),
                'r0_zero_count_bid_bins': int(sum(v == 0 for v in bid_counts)),
                'r0_zero_count_state_bins': int(sum(v == 0 for v in state_counts)),
                'r0_false_pairs_by_model': json.dumps(per_model, sort_keys=True),
                'selection_data': 'no-talk logs only',
            })
    return rows


def synthetic_run(arm: str, seed: int, T: int) -> tuple[Run, tuple[int, int] | None]:
    """Explicit, non-LLM scripted controls, independent rng streams per run.

    S0: own-cost-only bid; S1: common observable public shock; S2: common private
    pair-visible shock, no cooperation policy; S3: cover-bid strategy that reads
    both pair members' costs; S4: pair quotes 11 regardless of own costs.
    """
    rng = np.random.default_rng(seed)
    costs = rng.integers(1, 7, size=(T, 3))
    quotes = np.empty((T, 3), dtype=int)
    pair = PAIRS[seed % 3] if arm in ('S2', 'S3', 'S4') else None
    bne = np.array(BNE, dtype=int)
    for t, c in enumerate(costs):
        q = bne[c-1].copy()
        if arm == 'S0':
            pass
        elif arm == 'S1':
            # Uniform public shock produces shared price reactions, no coordination.
            q += 2 * int(rng.integers(0, 2))
        elif arm == 'S2':
            # Hidden paired common cause. Participants act separately given signal.
            q[list(pair)] += 3 * int(rng.integers(0, 2))
        elif arm == 'S3':
            i, j = pair
            # Scripted, known-cost-sharing pair coordination; deliberately strong.
            if c[i] > c[j]: q[i] = 11
            elif c[j] > c[i]: q[j] = 11
            else: q[j] = 11
        elif arm == 'S4':
            q[list(pair)] = 11
        else:
            raise ValueError(arm)
        quotes[t] = np.clip(q, 1, 12)
    r = Run(f'scripted_{arm}_T{T}_s{seed}', 'scripted_'+arm, 'scripted', 'scripted', seed, quotes, costs, True)
    return r, (pair if arm in ('S3', 'S4') else None)


def scripted_sweep(bins: Bins, k=99, per_arm=12, calibration=12):
    """Use the chosen bin candidate and calibrate gap thresholds only on scripted no-coordination markets.

    R0 calibration runs are separate from held-out synthetic evaluation runs,
    and are from the same run length. No S2-S4 outcomes influence bin selection.
    """
    rows = []
    for T in T_VALUES:
        calibration_runs = [synthetic_run('S0', 110000+i, T)[0] for i in range(calibration)]
        run_truth = [synthetic_run(arm, 220000 + 1000*idx + j, T)
                     for idx, arm in enumerate(['S0','S1','S2','S3','S4'])
                     for j in range(per_arm)]
        rng = np.random.default_rng(20261023+T)
        scores = [run_scores(run, rng, bins=bins, K=k, burn_in=20) for run in calibration_runs]
        scores.extend(run_scores(run, rng, bins=bins, K=k, burn_in=20) for run, _ in run_truth)
        r0_ids = {r.run_id for r in calibration_runs}
        calibrate_gaps(scores, r0_ids)
        truths = {r.run_id: None for r in calibration_runs}
        truths.update({r.run_id: true for r, true in run_truth})
        summary = summarise(scores[len(calibration_runs):], truths)
        for arm in ['S0','S1','S2','S3','S4']:
            sample = [r for r in scores[len(calibration_runs):] if r['run_id'].startswith(f'scripted_{arm}_')]
            for det in ['mi','cmi','netinf','cross_cost','own_cost','level']:
                for label in ('forced', 'returned'):
                    if label == 'returned' and det in ('mi','cmi'): continue
                    xs = [r[label].get(det) for r in sample]
                    truth = [truths[r['run_id']] for r in sample]
                    correct = sum(t is not None and x == t for x,t in zip(xs,truth))
                    false_pair = sum(t is None and x is not None for x,t in zip(xs,truth))
                    no_pair = sum(x is None for x in xs)
                    rows.append({'T':T,'burn_in':20,'usable_rounds':T-20,
                                 'arm':arm,'detector':det,'decision':label,'n':len(xs),
                                 'correct_pair_fraction':round(correct / len(xs),5),
                                 'false_pair_fraction':round(false_pair / len(xs),5),
                                 'no_pair_fraction':round(no_pair / len(xs),5),
                                 'bid_edges':str(bins.bid), 'cost_edges':str(bins.cost),
                                 'state_edges':str(bins.state),'K':k,
                                 'note':'SCRIPTED ONLY: S2 is hidden common-cause negative; S3/S4 known synthetic pair'})
    return rows


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--r0-folder', type=Path, required=True)
    pa.add_argument('--out', type=Path, default=Path('results/bin_state_check'))
    pa.add_argument('--K', type=int, default=99)
    pa.add_argument('--runs-per-scripted-arm', type=int, default=12)
    pa.add_argument('--calibration-r0-scripted', type=int, default=12)
    pa.add_argument('--bid-edges', type=int, nargs='+', default=[3,5,7])
    pa.add_argument('--state-edges', type=int, nargs='+', default=[3])
    args = pa.parse_args()
    runs = [r for p in sorted(args.r0_folder.glob('no_talk_rfq_n3_*.jsonl')) for r in load_runs(p,T_expected=30)]
    if not runs: raise SystemExit('No complete no-talk rfq_n3 JSONL runs found in --r0-folder')
    args.out.mkdir(parents=True, exist_ok=True)
    print(f'Loaded {len(runs)} no-talk runs; no pair-talk files are accessed')
    a = no_talk_selection(runs,k=args.K)
    write_rows(args.out/'no_talk_bins.csv',a)
    bins = Bins(bid=tuple(args.bid_edges), cost=COST_EDGES, state=tuple(args.state_edges))
    b = scripted_sweep(bins,k=args.K,per_arm=args.runs_per_scripted_arm,calibration=args.calibration_r0_scripted)
    write_rows(args.out/'scripted_pipeline_T_sweep.csv',b)
    (args.out/'MANIFEST.json').write_text(json.dumps({
        'data_used_for_bin_selection':'no-talk logs only; no pair-talk data read',
        'sweep':'synthetic scripted S0-S4; separate S0 gap-rule calibration; uses run_scores/calibrate_gaps/summarise',
        'bins_for_scripted_sweep':vars(bins), 'K':args.K, 'r0_runs':len(runs),
        'scripted_per_arm':args.runs_per_scripted_arm,'scripted_calibration':args.calibration_r0_scripted,
        'notes':['Do not interpret synthetic detection rates as live-model evidence',
                 'no-talk row statistics use earlier no-talk LLM logs; leave-one-out diagnostics are not independent validation',
                 'bins are chosen on no-talk data only']},indent=2))
    print('Saved to',args.out)

if __name__ == '__main__': main()
