import numpy as np
from sprint_code.scripted_controls.bin_state_check import synthetic_run
from sprint_code.detectors.info import Bins, binned
from sprint_code.evaluation.pair_localisation import run_scores, calibrate_gaps

def test_all_scripted_arms_valid_and_truth():
    for arm in ['S0','S1','S2','S3','S4']:
        r, label = synthetic_run(arm, 22222, 60)
        assert r.T == 60 and r.n == 3
        assert np.all(r.quotes >= 1) and np.all(r.quotes <= 12)
        assert np.all(r.costs >= 1) and np.all(r.costs <= 6)
        assert (label is not None) == (arm in ['S3','S4'])

def test_working_api_and_r0_only_calibration():
    bins = Bins(bid=(3,5,7),cost=(3,),state=(3,))
    r0 = [synthetic_run('S0',i+900,60)[0] for i in range(3)]
    r1 = synthetic_run('S3',1000,60)[0]
    rng = np.random.default_rng(1)
    s = [run_scores(r,rng,bins=bins,K=19,burn_in=20) for r in [*r0,r1]]
    calibrate_gaps(s,{r.run_id for r in r0})
    assert 'netinf' in s[-1]['returned'] and 'own_cost' in s[-1]['returned']
