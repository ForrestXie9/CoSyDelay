"""One real Training-only guarded fit proving the spawn overlay is usable."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from constants import INTERSECTION_CONFIGS
from data_processing import load_dataset_flexible, preprocess_data_flexible
from optimization_lane import prepare_optimization_context
from methods.cosydelay_v9_clean_from_scratch.run_formal_training_search import DEFAULT_DATA_DIR, lanes_for
from methods.cosydelay_v18_fit_timeout_guard.fit_timeout import GuardedCandidateFitter
from .fitter import accelerated_fit_supervisor_main

def main():
    iid=1; train=preprocess_data_flexible(load_dataset_flexible(str(DEFAULT_DATA_DIR/'Intersection_1_Train.jsonl'),iid),iid).reset_index(drop=True)
    cfg=INTERSECTION_CONFIGS[iid]; lanes,mapping=lanes_for(cfg); targets={a:train[f'Delay_{a}'].reset_index(drop=True) for a in cfg['approaches']}; prepared=prepare_optimization_context(train,lanes,mapping,iid); diagnostics={}
    static={'df':train,'lanes':lanes,'lane_to_approach':mapping,'approach_targets':targets,'intersection_id':iid,'prepared_context':prepared}
    with GuardedCandidateFitter(static_fit_kwargs=static,parallel_workers=4,maxiter=200,maxfun=20000,fit_timeout_seconds=100,startup_timeout_seconds=30,close_timeout_seconds=5,supervisor_target=accelerated_fit_supervisor_main) as fitter:
        params=fitter.fit(universal_expr='Cycle_Time*(a1+a2*flow_lane/GR_phase)',rng=np.random.default_rng(20260816),diagnostics=diagnostics,**static)
        audit=fitter.audit
    expected=['official_local_replay']*9+['sobol_role_wide']
    observed=[[restart['start_kind'] for restart in item['restarts']] for item in diagnostics.get('approaches',[])]
    if diagnostics.get('parallelization') != 'independent_approach_restart_tasks' or not diagnostics.get('numexpr_precompiled') or diagnostics.get('parent_warm_start_available') or any(kinds != expected for kinds in observed):
        raise RuntimeError(f'accelerated overlay not active: {diagnostics}')
    print(json.dumps({'status':'pass','lanes':len(params),'parallelization':diagnostics['parallelization'],'numexpr_precompiled':diagnostics['numexpr_precompiled'],'restart_policy':diagnostics['restart_policy'],'fit_events':len(audit)},indent=2))
    return 0
if __name__=='__main__': raise SystemExit(main())
