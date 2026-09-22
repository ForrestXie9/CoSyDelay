"""Run one V20 search: P10/G10 means 10 initial + 9 regeneration rounds."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path

import expression_rules
import expression_adaptation_lane
from methods.cosydelay.engine.training_protocol import run_p10g10_training as v16_run
from methods.cosydelay.engine.training_protocol import prompt as v16_prompt
from methods.cosydelay.engine.training_protocol import integration as v16_integration
from methods.cosydelay.engine.fit_guard import integration as v18_integration
from methods.cosydelay.engine.fit_guard import fit_timeout as v18_fit_timeout
from methods.cosydelay.engine.fit_guard import run_p10g10_training as v18_run
from methods.cosydelay.engine.retry_protocol import run_p10g10_training as v17_run
from methods.cosydelay.engine.fit_guard.contract import V18_CONTRACT
from methods.cosydelay.engine.fit_guard.policy import V18_POLICY
from methods.cosydelay.engine.accelerated_search import pairwise_population
from methods.cosydelay.engine.accelerated_search import paper_prompt as v20_prompt
from methods.cosydelay.engine.accelerated_search.fitter import accelerated_fit_supervisor_main
from methods.cosydelay.engine.accelerated_search.regeneration import install_regeneration_hooks
from methods.cosydelay.engine.accelerated_search.prompt_audit import audit_v20_prompts
from methods.cosydelay.engine.accelerated_search.llm_seed import install_reproducible_seed_schedule
from methods.cosydelay.engine.accelerated_search.normalization import install_expression_normalization
from methods.cosydelay.engine.accelerated_search.binary_fitness import install_binary_joint_fitness
from methods.cosydelay.engine.accelerated_search.prefit_r4 import install_prefit_r4_gate

V20_CONTRACT = replace(
    V18_CONTRACT,
    schema_version=14,
    method_id="cosydelay_v20_pairwise_simple_restarts_p10g10_100_t1",
    generations=9,
    successful_candidate_budget=100,
    temperature=1.0,
    omitted_sampling_fields_use_provider_defaults=False,
    maximum_coefficients=expression_rules.MAX_PYTHON_AST_NODES,
    maximum_duplicate_examples_in_prompt=0,
    fitness_contract_changed_from_v17=True,
    physical_scoring_contract_changed_from_v17=True,
    prompt_contract_changed_from_v17=True,
    prefit_filter_scope=(
        "parser_grammar_sampled_numerical_legality_unambiguous_coefficient_"
        "roles_exact_R4_time_homogeneity_and_proven_finite_positive_"
        "coefficient_R7_endpoint_rejection"
    ),
)
V20_POLICY = replace(
    V18_POLICY,
    method_id="cosydelay_v20_pairwise_simple_restarts_p10g10_100_t1",
    method_status="v20_pairwise_training_only_p10g10_exactly_100_temperature_1",
    llm_temperature=1.0,
    llm_sampling_defaults=(
        "temperature_1_explicit;_top_p_and_max_tokens_provider_defaults;_"
        "seed_base_plus_intersection_100000_plus_call_index"
    ),
    fitness_definition=(
        "training_mean_nonnegative_approach_r2_plus_strict_joint_pass_binary"
    ),
)

def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--intersection",type=int,choices=range(1,7),default=1)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--data-dir",type=Path,default=None)
    parser.add_argument("--seed-base",type=int,default=None)
    args=parser.parse_args()
    run_contract=(replace(V20_CONTRACT,requested_seed=args.seed_base)
                  if args.seed_base is not None else V20_CONTRACT)
    if args.output.exists(): raise FileExistsError(args.output)
    # The retained V16 API treats ``generations`` as mutation rounds.  V20
    # reports these parent-conditioned LLM calls as regeneration rounds. Nine rounds
    # after the ten-member initialization yields exactly 100 evaluated forms.
    v16_run.GENERATIONS=9
    original_restart_audit=v16_run._restart_composition_audit
    original_init_prompt=v16_prompt.build_init_prompt
    original_mutation_prompt=v16_prompt.build_mutation_prompt
    original_prompt_exclusion_limit=v16_integration.V16_MAX_PROMPT_EXCLUSIONS
    original_guarded_fitter=v18_integration.GuardedCandidateFitter
    original_engine_evolve=v16_run.engine.evolve_universal_lane_expression
    class V20GuardedCandidateFitter(v18_fit_timeout.GuardedCandidateFitter):
        def __init__(self, **kwargs):
            kwargs["supervisor_target"] = accelerated_fit_supervisor_main
            super().__init__(**kwargs)
    original_install_v18=v18_run.install_v18_candidate
    @contextmanager
    def install_v20_candidate(**kwargs):
        with install_regeneration_hooks(
            original_install_v18,
            pairwise_population,
            **kwargs,
        ) as runtime:
            with install_reproducible_seed_schedule(
                expression_adaptation_lane,
                base_seed=run_contract.requested_seed,
                intersection_id=args.intersection,
            ):
                with install_expression_normalization(expression_adaptation_lane):
                    with install_prefit_r4_gate(expression_adaptation_lane):
                        with install_binary_joint_fitness(
                            pairwise_population,
                            runtime,
                        ):
                            yield runtime
    previous=(v18_run.V18_CONTRACT,v18_run.V18_POLICY,v18_run.validate_v18_contract,v18_integration.V18_CONTRACT,v18_integration.V18_POLICY,v18_integration.validate_v18_contract,v18_run.validate_fit_guard_audit,v16_run.V16_POLICY,v16_run._restart_composition_audit,v18_integration.GuardedCandidateFitter,v16_run.engine.evolve_universal_lane_expression,v18_run.install_v18_candidate,v17_run._v17_prompt_audit,os.environ.get("LLM_TEMPERATURE"),expression_rules.MAX_COEFFICIENTS,v16_prompt.build_init_prompt,v16_prompt.build_mutation_prompt,v16_integration.V16_MAX_PROMPT_EXCLUSIONS)
    original_audit=v18_run.validate_fit_guard_audit
    try:
        v18_run.V18_CONTRACT=run_contract
        v18_run.V18_POLICY=V20_POLICY
        v18_run.validate_v18_contract=lambda: None
        v18_integration.V18_CONTRACT=run_contract
        v18_integration.V18_POLICY=V20_POLICY
        v18_integration.validate_v18_contract=lambda: None
        v16_run.V16_POLICY=V20_POLICY
        os.environ["LLM_TEMPERATURE"]="1.0"
        expression_rules.MAX_COEFFICIENTS=expression_rules.MAX_PYTHON_AST_NODES
        v16_prompt.build_init_prompt=v20_prompt.build_init_prompt
        v16_prompt.build_mutation_prompt=v20_prompt.build_regeneration_prompt
        # Keep canonical duplicate rejection as a program-side gate, but do not
        # paste historical formulas or duplicate-check mechanics into the prompt.
        v16_integration.V16_MAX_PROMPT_EXCLUSIONS=0
        v17_run._v17_prompt_audit=audit_v20_prompts
        v18_run.validate_fit_guard_audit=lambda audit: original_audit(audit,expected_candidates=100)
        v16_run._restart_composition_audit=lambda history: original_restart_audit(
            history, population=10, generations=9
        )
        v18_integration.GuardedCandidateFitter=V20GuardedCandidateFitter
        v18_run.install_v18_candidate=install_v20_candidate
        v16_run.engine.evolve_universal_lane_expression=pairwise_population.evolve_universal_lane_expression
        if v16_run.engine.evolve_universal_lane_expression.__module__ != pairwise_population.__name__:
            raise RuntimeError("V20 pairwise evolution engine was not installed")
        argv=["--intersection",str(args.intersection),"--output",str(args.output)]
        if args.data_dir is not None: argv += ["--data-dir",str(args.data_dir)]
        import sys
        old_argv=sys.argv; sys.argv=[sys.argv[0],*argv]
        try: code=v18_run.main()
        finally: sys.argv=old_argv
    finally:
        v18_run.V18_CONTRACT,v18_run.V18_POLICY,v18_run.validate_v18_contract,v18_integration.V18_CONTRACT,v18_integration.V18_POLICY,v18_integration.validate_v18_contract,v18_run.validate_fit_guard_audit,v16_run.V16_POLICY,v16_run._restart_composition_audit,v18_integration.GuardedCandidateFitter,v16_run.engine.evolve_universal_lane_expression,v18_run.install_v18_candidate,v17_run._v17_prompt_audit,old_temperature,expression_rules.MAX_COEFFICIENTS,v16_prompt.build_init_prompt,v16_prompt.build_mutation_prompt,v16_integration.V16_MAX_PROMPT_EXCLUSIONS=previous
        if old_temperature is None:
            os.environ.pop("LLM_TEMPERATURE",None)
        else:
            os.environ["LLM_TEMPERATURE"]=old_temperature
    if code: return int(code)
    result_path=args.output/'result.json'; result=json.loads(result_path.read_text(encoding='utf-8'))
    history=json.loads((args.output/'history.json').read_text(encoding='utf-8'))
    fitted_ids={int(item['candidate_id']) for item in history if isinstance(item,dict) and item.get('candidate_id') is not None}
    if len(fitted_ids) != 100 or int(result.get('candidate_evaluations',-1)) != 100:
        raise RuntimeError(f'V20 independent-candidate budget mismatch: ids={len(fitted_ids)}, result={result.get("candidate_evaluations")}')
    adapters=set()
    for item in history:
        if not isinstance(item,dict):
            continue
        details=item.get('evaluation_details') or {}
        # Current V16/V18 serialization stores supervisor diagnostics in
        # ``evaluation_details.fit``.  Retain the former key as a read-only
        # compatibility fallback for older artifacts.
        fit_diagnostics=details.get('fit') or details.get('fit_diagnostics') or {}
        adapters.add(fit_diagnostics.get('formal_method_adapter'))
    adapters.discard(None)
    if adapters != {'cosydelay_v19_accelerated_equivalent.fitter'}:
        raise RuntimeError(f'V20 accelerated fitter provenance mismatch: {sorted(adapters)}')
    result.update({'method_id':V20_POLICY.method_id,'method_status':V20_POLICY.method_status,'policy':V20_POLICY.to_dict(),'fitness_definition':'training_r2_plus_strict_joint_pass_binary','strict_joint_pass_binary_component':1.0 if bool(result.get('selected_enhanced_physics',{}).get('joint_pass')) else 0.0,'principlewise_fraction_role':'diagnostic_only_not_used_for_fitness','prefit_physical_gates':['R4_exact_time_dimension','R7_proven_finite_zero_green_limit'],'generation_operation_name':'regeneration','expression_normalization':{'caret_power_to_python_power':True,'stage':'after_response_archive_before_symbolic_parse','raw_llm_response_preserved':True},'llm_seed_schedule':{'strategy':'base_plus_intersection_100000_plus_call_index','base_seed':run_contract.requested_seed,'intersection_id':args.intersection,'first_call_seed':run_contract.requested_seed+args.intersection*100000+1},'regeneration_policy':{'checked_attempts':5,'fallback':'initialization_prompt','fallback_attempts':5,'fit_only_after_candidate_checks_pass':True,'canonical_duplicate_scope':'full_evaluated_history','historical_formulas_in_prompt':0},'v20_changes':['exact_R4_time_dimension_prefit_gate','strict_joint_pass_binary_fitness','pairwise_parent_child_selection','parent_conditioned_regeneration','initialization_prompt_fallback_after_rejected_regeneration','full_history_program_side_duplicate_retry_gate','caret_power_normalized_to_python_power_before_parse','distinct_reproducible_llm_seed_per_call','python_power_operator_instruction','program_side_duplicate_gate_without_prompt_history','paper_style_concise_prompt','nine_local_plus_one_sobol_for_all_candidates','restart_parallel_numexpr_precompiled','explicit_llm_temperature_1','prompt_omits_numeric_operating_domain','coefficient_count_not_limited_to_a8'],'method_contract':run_contract.to_dict(),'candidate_evaluations_required':100})
    result['regeneration_policy'].update({
        'checked_attempts': 3,
        'duplicate_retry_mode': 'same_global_regeneration_prompt_with_rejection_reason',
    })
    result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'V20 I{args.intersection} P10/G10=100 complete',flush=True)
    return 0
if __name__=='__main__': raise SystemExit(main())
