#!/usr/bin/env julia
"""
Original POMCPOW vs OneStepLookahead after the model.jl parameter change.

POMCPOW is the original solver (final_results_simulation_copy.jl lines 51-64):
10k queries, depth 25, MaxUCB(1.0), action PW k_action=5, DynamicProfit rollout.
OneStep is OneStepLookaheadPolicy with 30 AlphaAware rollouts.

Compare these numbers with outputs/test_train_final_results/results_p{2,3,4}_e6_new_params.csv,
which used the previous deposits and observation noise.

Planning price models 2, 3, 4 vs evaluation model 6.
Alphas 0.25, 0.5, 0.75, 1.0.
Train seeds 1001+ and test seeds 2001+.

Usage:
  julia --project=. scripts/test_pomcpow_solver_choice.jl [planning_model] [n_sims]
  julia --project=. scripts/test_pomcpow_solver_choice.jl 3 3
"""

include(joinpath(@__DIR__, "pomcpow_tuning_harness.jl"))

planning_model = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3
n_sims         = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
eval_model     = 6
alphas         = [0.25, 0.5, 0.75, 1.0]
train_seeds    = collect(1001:(1000 + n_sims))
test_seeds     = collect(2001:(2000 + n_sims))

configs = [
    "POMCPOW" => p -> make_pomcpow(p),
    "OneStep" => make_onestep,
]

println("SOLVER CHOICE after model.jl parameter change")
println("Original POMCPOW: ", ORIGINAL_POMCPOW)
println("planning=$planning_model  eval=$eval_model  alphas=$alphas")
println("train seeds=$train_seeds")
println("test  seeds=$test_seeds")

run_comparison("solver_choice_train", configs, planning_model, eval_model, alphas, train_seeds)
run_comparison("solver_choice_test",  configs, planning_model, eval_model, alphas, test_seeds)
