#!/usr/bin/env julia
"""
Test 2: one-at-a-time POMCPOW hyperparameter sweep vs OneStepLookahead.

Every config starts from the original solver (final_results_simulation_copy.jl 51-64)
and changes exactly one setting:

  orig                 10k queries, depth 25, MaxUCB(1.0), PW k=5 α=0.5, obs k=3 α=0.3, DynProfit rollout
  rollout=AlphaAware   leaf value matches the α-weighted reward (what OneStep uses)
  ucb_c=0.3            exploration constant closer to the reward scale (returns ~0.1-0.6)
  ucb_c=0.1            "
  k_action=10          widen to more actions per node
  k_action=20          "
  max_depth=10         shallower tree, more visits per node
  queries=20000        double the search budget (2x slower)

Default: alpha 0.5 ; planning model 3 ; eval model 6 ; 4 seeds.

Usage:
  julia --project=. scripts/test_pomcpow_hypertune.jl [n_seeds] [alphas] [planning_model]
  julia --project=. scripts/test_pomcpow_hypertune.jl 4 0.5,1.0 3
"""

include(joinpath(@__DIR__, "pomcpow_tuning_harness.jl"))

seeds, alphas, planning_model = parse_cli(4, [0.5], 3)
eval_model = 6

configs = [
    "orig"                 => p -> make_pomcpow(p),
    "rollout=AlphaAware"   => p -> make_pomcpow(p; rollout=:alpha_aware),
    "ucb_c=0.3"            => p -> make_pomcpow(p; ucb_c=0.3),
    "ucb_c=0.1"            => p -> make_pomcpow(p; ucb_c=0.1),
    "k_action=10"          => p -> make_pomcpow(p; k_action=10.0),
    "k_action=20"          => p -> make_pomcpow(p; k_action=20.0),
    "max_depth=10"         => p -> make_pomcpow(p; max_depth=10),
    "queries=20000"        => p -> make_pomcpow(p; tree_queries=20000),
    "OneStep"              => make_onestep,
]

println("TEST 2: one-at-a-time POMCPOW hyperparameter sweep vs OneStepLookahead")
println("Original POMCPOW: ", ORIGINAL_POMCPOW)
println("alphas=$alphas  seeds=$seeds  planning=$planning_model  eval=$eval_model")

run_comparison("hypertune", configs, planning_model, eval_model, alphas, seeds)
