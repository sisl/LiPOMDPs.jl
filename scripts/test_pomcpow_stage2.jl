#!/usr/bin/env julia
"""
Test 3 (stage 2): build on the stage-1 winner, still changing one thing at a time.

Stage 1 (test_pomcpow_hypertune.jl, α=0.5, p3/e6, 4 seeds) showed the original
POMCPOW stops at 2 mines while OneStep opens 3-4; the only single change that beat
OneStep was k_action=10 (+0.027). Here every config = original + k_action=10, plus
exactly one more change:

  k10                      original + k_action=10          (new base)
  k10+AlphaAware           + leaf rollout matches α-weighted reward
  k10+depth10              + shallower tree (more visits/node)
  k10+queries20k           + double search budget (2x slower)
  k10+alpha_action=0.7     + widen actions faster with visits
  k10+ucb_c=0.3            + less exploration pressure
  k10+k_obs=5              + widen observations more

Default: alphas 0.25,0.5,0.75,1.0 ; planning 3 ; eval 6 ; 4 seeds.

Usage:
  julia --project=. scripts/test_pomcpow_stage2.jl [n_seeds] [alphas] [planning_model] [tag]
  julia --project=. scripts/test_pomcpow_stage2.jl 4 0.5,0.75 3 a
"""

include(joinpath(@__DIR__, "pomcpow_tuning_harness.jl"))

seeds, alphas, planning_model = parse_cli(4, [0.25, 0.5, 0.75, 1.0], 3)
eval_model = 6
tag = length(ARGS) >= 4 ? "_" * ARGS[4] : ""

configs = [
    "k10"                   => p -> make_pomcpow(p; k_action=10.0),
    "k10+AlphaAware"        => p -> make_pomcpow(p; k_action=10.0, rollout=:alpha_aware),
    "k10+depth10"           => p -> make_pomcpow(p; k_action=10.0, max_depth=10),
    "k10+queries20k"        => p -> make_pomcpow(p; k_action=10.0, tree_queries=20000),
    "k10+alpha_action=0.7"  => p -> make_pomcpow(p; k_action=10.0, alpha_action=0.7),
    "k10+ucb_c=0.3"         => p -> make_pomcpow(p; k_action=10.0, ucb_c=0.3),
    "k10+k_obs=5"           => p -> make_pomcpow(p; k_action=10.0, k_observation=5.0),
    "OneStep"               => make_onestep,
]

println("TEST 3: stage-2 POMCPOW tuning from k_action=10 vs OneStepLookahead")
println("Original POMCPOW: ", ORIGINAL_POMCPOW)
println("alphas=$alphas  seeds=$seeds  planning=$planning_model  eval=$eval_model")

run_comparison("stage2" * tag, configs, planning_model, eval_model, alphas, seeds)
