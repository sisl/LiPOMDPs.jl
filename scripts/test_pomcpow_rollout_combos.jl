#!/usr/bin/env julia
"""
Test 1: rollout policy × action widening, all four POMCPOW combinations vs OneStepLookahead.

POMCPOW keeps the original solver settings (10k queries, depth 25, MaxUCB(1.0),
obs widening k=3 α=0.3). Only two things vary:
  rollout:          DynamicProfitMaximizer (original)  |  AlphaAware (same as OneStep)
  enable_action_pw: true (original, k=5 α=0.5)         |  false (full action set)

Default: alphas 0.25, 0.5, 0.75, 1.0 ; planning model 3 ; eval model 6 ; 3 seeds.

Usage:
  julia --project=. scripts/test_pomcpow_rollout_combos.jl [n_seeds] [alphas] [planning_model]
  julia --project=. scripts/test_pomcpow_rollout_combos.jl 3 0.25,0.5,0.75,1.0 3
"""

include(joinpath(@__DIR__, "pomcpow_tuning_harness.jl"))

seeds, alphas, planning_model = parse_cli(3, [0.25, 0.5, 0.75, 1.0], 3)
eval_model = 6

configs = [
    "POMCPOW DynProfit+PW (orig)" => p -> make_pomcpow(p; rollout=:dynamic_profit, enable_action_pw=true),
    "POMCPOW AlphaAware+PW"       => p -> make_pomcpow(p; rollout=:alpha_aware,    enable_action_pw=true),
    "POMCPOW DynProfit+AllActs"   => p -> make_pomcpow(p; rollout=:dynamic_profit, enable_action_pw=false),
    "POMCPOW AlphaAware+AllActs"  => p -> make_pomcpow(p; rollout=:alpha_aware,    enable_action_pw=false),
    "OneStep"                     => make_onestep,
]

println("TEST 1: POMCPOW rollout × action-widening combinations vs OneStepLookahead")
println("Original POMCPOW: ", ORIGINAL_POMCPOW)
println("alphas=$alphas  seeds=$seeds  planning=$planning_model  eval=$eval_model")

run_comparison("rollout_combos", configs, planning_model, eval_model, alphas, seeds)
