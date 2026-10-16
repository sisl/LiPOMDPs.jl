#!/usr/bin/env julia
"""
Which lever makes POMCPOW beat OneStepLookahead: solver settings or model.jl parameters?

One process = one (planning price model, alpha) cell. Inside it, five experiments.
Every experiment has its own OneStep baseline evaluated under the same model, so the
column to read is "Δ vs OneStep".

  A  base model          POMCPOW orig | k_action=10 | heuristic proposals | heuristic+k10 | OneStep
  B  exploration cost    x50 (DLE 60M, HR 20M) -> exploring becomes a real decision
  C  exploration noise   halved (σ_dle 10k, σ_hr 45k) -> one reading nearly pins the deposit
  D  reward_scale        1e9 instead of 1e10 (10x larger rewards inside the UCB rule)
  E  emission scale      10_000 instead of 1000 (emissions matter 10x more at low alpha)

B-E run POMCPOW orig vs OneStep only (model levers), A runs the solver levers.
"heuristic proposals" replaces POMCPOW's uniform-random action proposer with one that
tries the AlphaAware heuristic action first, then do-nothing, then random (see harness).

Usage:
  julia --project=. scripts/test_pomcpow_levers.jl <planning_model> <alpha> [n_seeds] [experiments]
  julia --project=. scripts/test_pomcpow_levers.jl 3 0.5 4 ABCDE
Seeds are test seeds 2001.. ; evaluation model is 6.
"""

include(joinpath(@__DIR__, "pomcpow_tuning_harness.jl"))

planning_model = parse(Int, ARGS[1])
α              = parse(Float64, ARGS[2])
n_seeds        = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 4
experiments    = length(ARGS) >= 4 ? ARGS[4] : "ABCDE"
eval_model     = 6
seeds          = collect(2001:(2000 + n_seeds))
tag            = "p$(planning_model)_a$(α)"

solver_configs = [
    "POMCPOW orig"       => p -> make_pomcpow(p),
    "POMCPOW k_action=10"=> p -> make_pomcpow(p; k_action=10.0),
    "POMCPOW heuristic"  => p -> make_pomcpow(p; next_action=:heuristic),
    "POMCPOW heur+k10"   => p -> make_pomcpow(p; next_action=:heuristic, k_action=10.0),
    "OneStep"            => make_onestep,
]
model_configs = [
    "POMCPOW orig" => p -> make_pomcpow(p),
    "OneStep"      => make_onestep,
]

println("LEVER TEST  planning=$planning_model  eval=$eval_model  α=$α  seeds=$seeds  experiments=$experiments")
println("Original POMCPOW: ", ORIGINAL_POMCPOW)

if 'A' in experiments
    println("\n### A: base model, solver levers")
    run_comparison("levers_A_base_$tag", solver_configs, planning_model, eval_model, [α], seeds)
end
if 'B' in experiments
    println("\n### B: exploration cost x50")
    run_comparison("levers_B_explcost_$tag", model_configs, planning_model, eval_model, [α], seeds;
                   model_kwargs=(exploration_cost_dle=6.0e7, exploration_cost_hard_rock=2.0e7))
end
if 'C' in experiments
    println("\n### C: exploration noise halved")
    run_comparison("levers_C_explnoise_$tag", model_configs, planning_model, eval_model, [α], seeds;
                   model_kwargs=(σ_obs_d_dle=10_000.0, σ_obs_d_hard_rock=45_000.0))
end
if 'D' in experiments
    println("\n### D: reward_scale 1e9 (disc rescaled to the 1e10 scale for comparison)")
    run_comparison("levers_D_rewardscale_$tag", model_configs, planning_model, eval_model, [α], seeds;
                   model_kwargs=(reward_scale=1.0e9,), reward_rescale=0.1)
end
if 'E' in experiments
    println("\n### E: emission_scale_factor 10_000")
    run_comparison("levers_E_emisscale_$tag", model_configs, planning_model, eval_model, [α], seeds;
                   model_kwargs=(emission_scale_factor=10_000.0,))
end
