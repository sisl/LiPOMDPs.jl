"""
Shared harness for small POMCPOW-vs-OneStepLookahead tuning experiments.

Included by:
  scripts/test_pomcpow_rollout_combos.jl
  scripts/test_pomcpow_hypertune.jl

Design
- Paired scenarios: every policy sees the same initial state and the same
  environment noise. The global RNG is reseeded from (scenario_seed, t) right
  before each environment step, because update_price / update_demand /
  observation draw from the global RNG (common random numbers).
- Planning POMDP drives the policy; evaluation POMDP drives transitions,
  rewards, costs and emissions (same split as final_results_simulation.jl).
- Metrics per scenario: discounted reward, profit (USD), emission cost (USD),
  CO2 tonnes, met-demand %, mines opened, capex (USD).
"""

using LiPOMDPs
using Random
using Statistics
using Printf
using POMDPs
using POMDPTools
using POMCPOW
using POMCPOW: POMCPOWPlanner
import LiPOMDPs: calculate_total_production, compute_costs, compute_emissions

# -----------------------------------------------------------------------------
# Original POMCPOW settings from final_results_simulation_copy.jl (lines 51-64)
# -----------------------------------------------------------------------------
const ORIGINAL_POMCPOW = (
    tree_queries      = 10000,
    max_depth         = 25,
    ucb_c             = 1.0,
    enable_action_pw  = true,
    k_action          = 5.0,
    alpha_action      = 0.5,
    k_observation     = 3.0,
    alpha_observation = 0.3,
    rollout           = :dynamic_profit,   # FORollout(DynamicProfitMaximizerPolicy)
)

"""Build a POMCPOW planner. Keyword overrides are applied on top of ORIGINAL_POMCPOW."""
function make_pomcpow(pomdp::LiPOMDP; kwargs...)
    cfg = merge(ORIGINAL_POMCPOW, (; kwargs...))

    est = if cfg.rollout == :dynamic_profit
        FORollout(DynamicProfitMaximizerPolicy(pomdp))
    elseif cfg.rollout == :alpha_aware
        FORollout(AlphaAwareRolloutPolicy(pomdp))
    elseif cfg.rollout == :zero
        0
    else
        error("unknown rollout $(cfg.rollout)")
    end

    solver = POMCPOWSolver(
        tree_queries      = cfg.tree_queries,
        max_depth         = cfg.max_depth,
        criterion         = MaxUCB(cfg.ucb_c),
        enable_action_pw  = cfg.enable_action_pw,
        k_action          = cfg.k_action,
        alpha_action      = cfg.alpha_action,
        k_observation     = cfg.k_observation,
        alpha_observation = cfg.alpha_observation,
        estimate_value    = est,
        rng               = MersenneTwister(7),
    )
    return solve(solver, pomdp)
end

make_onestep(pomdp::LiPOMDP) = OneStepLookaheadPolicy(pomdp, n_rollouts=30, debug=false)

# -----------------------------------------------------------------------------
# One paired scenario
# -----------------------------------------------------------------------------
function run_scenario(policy, planning::LiPOMDP, evaluation::LiPOMDP, updater, scenario_seed::Int)
    Random.seed!(scenario_seed)
    s0, b0 = create_initial_belief(planning)
    b, s = b0, s0

    disc = 0.0; profit = 0.0; emis = 0.0; co2 = 0.0
    production = 0.0; demand = 0.0; capex = 0.0

    for t in 1:planning.time_horizon
        a = action(policy, b)

        # Common random numbers for the environment step
        Random.seed!(hash((scenario_seed, t)))
        env = MersenneTwister(hash((scenario_seed, t, :env)))
        sp, o, r = gen(evaluation, s, a, env)

        costs = compute_costs(evaluation, s, a)
        e     = compute_emissions(evaluation, s, a)
        prod, _ = calculate_total_production(evaluation, s)

        for site in 1:evaluation.n_deposits
            if a.type[site] == 2 && !s.have_opened[site]
                capex += evaluation.deposit_types[site] ? evaluation.dle_capex : evaluation.mining_capex
            end
        end

        profit     += prod * s.price - costs
        emis       += e
        co2        += abs(e) / evaluation.CO2_cost
        production += prod
        demand     += s.company_demand
        disc       += evaluation.γ^(t - 1) * r

        isterminal(evaluation, sp) && break
        b = update(updater, b, a, o)
        s = sp
    end

    return (disc=disc, profit=profit, emis=emis, co2=co2,
            met=(demand > 0 ? 100 * production / demand : 0.0),
            mines=count(s.have_opened), capex=capex)
end

# -----------------------------------------------------------------------------
# Full comparison across configs × alphas × seeds
# configs :: Vector{Pair{String, Function}}  where f(pomdp) -> policy
# -----------------------------------------------------------------------------
function run_comparison(test_name::String, configs, planning_model::Int, eval_model::Int,
                        alphas::Vector{Float64}, seeds::Vector{Int};
                        horizon::Int=29, baseline::String="OneStep",
                        output_dir::String="outputs/pomcpow_tuning")

    mkpath(output_dir)
    csv_path = joinpath(output_dir, "$(test_name)_p$(planning_model)_e$(eval_model).csv")
    io = open(csv_path, "w")
    println(io, "test,config,alpha,seed,discounted_reward,profit,emission_cost,co2_tonnes,met_demand_pct,mines_opened,capex")

    names = first.(configs)
    results = Dict{Tuple{String,Float64}, Vector{NamedTuple}}()

    for α in alphas
        println("\n", "="^100)
        println("α = $α   planning=$planning_model  eval=$eval_model  seeds=$seeds")
        println("="^100)

        planning   = LiPOMDP(price_model_type=planning_model, time_horizon=horizon, alpha=α)
        evaluation = LiPOMDP(price_model_type=eval_model,     time_horizon=horizon, alpha=α)
        updater    = LiBeliefUpdater(P=evaluation)

        policies = [name => f(planning) for (name, f) in configs]
        for (name, _) in policies
            results[(name, α)] = NamedTuple[]
        end

        for seed in seeds
            println("  seed $seed")
            for (name, pol) in policies
                t0 = time()
                m = run_scenario(pol, planning, evaluation, updater, seed)
                push!(results[(name, α)], m)
                @printf("    %-28s disc=%7.3f  profit=\$%8.0fM  emis=\$%6.2fM  met=%5.1f%%  mines=%d  (%.0fs)\n",
                        name, m.disc, m.profit/1e6, m.emis/1e6, m.met, m.mines, time()-t0)
                println(io, join([test_name, name, α, seed, m.disc, m.profit, m.emis, m.co2, m.met, m.mines, m.capex], ","))
                flush(io)
            end
        end

        print_summary(α, names, results, baseline)
    end
    close(io)

    println("\nAll alphas: mean discounted reward")
    @printf("  %-28s", "config"); for α in alphas; @printf("  α=%-5s", α); end; println()
    for name in names
        @printf("  %-28s", name)
        for α in alphas
            @printf("  %7.3f", mean(getfield.(results[(name, α)], :disc)))
        end
        println()
    end
    println("\nCSV: $csv_path")
    return results
end

function print_summary(α, names, results, baseline)
    println("\n  Summary α=$α")
    @printf("  %-28s %9s %7s %10s %10s %7s %6s %9s\n",
            "config", "disc", "±std", "profit\$M", "emis\$M", "met%", "mines", "Δ vs $baseline")
    base = haskey(results, (baseline, α)) ? getfield.(results[(baseline, α)], :disc) : nothing
    for name in names
        rs = results[(name, α)]
        d  = getfield.(rs, :disc)
        Δ  = base === nothing ? NaN : mean(d .- base)
        @printf("  %-28s %9.3f %7.3f %10.0f %10.2f %7.1f %6.2f %+9.3f\n",
                name, mean(d), length(d) > 1 ? std(d) : 0.0,
                mean(getfield.(rs, :profit))/1e6, mean(getfield.(rs, :emis))/1e6,
                mean(getfield.(rs, :met)), mean(getfield.(rs, :mines)), Δ)
    end
end

# -----------------------------------------------------------------------------
# CLI helpers:  julia script.jl [n_seeds] [alphas e.g. 0.5,1.0] [planning_model]
# -----------------------------------------------------------------------------
function parse_cli(default_seeds::Int, default_alphas::Vector{Float64}, default_planning::Int)
    n_seeds  = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : default_seeds
    alphas   = length(ARGS) >= 2 ? parse.(Float64, split(ARGS[2], ",")) : default_alphas
    planning = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : default_planning
    seeds    = collect(1001:(1000 + n_seeds))
    return seeds, alphas, planning
end
