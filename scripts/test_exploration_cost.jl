"""
Sensitivity of POMCPOW vs OneStepLookahead to exploration cost.

Both exploration costs (DLE and hard rock) are scaled by a multiplier; both planners use the
timing-aware rollout/base policy. Paired seeds as in the other harness scripts.

  julia --project=. -t 20 scripts/test_exploration_cost.jl [n_seeds] [alphas] [planning_models] [multipliers]
  julia --project=. -t 20 scripts/test_exploration_cost.jl 10 0.5,1.0 3,6 0.25,0.5,2.0
"""

include(joinpath(@__DIR__, "pomcpow_tuning_harness.jl"))

n_seeds         = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
alphas          = length(ARGS) >= 2 ? parse.(Float64, split(ARGS[2], ",")) : [0.5, 1.0]
planning_models = length(ARGS) >= 3 ? parse.(Int, split(ARGS[3], ",")) : [3, 6]
multipliers     = length(ARGS) >= 4 ? parse.(Float64, split(ARGS[4], ",")) : [0.25, 0.5, 2.0]
eval_model      = 6
seeds           = collect(4001:(4000 + n_seeds))
const BASE = LiPOMDP()

configs = [
    "POMCPOW timing q=10000" => p -> make_pomcpow(p; rollout=:timing),
    "OneStep timing n=30"    => make_onestep_timing,
]
names = first.(configs)
cost_kwargs(m) = (exploration_cost_dle = m * BASE.exploration_cost_dle, exploration_cost_hard_rock = m * BASE.exploration_cost_hard_rock)

jobs = [(m, pm, α, name, seed) for name in names for m in multipliers for pm in planning_models for α in alphas for seed in seeds]
results = Dict{Tuple{Float64,Int,Float64,String,Int}, NamedTuple}()
lk = ReentrantLock()

println("EXPLORATION COST SENSITIVITY  multipliers=$multipliers (base DLE \$$(BASE.exploration_cost_dle/1e6)M, HR \$$(BASE.exploration_cost_hard_rock/1e6)M)  planning=$planning_models  α=$alphas  seeds=$(length(seeds))  threads=$(Threads.nthreads())")
t0 = time()
Threads.@threads :greedy for (m, pm, α, name, seed) in jobs
    planning   = LiPOMDP(; price_model_type=pm,         alpha=α, cost_kwargs(m)...)
    evaluation = LiPOMDP(; price_model_type=eval_model, alpha=α, cost_kwargs(m)...)
    policy = Dict(configs)[name](planning)
    r = run_scenario(policy, planning, evaluation, LiBeliefUpdater(P=planning), seed)
    lock(lk) do
        results[(m, pm, α, name, seed)] = r
        @printf("  [%d/%d] %.0f min  ×%.2f p%d α=%.2f %s seed=%d  disc=%.3f expl=%d\n",
                length(results), length(jobs), (time() - t0) / 60, m, pm, α, name, seed, r.disc, r.explores)
        flush(stdout)
    end
end
@printf("%d episodes in %.0f min\n", length(jobs), (time() - t0) / 60)

mkpath("outputs/pomcpow_vs_heuristics")
csv_path = "outputs/pomcpow_vs_heuristics/exploration_cost_p$(join(planning_models, "-"))_a$(join(alphas, "-"))_e$(eval_model).csv"
open(csv_path, "w") do io
    println(io, "cost_multiplier,planning_model,eval_model,alpha,policy,seed,discounted_reward,profit,emission_cost,co2_tonnes,met_demand_pct,mines_opened,capex,explorations")
    for ((m, pm, α, name, seed), r) in sort(collect(results), by=first)
        println(io, join([m, pm, eval_model, α, name, seed, r.disc, r.profit, r.emis, r.co2, r.met, r.mines, r.capex, r.explores], ","))
    end
end

se(x) = std(x) / sqrt(length(x))
get(m, pm, α, n, f) = [getfield(results[(m, pm, α, n, s)], f) for s in seeds]
println("\nPOMCPOW − OneStep (paired ± SE); mean discounted reward [explorations]")
for pm in planning_models, α in alphas, m in multipliers
    a, b = get(m, pm, α, names[1], :disc), get(m, pm, α, names[2], :disc)
    @printf("p%d α=%.2f ×%-5.2f  POMCPOW %6.3f [%4.1f]  OneStep %6.3f [%4.1f]   Δ %+6.3f ± %.3f\n", pm, α, m,
            mean(a), mean(get(m, pm, α, names[1], :explores)), mean(b), mean(get(m, pm, α, names[2], :explores)),
            mean(a .- b), se(a .- b))
end
println("\nCSV: $csv_path")
