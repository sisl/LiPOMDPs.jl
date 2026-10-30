"""
Does POMCPOW's action proposer explain its gap to OneStepLookahead?

The grid (test_pomcpow_vs_heuristics.jl) showed POMCPOW opening mines better than
OneStepLookahead but exploring ~10× more (≈40 vs ≈4 per episode). This compares the
default uniform proposer with the heuristic and sparse-explore proposers from
pomcpow_tuning_harness.jl on the cells where POMCPOW lost.

  julia --project=. -t 32 scripts/test_pomcpow_action_proposer.jl [n_seeds] [alphas] [planning_models]
"""

include(joinpath(@__DIR__, "pomcpow_tuning_harness.jl"))

n_seeds         = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
alphas          = length(ARGS) >= 2 ? parse.(Float64, split(ARGS[2], ",")) : [0.5, 1.0]
planning_models = length(ARGS) >= 3 ? parse.(Int, split(ARGS[3], ",")) : [2, 3, 6]
eval_model      = 6
seeds           = collect(4001:(4000 + n_seeds))   # same seeds as the grid, so OneStep numbers line up

configs = [
    "POMCPOW random"         => p -> make_pomcpow(p),
    "POMCPOW heuristic"      => p -> make_pomcpow(p; next_action=:heuristic),
    "POMCPOW sparse_explore" => p -> make_pomcpow(p; next_action=:sparse_explore),
    "OneStepLookahead"       => make_onestep,
]
names = first.(configs)

jobs = [(pm, α, name, seed) for name in names for pm in planning_models for α in alphas for seed in seeds]
results = Dict{Tuple{Int,Float64,String,Int}, NamedTuple}()
lk = ReentrantLock()

println("ACTION PROPOSER TEST  planning=$planning_models  eval=$eval_model  α=$alphas  seeds=$(length(seeds))  threads=$(Threads.nthreads())")
t0 = time()
Threads.@threads :greedy for (pm, α, name, seed) in jobs
    planning   = LiPOMDP(price_model_type=pm,         alpha=α)
    evaluation = LiPOMDP(price_model_type=eval_model, alpha=α)
    policy = Dict(configs)[name](planning)
    r = run_scenario(policy, planning, evaluation, LiBeliefUpdater(P=planning), seed)
    lock(lk) do
        results[(pm, α, name, seed)] = r
        @printf("  [%d/%d] %.0f min  p%d α=%.2f %s seed=%d  disc=%.3f expl=%d\n",
                length(results), length(jobs), (time() - t0) / 60, pm, α, name, seed, r.disc, r.explores)
        flush(stdout)
    end
end
@printf("%d episodes in %.0f min\n", length(jobs), (time() - t0) / 60)

mkpath("outputs/pomcpow_vs_heuristics")
csv_path = "outputs/pomcpow_vs_heuristics/action_proposer_e$(eval_model).csv"
open(csv_path, "w") do io
    println(io, "planning_model,eval_model,alpha,policy,seed,discounted_reward,profit,emission_cost,co2_tonnes,met_demand_pct,mines_opened,capex,explorations")
    for ((pm, α, name, seed), r) in sort(collect(results), by=first)
        println(io, join([pm, eval_model, α, name, seed, r.disc, r.profit, r.emis, r.co2, r.met, r.mines, r.capex, r.explores], ","))
    end
end

se(x) = std(x) / sqrt(length(x))
get(pm, α, n, f) = [getfield(results[(pm, α, n, s)], f) for s in seeds]

println("\nMean discounted reward [explorations]; Δ = POMCPOW variant − OneStepLookahead, paired ± SE")
for pm in planning_models, α in alphas
    @printf("p%d α=%.2f\n", pm, α)
    base = get(pm, α, "OneStepLookahead", :disc)
    for n in names
        d = get(pm, α, n, :disc)
        @printf("  %-24s %7.3f  [%5.1f expl]", n, mean(d), mean(get(pm, α, n, :explores)))
        n == "OneStepLookahead" || @printf("   Δ %+7.3f ± %.3f", mean(d .- base), se(d .- base))
        println()
    end
end
println("\nCSV: $csv_path")
