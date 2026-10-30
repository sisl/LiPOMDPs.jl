"""
POMCPOW with naive rollouts vs OneStepLookahead with the AlphaAware base policy, all planning price
models × all α, evaluated on the historical price model (6). Paired seeds. Two POMCPOW variants:
  - DynamicProfitMaximizer rollout (the original final_results configuration)
  - AlphaAwareRolloutPolicy rollout: the same policy OneStep uses as its base, so the two planners
    differ only in how they search, not in their rollout policy

  julia --project=. -t 40 scripts/test_dynprofit_pomcpow_vs_onestep.jl [n_seeds] [tree_queries]
"""

include(joinpath(@__DIR__, "pomcpow_tuning_harness.jl"))

n_seeds = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
queries = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 30_000
seeds   = collect(4001:(4000 + n_seeds))
cells   = [(pm, α) for pm in (1, 2, 3, 4, 6) for α in (0.0, 0.25, 0.5, 0.75, 1.0)]
configs = [
    "POMCPOW dynprofit q=$queries"   => p -> make_pomcpow(p; rollout=:dynamic_profit, tree_queries=queries),
    "POMCPOW alphaaware q=$queries"  => p -> make_pomcpow(p; rollout=:alpha_aware,    tree_queries=queries),
    "OneStepLookahead (AlphaAware)" => make_onestep,
]
names = first.(configs)

jobs = [(c, name, seed) for (name, _) in configs for c in cells for seed in seeds]
results = Dict{Tuple{Tuple{Int,Float64},String,Int}, NamedTuple}()
lk = ReentrantLock(); t0 = time()
println("POMCPOW (DynamicProfit and AlphaAware rollouts, q=$queries) vs OneStep (AlphaAware): $(length(jobs)) episodes on $(Threads.nthreads()) threads, γ=$(LiPOMDP().γ)")
Threads.@threads :greedy for (c, name, seed) in jobs
    planning   = LiPOMDP(price_model_type=c[1], alpha=c[2])
    evaluation = LiPOMDP(price_model_type=6,    alpha=c[2])
    r = run_scenario(Dict(configs)[name](planning), planning, evaluation, LiBeliefUpdater(P=planning), seed)
    lock(lk) do
        results[(c, name, seed)] = r
        @printf("  [%d/%d] %.0f min  p%d α=%.2f %s seed=%d  disc=%.3f expl=%d\n",
                length(results), length(jobs), (time() - t0) / 60, c..., name, seed, r.disc, r.explores)
        flush(stdout)
    end
end
@printf("%d episodes in %.0f min\n", length(jobs), (time() - t0) / 60)

mkpath("outputs/pomcpow_vs_heuristics")
csv_path = "outputs/pomcpow_vs_heuristics/naive_rollout_pomcpow_vs_onestep_q$(queries)_e6.csv"
open(csv_path, "w") do io
    println(io, "planning_model,eval_model,alpha,policy,seed,discounted_reward,profit,emission_cost,co2_tonnes,met_demand_pct,mines_opened,capex,explorations")
    for ((c, name, seed), r) in sort(collect(results), by=first)
        println(io, join([c[1], 6, c[2], "\"$name\"", seed, r.disc, r.profit, r.emis, r.co2, r.met, r.mines, r.capex, r.explores], ","))
    end
end

se(x) = std(x) / sqrt(length(x))
get(c, n, f) = [getfield(results[(c, n, s)], f) for s in seeds]
onestep = names[end]
for pname in names[1:end-1]
    println("\n$pname − $onestep, paired over $(length(seeds)) seeds")
    @printf("%-12s %9s %9s %18s  %s\n", "cell", "POMCPOW", "OneStep", "Δ ± SE", "verdict")
    wins = sig = losses = 0
    for c in cells
        a, b = get(c, pname, :disc), get(c, onestep, :disc)
        Δ = a .- b; m, s = mean(Δ), se(Δ)
        tag = (m == 0 && s == 0) ? "tie" : m > 2s ? "WIN (>2SE)" : m > 0 ? "win (noise)" : m > -2s ? "loss (noise)" : "LOSS (>2SE)"
        wins += m > 0; sig += m > 2s; losses += m < -2s
        @printf("p%d α=%-6.2f %9.3f %9.3f %+9.3f ± %.3f  %s\n", c..., mean(a), mean(b), m, s, tag)
    end
    println("$pname ahead in $wins/25 cells ($sig beyond 2 SE); clear losses $losses")
end
println("CSV: $csv_path")
