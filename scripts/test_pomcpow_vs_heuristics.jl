"""
POMCPOW against every heuristic, across planning price models and alphas.

Planning models 1, 2, 3, 4, 6 are each evaluated in the historical-price world (model 6),
for α ∈ {0, 0.25, 0.5, 0.75, 1}. Paired design from pomcpow_tuning_harness.jl: for a given
seed every policy sees the same initial state and the same environment noise, so the
POMCPOW − heuristic differences are paired and their standard errors are small.

The belief updater uses the PLANNING model (the agent's own model of price and demand
dynamics); the evaluation model is only used for the true transitions. Flip UPDATER_MODEL
to :evaluation to give the agent the true dynamics for filtering.

Start Julia with threads; every (model, α, policy, seed) episode runs in parallel:

  julia --project=. -t 64 scripts/test_pomcpow_vs_heuristics.jl [n_seeds] [alphas] [planning_models] [gamma]
  julia --project=. -t 64 scripts/test_pomcpow_vs_heuristics.jl 8 0.0,0.25,0.5,0.75,1.0 1,2,3,4,6 0.926
"""

include(joinpath(@__DIR__, "pomcpow_tuning_harness.jl"))

n_seeds         = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8
alphas          = length(ARGS) >= 2 ? parse.(Float64, split(ARGS[2], ",")) : [0.0, 0.25, 0.5, 0.75, 1.0]
planning_models = length(ARGS) >= 3 ? parse.(Int, split(ARGS[3], ",")) : [1, 2, 3, 4, 6]
γ               = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : LiPOMDP().γ
eval_model      = 6
seeds           = collect(4001:(4000 + n_seeds))
const UPDATER_MODEL = :planning

# Same POMCPOW settings as final_results_simulation.jl: no rollout value when α = 0
configs = [
    "POMCPOW"                  => p -> make_pomcpow(p; rollout = p.alpha == 0.0 ? :zero : :dynamic_profit),
    "OneStepLookahead"         => make_onestep,
    "DynamicProfitMaximizer"   => p -> DynamicProfitMaximizerPolicy(p),
    "DynamicEmissionMinimizer" => p -> DynamicEmissionMinimizerPolicy(p),
    "AlphaAware"               => p -> AlphaAwareRolloutPolicy(p),
    "ExploreOnly"              => p -> ExploreOnlyPolicy(p),
    "OneMineProfit"            => p -> ProfitMaximizerPolicy(p, top_k=1, times=[1]),
    "OneMineEmission"          => p -> EmissionMinimizerPolicy(p, top_k=1, times=[1]),
    "Random"                   => p -> RandomHeuristicPolicy(p, debug=false),
]
names = first.(configs)
heuristics = names[2:end]

# POMCPOW jobs first: they are the slow ones, so the tail of the run stays parallel
jobs = [(pm, α, name, seed) for name in names for pm in planning_models for α in alphas for seed in seeds]
results = Dict{Tuple{Int,Float64,String,Int}, NamedTuple}()
lk = ReentrantLock()

println("POMCPOW vs HEURISTICS  planning=$planning_models  eval=$eval_model  α=$alphas  γ=$γ  seeds=$(length(seeds))  threads=$(Threads.nthreads())  updater=$UPDATER_MODEL")
println("POMCPOW: ", ORIGINAL_POMCPOW)
t0 = time()
Threads.@threads :greedy for (pm, α, name, seed) in jobs
    planning   = LiPOMDP(price_model_type=pm,         alpha=α, γ=γ)
    evaluation = LiPOMDP(price_model_type=eval_model, alpha=α, γ=γ)
    updater    = LiBeliefUpdater(P = UPDATER_MODEL == :planning ? planning : evaluation)
    policy = Dict(configs)[name](planning)   # one policy per job: planners are not thread-safe
    r = run_scenario(policy, planning, evaluation, updater, seed)
    lock(lk) do
        results[(pm, α, name, seed)] = r
        @printf("  [%d/%d] %.0f min  p%d α=%.2f %s seed=%d  disc=%.3f\n",
                length(results), length(jobs), (time() - t0) / 60, pm, α, name, seed, r.disc)
        flush(stdout)
    end
end
@printf("%d episodes in %.0f min\n", length(jobs), (time() - t0) / 60)

mkpath("outputs/pomcpow_vs_heuristics")
csv_path = "outputs/pomcpow_vs_heuristics/grid_e$(eval_model)_g$(γ)_recovery_budget.csv"
open(csv_path, "w") do io
    println(io, "planning_model,eval_model,alpha,policy,seed,discounted_reward,profit,emission_cost,co2_tonnes,met_demand_pct,mines_opened,capex,explorations")
    for ((pm, α, name, seed), r) in sort(collect(results), by=first)
        println(io, join([pm, eval_model, α, name, seed, r.disc, r.profit, r.emis, r.co2, r.met, r.mines, r.capex, r.explores], ","))
    end
end

se(x) = std(x) / sqrt(length(x))
disc(pm, α, name) = [results[(pm, α, name, s)].disc for s in seeds]

println("\nMean discounted reward (evaluation model $eval_model). Last columns: POMCPOW − best heuristic, paired ± SE")
@printf("%-5s %-5s", "plan", "α")
for n in names; @printf(" %9s", first(n, 9)); end
println("   best heuristic            Δ ± SE")
wins = 0; cells = 0
for pm in planning_models, α in alphas
    global wins, cells
    @printf("%-5d %-5.2f", pm, α)
    for n in names; @printf(" %9.3f", mean(disc(pm, α, n))); end
    best = heuristics[argmax([mean(disc(pm, α, h)) for h in heuristics])]
    Δ = disc(pm, α, "POMCPOW") .- disc(pm, α, best)
    cells += 1; wins += mean(Δ) > 0
    @printf("   %-24s %+7.3f ± %.3f\n", best, mean(Δ), se(Δ))
end
println("\nPOMCPOW best in $wins / $cells cells")

println("\nMean over all cells per policy:")
for n in names
    @printf("  %-26s %8.3f\n", n, mean(mean(disc(pm, α, n)) for pm in planning_models, α in alphas))
end
println("\nCSV: $csv_path")
