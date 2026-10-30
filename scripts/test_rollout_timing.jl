"""
Does a timing-aware rollout policy fix POMCPOW and OneStepLookahead?

Both planners estimate future value with rollouts from a heuristic. The default heuristics
(DynamicProfitMaximizer for POMCPOW, AlphaAware for OneStep) open a mine at the first chance, so
rollouts undervalue waiting. This compares each planner with its default rollout, with
TimingAwarePolicy (optimal stopping on the expected path), and POMCPOW with no rollout (leaf
value 0). TimingAwarePolicy alone is included as a baseline the planners must beat.

  julia --project=. -t 64 scripts/test_rollout_timing.jl [n_seeds] [alphas] [planning_models] [gamma]
"""

include(joinpath(@__DIR__, "pomcpow_tuning_harness.jl"))

n_seeds         = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
alphas          = length(ARGS) >= 2 ? parse.(Float64, split(ARGS[2], ",")) : [0.5, 1.0]
planning_models = length(ARGS) >= 3 ? parse.(Int, split(ARGS[3], ",")) : [3, 6]
γ               = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : LiPOMDP().γ
eval_model      = 6
seeds           = collect(4001:(4000 + n_seeds))   # same seeds as the grids

configs = [
    "POMCPOW dynprofit rollout" => p -> make_pomcpow(p; rollout = p.alpha == 0.0 ? :zero : :dynamic_profit),
    "POMCPOW timing rollout"    => p -> make_pomcpow(p; rollout = :timing),
    "POMCPOW leaf value 0"      => p -> make_pomcpow(p; rollout = :zero),
    "OneStep alphaaware base"   => make_onestep,
    "OneStep timing base"       => make_onestep_timing,
    "TimingAware"               => p -> TimingAwarePolicy(p),
]
names = first.(configs)

jobs = [(pm, α, name, seed) for name in names for pm in planning_models for α in alphas for seed in seeds]
results = Dict{Tuple{Int,Float64,String,Int}, NamedTuple}()
lk = ReentrantLock()

println("ROLLOUT TIMING TEST  planning=$planning_models  eval=$eval_model  α=$alphas  γ=$γ  seeds=$(length(seeds))  threads=$(Threads.nthreads())")
t0 = time()
Threads.@threads :greedy for (pm, α, name, seed) in jobs
    planning   = LiPOMDP(price_model_type=pm,         alpha=α, γ=γ)
    evaluation = LiPOMDP(price_model_type=eval_model, alpha=α, γ=γ)
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
csv_path = "outputs/pomcpow_vs_heuristics/rollout_timing_p$(join(planning_models, "-"))_a$(join(alphas, "-"))_g$(γ)_e$(eval_model).csv"
open(csv_path, "w") do io
    println(io, "planning_model,eval_model,alpha,policy,seed,discounted_reward,profit,emission_cost,co2_tonnes,met_demand_pct,mines_opened,capex,explorations")
    for ((pm, α, name, seed), r) in sort(collect(results), by=first)
        println(io, join([pm, eval_model, α, name, seed, r.disc, r.profit, r.emis, r.co2, r.met, r.mines, r.capex, r.explores], ","))
    end
end

se(x) = std(x) / sqrt(length(x))
get(pm, α, n, f) = [getfield(results[(pm, α, n, s)], f) for s in seeds]

println("\nMean discounted reward [explorations, mines]; Δ vs TimingAware, paired ± SE")
for pm in planning_models, α in alphas
    @printf("p%d α=%.2f\n", pm, α)
    base = get(pm, α, "TimingAware", :disc)
    for n in names
        d = get(pm, α, n, :disc)
        @printf("  %-26s %7.3f  [%5.1f expl, %.1f mines]", n, mean(d), mean(get(pm, α, n, :explores)), mean(get(pm, α, n, :mines)))
        n == "TimingAware" || @printf("   Δ %+7.3f ± %.3f", mean(d .- base), se(d .- base))
        println()
    end
end
println("\nCSV: $csv_path")
