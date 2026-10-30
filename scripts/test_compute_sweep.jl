"""
Performance vs compute for the two planners, both with the timing-aware rollout/base policy.

POMCPOW is run at several tree-query budgets and OneStepLookahead at several rollout counts.
Wall-clock time per episode is recorded for every run so results can be reported as
performance against compute, not as a single pairing of budgets.

  julia --project=. -t 40 scripts/test_compute_sweep.jl [n_seeds] [alphas] [planning_models] [queries] [rollouts] [gamma] [capital_budget]
  julia --project=. -t 40 scripts/test_compute_sweep.jl 10 0.5,1.0 3,6 30000,50000 10,20
  julia --project=. -t 40 scripts/test_compute_sweep.jl 10 0.5,1.0 3,6 20000 none 0.926   # POMCPOW only
"""

include(joinpath(@__DIR__, "pomcpow_tuning_harness.jl"))

n_seeds         = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
alphas          = length(ARGS) >= 2 ? parse.(Float64, split(ARGS[2], ",")) : [0.5, 1.0]
planning_models = length(ARGS) >= 3 ? parse.(Int, split(ARGS[3], ",")) : [3, 6]
queries         = length(ARGS) >= 4 ? (ARGS[4] == "none" ? Int[] : parse.(Int, split(ARGS[4], ","))) : [30_000, 50_000]
rollouts        = length(ARGS) >= 5 ? (ARGS[5] == "none" ? Int[] : parse.(Int, split(ARGS[5], ","))) : [10, 20]
γ               = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : LiPOMDP().γ
budget          = length(ARGS) >= 7 ? parse(Float64, ARGS[7]) : LiPOMDP().capital_budget   # "Inf" = no limit
eval_model      = 6
seeds           = collect(4001:(4000 + n_seeds))   # same seeds as the grids

configs = vcat(
    ["POMCPOW timing q=$q" => (p -> make_pomcpow(p; rollout=:timing, tree_queries=q)) for q in queries],
    ["OneStep timing n=$n" => (p -> OneStepLookaheadPolicy(p, n_rollouts=n, rollout_policy=TimingAwarePolicy(p))) for n in rollouts],
    ["TimingAware" => (p -> TimingAwarePolicy(p))],   # cheap; lets every cell be checked against it
)
names = first.(configs)

# Slow POMCPOW jobs first, largest budget first, so the tail of the run stays parallel
jobs = [(pm, α, name, seed) for name in names for pm in planning_models for α in alphas for seed in seeds]
results = Dict{Tuple{Int,Float64,String,Int}, NamedTuple}()
lk = ReentrantLock()

println("COMPUTE SWEEP  planning=$planning_models  eval=$eval_model  α=$alphas  γ=$γ  queries=$queries  rollouts=$rollouts  budget=$(budget)  seeds=$(length(seeds))  threads=$(Threads.nthreads())")
t0 = time()
Threads.@threads :greedy for (pm, α, name, seed) in jobs
    planning   = LiPOMDP(price_model_type=pm,         alpha=α, γ=γ, capital_budget=budget)
    evaluation = LiPOMDP(price_model_type=eval_model, alpha=α, γ=γ, capital_budget=budget)
    policy = Dict(configs)[name](planning)
    t_ep = time()
    r = run_scenario(policy, planning, evaluation, LiBeliefUpdater(P=planning), seed)
    r = merge(r, (seconds = time() - t_ep,))
    lock(lk) do
        results[(pm, α, name, seed)] = r
        @printf("  [%d/%d] %.0f min  p%d α=%.2f %s seed=%d  disc=%.3f expl=%d  (%.0fs)\n",
                length(results), length(jobs), (time() - t0) / 60, pm, α, name, seed, r.disc, r.explores, r.seconds)
        flush(stdout)
    end
end
@printf("%d episodes in %.0f min\n", length(jobs), (time() - t0) / 60)

mkpath("outputs/pomcpow_vs_heuristics")
csv_path = "outputs/pomcpow_vs_heuristics/compute_sweep_p$(join(planning_models, "-"))_a$(join(alphas, "-"))_g$(γ)$(budget == LiPOMDP().capital_budget ? "" : "_b$(budget)")_e$(eval_model).csv"
open(csv_path, "w") do io
    println(io, "planning_model,eval_model,alpha,policy,seed,discounted_reward,profit,emission_cost,co2_tonnes,met_demand_pct,mines_opened,capex,explorations,seconds")
    for ((pm, α, name, seed), r) in sort(collect(results), by=first)
        println(io, join([pm, eval_model, α, name, seed, r.disc, r.profit, r.emis, r.co2, r.met, r.mines, r.capex, r.explores, r.seconds], ","))
    end
end

get(pm, α, n, f) = [getfield(results[(pm, α, n, s)], f) for s in seeds]
println("\nMean discounted reward [explorations, seconds per episode under load]")
for pm in planning_models, α in alphas
    @printf("p%d α=%.2f\n", pm, α)
    for n in names
        @printf("  %-22s %7.3f  [%5.1f expl, %6.0f s]\n", n, mean(get(pm, α, n, :disc)), mean(get(pm, α, n, :explores)), mean(get(pm, α, n, :seconds)))
    end
end
println("\nCSV: $csv_path")
