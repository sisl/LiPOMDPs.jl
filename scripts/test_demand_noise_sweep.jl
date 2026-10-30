"""
Sensitivity of policy performance to demand uncertainty.

Sets `demand_uncertainty` (std of next-year global production as a fraction of its historical
value, which drives company demand) to each level in LEVELS, for both the planning and the
evaluation model. 0.05 is the default model.

Paired design from pomcpow_tuning_harness.jl: every policy sees the same initial state and the
same environment noise for a given seed. Rows are appended to the CSV as episodes finish, so a
long run can be inspected (or resumed by hand) part-way. Start Julia with threads:

  julia --project=. -t auto scripts/test_demand_noise_sweep.jl [n_seeds] [alphas] [planning_models] [tree_queries] [eval_model]
  julia --project=. -t 80 scripts/test_demand_noise_sweep.jl 20 0.5 1,2,3,4,6 30000 6
"""

include(joinpath(@__DIR__, "pomcpow_tuning_harness.jl"))

n_seeds         = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20
alphas          = length(ARGS) >= 2 ? parse.(Float64, split(ARGS[2], ",")) : [0.5]
planning_models = length(ARGS) >= 3 ? parse.(Int, split(ARGS[3], ",")) : [1, 2, 3, 4, 6]
tree_queries    = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 30_000
eval_model      = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 6
seeds           = collect(7001:(7000 + n_seeds))

const LEVELS = [0.05, 0.10, 0.20]

configs = [
    "POMCPOW"                => p -> make_pomcpow(p; rollout=:timing, tree_queries=tree_queries),
    "OneStep"                => make_onestep,
    "TimingAware"            => p -> TimingAwarePolicy(p),
    "DynamicProfitMaximizer" => p -> DynamicProfitMaximizerPolicy(p),
]
names = first.(configs)

# Slow POMCPOW jobs first so the tail of the run stays parallel
jobs = [(d, pm, α, name, seed) for name in names for d in LEVELS for pm in planning_models
        for α in alphas for seed in seeds]
results = Dict{Tuple{Float64,Int,Float64,String,Int}, NamedTuple}()
lk = ReentrantLock()

mkpath("outputs/demand_noise_sweep")
csv_path = "outputs/demand_noise_sweep/sweep_p$(join(planning_models, "-"))_e$(eval_model)_q$(tree_queries).csv"
open(csv_path, "w") do io
    println(io, "demand_uncertainty,planning_model,eval_model,alpha,policy,seed,discounted_reward,profit,emission_cost,met_demand_pct,mines_opened,explorations")
end

println("DEMAND NOISE SWEEP  planning=$planning_models  eval=$eval_model  α=$alphas  seeds=$(length(seeds))  " *
        "POMCPOW q=$tree_queries  threads=$(Threads.nthreads())")
println("levels: $LEVELS   jobs: $(length(jobs))")
t0 = time()
Threads.@threads :greedy for (d, pm, α, name, seed) in jobs
    planning   = LiPOMDP(; price_model_type=pm,         alpha=α, demand_uncertainty=d)
    evaluation = LiPOMDP(; price_model_type=eval_model, alpha=α, demand_uncertainty=d)
    policy = Dict(configs)[name](planning)   # one policy per job: planners are not thread-safe
    r = run_scenario(policy, planning, evaluation, LiBeliefUpdater(P=planning), seed)   # agent filters with its own model
    lock(lk) do
        results[(d, pm, α, name, seed)] = r
        open(csv_path, "a") do io
            println(io, join([d, pm, eval_model, α, name, seed,
                              r.disc, r.profit, r.emis, r.met, r.mines, r.explores], ","))
        end
        @printf("  [%d/%d] %.0f min  %s d=%.2f p%d α=%.2f seed=%d  disc=%.3f\n",
                length(results), length(jobs), (time() - t0) / 60, name, d, pm, α, seed, r.disc)
        flush(stdout)
    end
end
@printf("%d episodes in %.0f min\n", length(jobs), (time() - t0) / 60)

se(x) = std(x) / sqrt(length(x))
for α in alphas, pm in planning_models
    println("\np$pm α = $α: mean discounted reward, and paired difference POMCPOW − policy (± SE)")
    @printf("  %-6s", "d")
    for n in names; @printf(" %10s", first(n, 10)); end
    for n in names[2:end]; @printf("   POMCPOW−%-10s", first(n, 10)); end
    println()
    for d in LEVELS
        v(n) = [results[(d, pm, α, n, s)].disc for s in seeds]
        @printf("  %-6.2f", d)
        for n in names; @printf(" %10.3f", mean(v(n))); end
        for n in names[2:end]
            Δ = v("POMCPOW") .- v(n)
            @printf("   %+7.3f ± %-8.3f", mean(Δ), se(Δ))
        end
        println()
    end
end
println("\nCSV: $csv_path")
