"""
Sensitivity of policy performance to prior deposit uncertainty.

Scales the coefficients of variation of the deposit prior (dle_uncertainty_factor,
mining_uncertainty_factor) by a multiplier m, keeping the DLE/hard-rock ratio fixed.
m = 1 is the default model (Inferred-resource level uncertainty).

Paired design from pomcpow_tuning_harness.jl: every policy sees the same initial state
and the same environment noise for a given seed. Episodes run in parallel, so start
Julia with threads:

  julia --project=. -t auto scripts/test_uncertainty_sweep.jl [n_seeds] [alphas] [planning_model] [eval_model]
  julia --project=. -t 80 scripts/test_uncertainty_sweep.jl 20 0.5 3 6
"""

include(joinpath(@__DIR__, "pomcpow_tuning_harness.jl"))

n_seeds        = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20
alphas         = length(ARGS) >= 2 ? parse.(Float64, split(ARGS[2], ",")) : [0.5]
planning_model = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 3
eval_model     = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 6
seeds          = collect(3001:(3000 + n_seeds))

const MULTIPLIERS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
const BASE_CV = (dle = LiPOMDP().dle_uncertainty_factor, mining = LiPOMDP().mining_uncertainty_factor)

configs = [
    "POMCPOW"                => p -> make_pomcpow(p),
    "OneStep"                => make_onestep,
    "AlphaAware"             => p -> AlphaAwareRolloutPolicy(p),
    "DynamicProfitMaximizer" => p -> DynamicProfitMaximizerPolicy(p),
]
names = first.(configs)

model_kwargs(m) = (dle_uncertainty_factor = m * BASE_CV.dle, mining_uncertainty_factor = m * BASE_CV.mining)

# Slow POMCPOW jobs first so the tail of the run stays parallel
jobs = [(m, α, name, seed) for name in names for m in MULTIPLIERS for α in alphas for seed in seeds]
results = Dict{Tuple{Float64,Float64,String,Int}, NamedTuple}()
lk = ReentrantLock()

println("UNCERTAINTY SWEEP  planning=$planning_model  eval=$eval_model  α=$alphas  seeds=$(length(seeds))  threads=$(Threads.nthreads())")
println("base CV: DLE=$(BASE_CV.dle)  hard rock=$(BASE_CV.mining)   multipliers=$MULTIPLIERS")
t0 = time()
Threads.@threads :greedy for (m, α, name, seed) in jobs
    kw = model_kwargs(m)
    planning   = LiPOMDP(; price_model_type=planning_model, alpha=α, kw...)
    evaluation = LiPOMDP(; price_model_type=eval_model,     alpha=α, kw...)
    policy = Dict(configs)[name](planning)   # one policy per job: planners are not thread-safe
    r = run_scenario(policy, planning, evaluation, LiBeliefUpdater(P=planning), seed)   # agent filters with its own model
    lock(lk) do
        results[(m, α, name, seed)] = r
        @printf("  [%d/%d] %.0f min  %s m=%.2f seed=%d\n", length(results), length(jobs), (time() - t0) / 60, name, m, seed)
        flush(stdout)
    end
end
@printf("%d episodes in %.0f min\n", length(jobs), (time() - t0) / 60)

mkpath("outputs/uncertainty_sweep")
csv_path = "outputs/uncertainty_sweep/sweep_p$(planning_model)_e$(eval_model).csv"
open(csv_path, "w") do io
    println(io, "multiplier,cv_dle,cv_mining,alpha,policy,seed,discounted_reward,profit,emission_cost,met_demand_pct,mines_opened,explorations")
    for ((m, α, name, seed), r) in sort(collect(results), by=first)
        println(io, join([m, m * BASE_CV.dle, m * BASE_CV.mining, α, name, seed,
                          r.disc, r.profit, r.emis, r.met, r.mines, r.explores], ","))
    end
end

se(x) = std(x) / sqrt(length(x))
for α in alphas
    println("\nα = $α: mean discounted reward, and paired difference POMCPOW − policy (± SE)")
    @printf("  %-5s %-11s", "m", "CV dle/hr")
    for n in names; @printf(" %10s", first(n, 10)); end
    for n in names[2:end]; @printf("   POMCPOW−%-10s", first(n, 10)); end
    println()
    for m in MULTIPLIERS
        d(n) = [results[(m, α, n, s)].disc for s in seeds]
        @printf("  %-5.2f %4.2f/%-6.2f", m, m * BASE_CV.dle, m * BASE_CV.mining)
        for n in names; @printf(" %10.3f", mean(d(n))); end
        for n in names[2:end]
            Δ = d("POMCPOW") .- d(n)
            @printf("   %+7.3f ± %-8.3f", mean(Δ), se(Δ))
        end
        println()
    end
end
println("\nCSV: $csv_path")
