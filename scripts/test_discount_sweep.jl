"""
Sensitivity of POMCPOW vs heuristics to the discount factor γ.

γ = 1/(1+r): 0.90 ≈ 11%, 0.926 ≈ 8% (typical real discount rate in mining feasibility
studies), 0.95 ≈ 5%, 0.97 ≈ 3%. Returns grow with γ, so POMCPOW's UCB exploration
constant is swept too (its scale is tied to the return magnitude).

Discounted rewards are only comparable WITHIN a γ; compare policies per row, not across rows.

  julia --project=. -t 48 scripts/test_discount_sweep.jl [n_seeds] [alphas] [planning_models]
  julia --project=. -t 48 scripts/test_discount_sweep.jl 10 0.5,1.0 3,4
"""

include(joinpath(@__DIR__, "pomcpow_tuning_harness.jl"))

n_seeds         = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
alphas          = length(ARGS) >= 2 ? parse.(Float64, split(ARGS[2], ",")) : [0.5, 1.0]
planning_models = length(ARGS) >= 3 ? parse.(Int, split(ARGS[3], ",")) : [3, 4]
eval_model      = 6
seeds           = collect(5001:(5000 + n_seeds))

const GAMMAS = [0.90, 0.926, 0.95, 0.97]
const UCB_CS = [1.0, 3.0]

configs = vcat(
    ["POMCPOW c=$c" => (p -> make_pomcpow(p; ucb_c = c, rollout = p.alpha == 0.0 ? :zero : :dynamic_profit)) for c in UCB_CS],
    [
        "OneStepLookahead"       => make_onestep,
        "DynamicProfitMaximizer" => p -> DynamicProfitMaximizerPolicy(p),
        "ExploreOnly"            => p -> ExploreOnlyPolicy(p),
        "OneMineProfit"          => p -> ProfitMaximizerPolicy(p, top_k=1, times=[1]),
    ],
)
names = first.(configs)
pomcpow_names = filter(n -> startswith(n, "POMCPOW"), names)
heuristics = filter(n -> !startswith(n, "POMCPOW"), names)

# POMCPOW jobs first: they are the slow ones, so the tail of the run stays parallel
jobs = [(γ, pm, α, name, seed) for name in names for γ in GAMMAS for pm in planning_models for α in alphas for seed in seeds]
results = Dict{Tuple{Float64,Int,Float64,String,Int}, NamedTuple}()
lk = ReentrantLock()

println("DISCOUNT SWEEP  γ=$GAMMAS  ucb_c=$UCB_CS  planning=$planning_models  eval=$eval_model  α=$alphas  seeds=$(length(seeds))  threads=$(Threads.nthreads())")
t0 = time()
Threads.@threads :greedy for (γ, pm, α, name, seed) in jobs
    planning   = LiPOMDP(price_model_type=pm,         alpha=α, γ=γ)
    evaluation = LiPOMDP(price_model_type=eval_model, alpha=α, γ=γ)
    policy = Dict(configs)[name](planning)   # one policy per job: planners are not thread-safe
    r = run_scenario(policy, planning, evaluation, LiBeliefUpdater(P=planning), seed)   # agent filters with its own model
    lock(lk) do
        results[(γ, pm, α, name, seed)] = r
        @printf("  [%d/%d] %.0f min  γ=%.3f p%d α=%.2f %s seed=%d  disc=%.3f\n",
                length(results), length(jobs), (time() - t0) / 60, γ, pm, α, name, seed, r.disc)
        flush(stdout)
    end
end
@printf("%d episodes in %.0f min\n", length(jobs), (time() - t0) / 60)

mkpath("outputs/discount_sweep")
csv_path = "outputs/discount_sweep/discount_sweep_e$(eval_model).csv"
open(csv_path, "w") do io
    println(io, "gamma,planning_model,eval_model,alpha,policy,seed,discounted_reward,profit,emission_cost,co2_tonnes,met_demand_pct,mines_opened,capex,explorations")
    for ((γ, pm, α, name, seed), r) in sort(collect(results), by=first)
        println(io, join([γ, pm, eval_model, α, name, seed, r.disc, r.profit, r.emis, r.co2, r.met, r.mines, r.capex, r.explores], ","))
    end
end

se(x) = std(x) / sqrt(length(x))
disc(γ, pm, α, name) = [results[(γ, pm, α, name, s)].disc for s in seeds]

println("\nMean discounted reward (compare within a row). Last columns: best POMCPOW − best heuristic, paired ± SE")
@printf("%-6s %-5s %-5s", "γ", "plan", "α")
for n in names; @printf(" %10s", first(replace(n, "POMCPOW " => "PO "), 10)); end
println("   best heuristic           Δ ± SE")
for γ in GAMMAS, pm in planning_models, α in alphas
    @printf("%-6.3f %-5d %-5.2f", γ, pm, α)
    for n in names; @printf(" %10.3f", mean(disc(γ, pm, α, n))); end
    bp = pomcpow_names[argmax([mean(disc(γ, pm, α, n)) for n in pomcpow_names])]
    bh = heuristics[argmax([mean(disc(γ, pm, α, h)) for h in heuristics])]
    Δ = disc(γ, pm, α, bp) .- disc(γ, pm, α, bh)
    @printf("   %-22s %+7.3f ± %.3f  (%s)\n", bh, mean(Δ), se(Δ), bp)
end

println("\nUndiscounted outcomes per γ (comparable across rows): mean profit \$M and mines opened")
@printf("%-6s", "γ"); for n in names; @printf(" %18s", first(n, 18)); end; println()
for γ in GAMMAS
    @printf("%-6.3f", γ)
    for n in names
        rs = [results[(γ, pm, α, n, s)] for pm in planning_models for α in alphas for s in seeds]
        @printf(" %10.0f / %4.2f", mean(getfield.(rs, :profit)) / 1e6, mean(getfield.(rs, :mines)))
    end
    println()
end
println("\nCSV: $csv_path")
