"""
Model levers for information value (2026-10-21): does POMCPOW gain on TimingAware / OneStepLookahead-TA
when recovery is more uncertain and/or the capital budget forces a choice between sites?

Variants of the model (applied to BOTH planning and evaluation POMDPs):
  current            recovery sd 0.10, budget \$5B
  rec15              recovery sd 0.15, budget \$5B
  rec15_20           recovery sd 0.15 (DLE) / 0.20 (hard rock), budget \$5B
  rec10_b3           recovery sd 0.10, budget \$3B
  rec15_b3           recovery sd 0.15, budget \$3B
Paired scenarios (training seeds), POMCPOW in the final configuration vs TA and OSL-TA.

  julia --project=. -t 16 scripts/test_model_levers.jl [n_seeds] [planning models e.g. 1,4,6] [alphas]
"""

include(joinpath(@__DIR__, "pomcpow_tuning_harness.jl"))

n_seeds = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
confirm = length(ARGS) >= 4 && ARGS[4] == "confirm"     # fresh seeds 1011+, only current vs rec15_20
sens    = length(ARGS) >= 4 && ARGS[4] == "sens"        # sensitivity: rec15_20 vs wider priors (0.20/0.25, 0.25/0.30)
sens2   = length(ARGS) >= 4 && ARGS[4] == "sens2"       # widest well-formed priors (both Beta parameters ≥ 1): 0.15/0.22, 0.17/0.22
pms     = length(ARGS) >= 2 ? parse.(Int, split(ARGS[2], ",")) : [1, 4, 6]
alphas  = length(ARGS) >= 3 ? parse.(Float64, split(ARGS[3], ",")) : [0.5, 1.0]
seeds   = (confirm || sens || sens2) ? collect(1011:(1010 + n_seeds)) : collect(1001:(1000 + n_seeds))
outdir  = "outputs/model_levers"; mkpath(outdir)

variants = [
    "current"  => (;),
    "rec15"    => (recovery_uncertainty = fill(0.15, 4),),
    "rec15_20" => (recovery_uncertainty = [0.15, 0.15, 0.20, 0.20],),
    "rec10_b3" => (capital_budget = 3.0e9,),
    "rec15_b3" => (recovery_uncertainty = fill(0.15, 4), capital_budget = 3.0e9),
]
confirm && (variants = variants[[1, 3]])
sens && (variants = [
    "rec15_20" => (recovery_uncertainty = [0.15, 0.15, 0.20, 0.20],),
    "rec20_25" => (recovery_uncertainty = [0.20, 0.20, 0.25, 0.25],),
    "rec25_30" => (recovery_uncertainty = [0.25, 0.25, 0.30, 0.30],),
])
# all three priors in one run, side by side on the same seeds
sens2 && (variants = [
    "rec15_20" => (recovery_uncertainty = [0.15, 0.15, 0.20, 0.20],),
    "rec15_22" => (recovery_uncertainty = [0.15, 0.15, 0.22, 0.22],),
    "rec17_22" => (recovery_uncertainty = [0.17, 0.17, 0.22, 0.22],),
])
# optional 5th argument: comma-separated variant names to run (split one study across machines)
length(ARGS) >= 5 && (variants = [v for v in variants if first(v) in split(ARGS[5], ",")])
policies = [
    "POMCPOW" => P -> make_pomcpow(P; rollout=:timing, ucb_c=4.0, max_depth=25, tree_queries=10_000),
    "TA"      => P -> TimingAwarePolicy(P),
    "OSL-TA"  => P -> OneStepLookaheadPolicy(P, n_rollouts=10, debug=false, rollout_policy=TimingAwarePolicy(P)),
]

jobs = [(v, pm, α, n, s) for (v, _) in variants for pm in pms for α in alphas for (n, _) in policies for s in seeds]
res = Dict(); lk = ReentrantLock(); t0 = time()
println("$(length(jobs)) episodes on $(Threads.nthreads()) threads")
Threads.@threads :greedy for (v, pm, α, n, s) in jobs
    kw = Dict(variants)[v]
    P = LiPOMDP(; price_model_type=pm, alpha=α, kw...); E = LiPOMDP(; price_model_type=6, alpha=α, kw...)
    r = run_scenario(Dict(policies)[n](P), P, E, LiBeliefUpdater(P=P), s)
    lock(lk) do; res[(v, pm, α, n, s)] = r; end
end
@printf("done in %.0f min\n", (time() - t0) / 60)

open(joinpath(outdir, "model_levers$(confirm ? "_confirm" : sens ? "_sens" : sens2 ? "_sens2" : "")_n$(n_seeds).csv"), "w") do io
    println(io, "variant,planning_model,alpha,policy,seed,discounted_reward,profit,explorations,mines_opened")
    for ((v, pm, α, n, s), r) in sort(collect(res), by=first)
        println(io, join([v, pm, α, n, s, r.disc, r.profit, r.explores, r.mines], ","))
    end
end

se(d) = std(d) / sqrt(length(d))
for (v, _) in variants
    println("\n===== variant $v")
    @printf("  %-8s %-10s %7s %7s %7s   %16s %16s   %s\n", "cell", "", "POMCPOW", "TA", "OSL-TA", "PC−TA ± SE", "PC−OSL ± SE", "expl PC/OSL")
    for pm in pms, α in alphas
        x = [res[(v, pm, α, "POMCPOW", s)].disc for s in seeds]; ta = [res[(v, pm, α, "TA", s)].disc for s in seeds]
        os = [res[(v, pm, α, "OSL-TA", s)].disc for s in seeds]
        @printf("  p%d α=%-4s %10s %7.3f %7.3f %7.3f   %+8.3f ± %.3f %+8.3f ± %.3f   %.0f/%.0f\n", pm, α, "", mean(x), mean(ta), mean(os),
                mean(x .- ta), se(x .- ta), mean(x .- os), se(x .- os),
                mean(res[(v, pm, α, "POMCPOW", s)].explores for s in seeds), mean(res[(v, pm, α, "OSL-TA", s)].explores for s in seeds))
    end
    x = [res[(v, pm, α, "POMCPOW", s)].disc for pm in pms for α in alphas for s in seeds]
    ta = [res[(v, pm, α, "TA", s)].disc for pm in pms for α in alphas for s in seeds]
    os = [res[(v, pm, α, "OSL-TA", s)].disc for pm in pms for α in alphas for s in seeds]
    @printf("  POOLED: PC−TA %+.3f ± %.3f   PC−OSL-TA %+.3f ± %.3f\n", mean(x .- ta), se(x .- ta), mean(x .- os), se(x .- os))
end
