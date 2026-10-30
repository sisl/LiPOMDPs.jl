# UCB constant c ∈ {1, 2, 4} for POMCPOW WITHOUT exploration controls (the final_results_best_pomcpow.jl setup).
# TRAINING seeds 1001-1025, historical (6) & GBM (4) planning, α ∈ {0.5, 1}, evaluated on historical prices.
# Selection rule (fixed before running): highest mean discounted reward pooled over all 100 training episodes;
# keep c = 1 unless the winner beats c = 1 by more than one standard error of the paired difference.
include("/home/edmondsa/LiPOMDPs.jl/scripts/pomcpow_tuning_harness.jl")
seeds = 1001:1025; q = 10_000; cs = [4, 6, 8, 10, 20]
cfgs = vcat(["PC c=$c" => (p -> make_pomcpow(p; rollout=:timing, tree_queries=q, ucb_c=Float64(c))) for c in cs],
            Pair{String,Function}[])
names = first.(cfgs)
cells = [(pm, α) for pm in (6, 4) for α in (0.5, 1.0)]
jobs = [(pm, α, n, s) for n in names for (pm, α) in cells for s in seeds]
res = Dict(); lk = ReentrantLock(); t0 = time()
Threads.@threads :greedy for (pm, α, n, s) in jobs
    P = LiPOMDP(price_model_type=pm, alpha=α); E = LiPOMDP(price_model_type=6, alpha=α)
    r = run_scenario(Dict(cfgs)[n](P), P, E, LiBeliefUpdater(P=P), s)
    lock(lk) do; res[(pm, α, n, s)] = r; end
end
@printf("%d episodes in %.0f min\n", length(jobs), (time()-t0)/60)
open("/home/edmondsa/LiPOMDPs.jl/outputs/ucb_check/ucb_high_nocontrols.csv", "w") do io
    println(io, "planning_model,alpha,policy,seed,discounted_reward,profit,explorations,mines_opened")
    for ((pm, α, n, s), r) in sort(collect(res), by=first)
        println(io, join([pm, α, "\"$n\"", s, r.disc, r.profit, r.explores, r.mines], ","))
    end
end
v(n) = [res[(pm, α, n, s)].disc for (pm, α) in cells for s in seeds]          # pooled, paired order
for (pm, α) in cells
    println("\nplanning p$pm, α=$α  (training seeds 1001–1025)")
    for n in names
        x = [res[(pm, α, n, s)].disc for s in seeds]
        d = x .- [res[(pm, α, "PC c=4", s)].disc for s in seeds]
        @printf("  %-8s disc %.3f   − PC c=4 %+.3f ± %.3f   expl %.1f\n", n, mean(x),
                mean(d), std(d)/5, mean(res[(pm, α, n, s)].explores for s in seeds))
    end
end
println("\nPOOLED over 4 cells × 25 seeds:")
base = v("PC c=4")
best_c, best_m = 4, mean(base)
for c in cs
    x = v("PC c=$c"); d = x .- base
    @printf("  c=%d  mean disc %.3f   − c=4 %+.3f ± %.3f\n", c, mean(x), mean(d), std(d)/sqrt(length(d)))
    if mean(x) > best_m && mean(d) > std(d)/sqrt(length(d))
        global best_c, best_m = c, mean(x)
    end
end
println("SELECTED c = $best_c")
write("/home/edmondsa/LiPOMDPs.jl/outputs/ucb_check/best_c_high.txt", string(best_c))
