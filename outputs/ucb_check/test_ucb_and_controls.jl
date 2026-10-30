# UCB constant (with exploration controls) and controls on/off at c=1. TRAINING seeds 1001-1025,
# historical & GBM planning, evaluated on historical (the final-results setup).
include("/home/edmondsa/LiPOMDPs.jl/scripts/pomcpow_tuning_harness.jl")
seeds = 1001:1025; q = 10_000
pc(c; ctl=true) = p -> ctl ? make_pomcpow(p; rollout=:timing, tree_queries=q, ucb_c=c, next_action=:info_gated, final_criterion=:cheapest) :
                             make_pomcpow(p; rollout=:timing, tree_queries=q, ucb_c=c)
cfgs = ["PC c=1, no controls" => pc(1.0; ctl=false),
        "PC c=0.5 + controls" => pc(0.5),
        "PC c=1 + controls"   => pc(1.0),
        "PC c=2 + controls"   => pc(2.0),
        "PC c=4 + controls"   => pc(4.0),
        "OSL-TA"              => p -> OneStepLookaheadPolicy(p, n_rollouts=10, rollout_policy=TimingAwarePolicy(p)),
        "TA"                  => p -> TimingAwarePolicy(p)]
names = first.(cfgs)
jobs = [(pm, α, n, s) for n in names for pm in (6, 4) for α in (0.5, 1.0) for s in seeds]
res = Dict(); lk = ReentrantLock(); t0 = time()
Threads.@threads :greedy for (pm, α, n, s) in jobs
    P = LiPOMDP(price_model_type=pm, alpha=α); E = LiPOMDP(price_model_type=6, alpha=α)
    r = run_scenario(Dict(cfgs)[n](P), P, E, LiBeliefUpdater(P=P), s)
    lock(lk) do; res[(pm, α, n, s)] = r; end
end
@printf("%d episodes in %.0f min\n", length(jobs), (time()-t0)/60)
open("/home/edmondsa/LiPOMDPs.jl/outputs/ucb_check/ucb_and_controls.csv", "w") do io
    println(io, "planning_model,alpha,policy,seed,discounted_reward,profit,explorations,mines_opened")
    for ((pm, α, n, s), r) in sort(collect(res), by=first)
        println(io, join([pm, α, "\"$n\"", s, r.disc, r.profit, r.explores, r.mines], ","))
    end
end
for pm in (6, 4), α in (0.5, 1.0)
    println("\nplanning p$pm, α=$α  (training seeds 1001–1025; paired differences ± SE)")
    for n in names
        v = [res[(pm, α, n, s)].disc for s in seeds]
        d = v .- [res[(pm, α, "OSL-TA", s)].disc for s in seeds]
        g = v .- [res[(pm, α, "PC c=1, no controls", s)].disc for s in seeds]
        @printf("  %-22s disc %.3f  − OSL-TA %+.3f ± %.3f  − PC c=1 no ctl %+.3f ± %.3f  expl %.1f\n", n, mean(v),
                mean(d), std(d)/5, mean(g), std(g)/5, mean(res[(pm, α, n, s)].explores for s in seeds))
    end
end
