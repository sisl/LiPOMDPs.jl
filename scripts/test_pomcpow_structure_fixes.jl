"""
Structural fixes for POMCPOW, tested paired against TimingAware (TA) and OneStepLookahead-TA.

Two defects of the final_results_best_pomcpow.jl configuration, and the remedies tested here:
1. Rollouts are cut at `max_depth − depth` steps (BasicPOMCP's RolloutSimulator), so with
   max_depth = 25 and a 31-year horizon the planner never sees 2021–2025 while deciding in
   1995–2000.                                            → max_depth = time_horizon (31).
2. Leaf values are single stochastic TA rollouts over up to 30 years of GBM prices: return std
   ≈ 1–2 reward units for ~80 root actions sharing 10k samples, so the MaxQ root choice is the
   optimizer's curse (spurious explores, poor timing).  → certainty-equivalent leaf value
   (`mean_path_value`: TA followed on the expected price/demand path), and/or a proposer that
   keeps the TA action and "do nothing" in every node (`:timing_first`), and/or a visit-count
   final criterion.
Also: the default proposer draws from all 256 joint actions below the root (`:valid_random`).

  julia --project=. -t 80 scripts/test_pomcpow_structure_fixes.jl [n_seeds] [queries] [round1|round2]
Training seeds 1001+, cells (planning model, α) listed below, evaluation on historical prices.
round2: the root-value diagnostic (test_pomcpow_root_values.jl) showed that with k_action = 5 a
depth-1 node tries a new random action on almost every visit, so root Q values average over junk
continuations. round2 keeps the TA-first proposer and shrinks the widening constant so sparsely
visited nodes evaluate only the TA action / do-nothing, plus the visit-count final criterion.
"""

include(joinpath(@__DIR__, "pomcpow_tuning_harness.jl"))

n_seeds = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
q       = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10_000
stage   = length(ARGS) >= 3 ? ARGS[3] : "round1"
# "inmodel": evaluate in a GBM world (evaluation model 4 = planning model), to separate solver bias
# from price-model misspecification: if POMCPOW still opens earlier than TimingAware and loses when
# the world IS its model, the tree is biased; if it wins in-model, the loss on historical prices is
# the GBM model being wrong about 1995-2015, which the certainty-equivalent heuristic survives better.
eval_model = stage in ("inmodel", "inmodel3", "inmodel3b", "inmodel_scale", "inmodel_noexplore", "inmodel7") ? 4 : 6
seeds   = collect(1001:(1000 + n_seeds))
cells   = stage == "inmodel" ? [(4, 0.5), (4, 1.0)] :
          stage in ("inmodel3", "inmodel3b", "inmodel_scale", "inmodel_noexplore", "inmodel7") ? [(4, 0.5), (4, 1.0)] :
          stage == "round7" ? [(pm, α) for pm in parse.(Int, split(ARGS[4], ",")) for α in (0.5, 1.0)] :
          stage in ("gbm8", "gbm8tt") ? [(4, 0.5), (4, 1.0)] :
          stage == "oslroll" ? [(4, 0.5), (4, 1.0), (6, 0.5), (6, 1.0)] :
          stage == "round3b" ? [(4, 0.5), (4, 1.0), (6, 0.5), (6, 1.0)] :
          stage in ("round4", "round5") ? [(pm, α) for pm in parse.(Int, split(ARGS[4], ",")) for α in (0.5, 1.0)] :
          stage == "round3" ? [(4, 0.5), (4, 1.0), (6, 0.5), (6, 1.0)] :
          stage == "round3m" ? [(1, 0.5), (1, 1.0), (2, 0.5), (2, 1.0), (3, 0.5), (3, 1.0)] :
          [(4, 0.5), (4, 1.0), (6, 0.5), (6, 1.0), (1, 0.25)]
outdir  = "outputs/structure_fixes"; mkpath(outdir)

pc(; kw...) = p -> make_pomcpow(p; rollout=:timing, ucb_c=4.0, tree_queries=q, kw...)
pomcpow_configs = stage == "round1" ? [
    "PC base d25"        => pc(max_depth=25),
    "PC d31"             => pc(max_depth=31),
    "PC d31 valid"       => pc(max_depth=31, next_action=:valid_random),
    "PC d31 TAfirst"     => pc(max_depth=31, next_action=:timing_first),
    "PC d31 MP"          => pc(max_depth=31, rollout=:mean_path),
    "PC d31 MP TAfirst"  => pc(max_depth=31, rollout=:mean_path, next_action=:timing_first),
    "PC d31 MP visits"   => pc(max_depth=31, rollout=:mean_path, final_criterion=:visits),
] : stage == "round2" ? [
    "PC base d25"           => pc(max_depth=25),
    "PC d31 MP TAfirst"     => pc(max_depth=31, rollout=:mean_path, next_action=:timing_first),
    "PC d31 MP TAfirst k1"  => pc(max_depth=31, rollout=:mean_path, next_action=:timing_first, k_action=1.0),
    "PC d31 MP TAfirst k2"  => pc(max_depth=31, rollout=:mean_path, next_action=:timing_first, k_action=2.0),
    "PC d31 MP TAfirst k1 visits" => pc(max_depth=31, rollout=:mean_path, next_action=:timing_first, k_action=1.0, final_criterion=:visits),
    "PC d31 TAfirst k1"     => pc(max_depth=31, next_action=:timing_first, k_action=1.0),
    "PC d31 MP TAtree"      => pc(max_depth=31, rollout=:mean_path, next_action=:timing_tree),
    "PC d31 MP TAtree visits" => pc(max_depth=31, rollout=:mean_path, next_action=:timing_tree, final_criterion=:visits),
    "PC d31 TAtree"         => pc(max_depth=31, next_action=:timing_tree),
] : stage == "inmodel" ? [
    "PC base d25"           => pc(max_depth=25),
    "PC d31 MP"             => pc(max_depth=31, rollout=:mean_path),
    "PC d31 MP visits"      => pc(max_depth=31, rollout=:mean_path, final_criterion=:visits),
    "PC d31 MP TAtree"      => pc(max_depth=31, rollout=:mean_path, next_action=:timing_tree),
] : stage in ("round3", "round3m", "inmodel3") ? [
    # Single-site root action abstraction on the timing tree (timing_tree_single_action_generator), and
    # observation widening. The root dump showed "explore i" absorbing most visits: with k_obs=3,
    # α_obs=0.3 each observation child holds ~1 particle, so an exploring branch behaves as if the
    # deposit were revealed exactly (perfect-information VOI) — a POMCPOW artifact with continuous
    # observations. Fewer observation children (k_obs=1, α_obs=0.1) keep many particles per child.
    "PC base d25"                 => pc(max_depth=25),
    "PC d31 MP TAtree1"           => pc(max_depth=31, rollout=:mean_path, next_action=:timing_tree_single),
    "PC d31 MP TAtree1 ko1 ao.1"  => pc(max_depth=31, rollout=:mean_path, next_action=:timing_tree_single, k_observation=1.0, alpha_observation=0.1),
    "PC d31 MP TAtree1 ko2 ao.2"  => pc(max_depth=31, rollout=:mean_path, next_action=:timing_tree_single, k_observation=2.0, alpha_observation=0.2),
    "PC d31 MP TAtree1 c1 ko1 ao.1" => pc(max_depth=31, rollout=:mean_path, next_action=:timing_tree_single, k_observation=1.0, alpha_observation=0.1, ucb_c=1.0),
    "PC d31 MP TAtree1 cheap"     => pc(max_depth=31, rollout=:mean_path, next_action=:timing_tree_single, final_criterion=:cheapest),
    "PC d31 TAtree1 ko1 ao.1"     => pc(max_depth=31, next_action=:timing_tree_single, k_observation=1.0, alpha_observation=0.1),
    "PC d31 MP ko1 ao.1"          => pc(max_depth=31, rollout=:mean_path, k_observation=1.0, alpha_observation=0.1),
] : stage in ("round3b", "inmodel3b") ? [
    # observation widening alone, on the unchanged final_results configuration (stochastic TA rollout,
    # random proposer, c=4): the minimal change if fewer observation children is the lever
    "PC base d25"                 => pc(max_depth=25),
    "PC base d25 ko1 ao.1"        => pc(max_depth=25, k_observation=1.0, alpha_observation=0.1),
    "PC base d25 ko2 ao.2"        => pc(max_depth=25, k_observation=2.0, alpha_observation=0.2),
    "PC base d31 ko1 ao.1"        => pc(max_depth=31, k_observation=1.0, alpha_observation=0.1),
    "PC base d25 ko1 ao.1 cheap"  => pc(max_depth=25, k_observation=1.0, alpha_observation=0.1, final_criterion=:cheapest),
] : stage == "round4" ? [
    # final candidates on the unchanged configuration: horizon-complete rollouts (d31) and the
    # CheapestNearBest root tie-break (drop explorations that change the root value by < 2%)
    "PC base d25"           => pc(max_depth=25),
    "PC base d31"           => pc(max_depth=31),
    "PC base d31 cheap"     => pc(max_depth=31, final_criterion=:cheapest),
    "PC base d25 cheap"     => pc(max_depth=25, final_criterion=:cheapest),
] : stage == "round5" ? [
    # Anna's request: tree depth 27 on every planning model (25 and 31 were indistinguishable)
    "PC base d25"           => pc(max_depth=25),
    "PC base d27"           => pc(max_depth=27),
    "PC base d27 cheap"     => pc(max_depth=27, final_criterion=:cheapest),
] : stage == "inmodel_scale" ? [
    # Does POMCPOW converge to ≥ TimingAware in its OWN GBM world as the search budget grows? A correct
    # planner must; if 100k queries still trail TA, the shortfall is structural rather than compute.
    "PC base 10k"           => pc(max_depth=25, tree_queries=10_000),
    "PC base 30k"           => pc(max_depth=25, tree_queries=30_000),
    "PC base 100k"          => pc(max_depth=25, tree_queries=100_000),
] : stage == "inmodel_noexplore" ? [
    # Mechanism test: if POMCPOW matches TA in its own GBM world once it cannot explore, the loss comes
    # from how the tree values exploration branches (one-particle observation children), not from search noise.
    "PC base 10k"               => pc(max_depth=25),
    "PC base 10k no-explore"    => pc(max_depth=25, next_action=:sparse_explore, explore_p=0.0),
] : stage in ("round7", "inmodel7") ? [
    # robust root rule against the open/wait ratchet (FewerOpensNearBest)
    "PC base"                   => pc(max_depth=25),
    "PC base fewer-opens 5%"    => pc(max_depth=25, final_criterion=:fewer_opens, open_tol=0.05),
    "PC base fewer-opens 10%"   => pc(max_depth=25, final_criterion=:fewer_opens, open_tol=0.10),
    "PC base fewer-opens 20%"   => pc(max_depth=25, final_criterion=:fewer_opens, open_tol=0.20),
] : stage == "gbm8" ? [
    # quick GBM check on the final model (recovery 0.20/0.25): queries and UCB
    "PC base 10k c4"            => pc(max_depth=25),
    "PC 30k c4"                 => pc(max_depth=25, tree_queries=30_000),
    "PC 10k c2"                 => pc(max_depth=25, ucb_c=2.0),
] : stage == "gbm8tt" ? [
    # timing-tree variants on the final model (same seeds as gbm8, so directly comparable)
    "PC TAtree c4"              => pc(max_depth=25, next_action=:timing_tree),
    "PC TAtree1 MP d31 c4"      => pc(max_depth=31, rollout=:mean_path, next_action=:timing_tree_single),
] : stage == "oslroll" ? [
    # how much OneStepLookahead's rollout budget matters (POMCPOW and TA are the references listed below)
    "PC base 10k c4"            => pc(max_depth=25),
    "OSL-TA 10 rollouts"        => p -> OneStepLookaheadPolicy(p, n_rollouts=10, debug=false, rollout_policy=TimingAwarePolicy(p)),
    "OSL-TA 30 rollouts"        => p -> OneStepLookaheadPolicy(p, n_rollouts=30, debug=false, rollout_policy=TimingAwarePolicy(p)),
] : error("unknown stage $stage")
configs = vcat(pomcpow_configs, [
    "TA"                 => p -> TimingAwarePolicy(p),
    "OSL-TA"             => p -> OneStepLookaheadPolicy(p, n_rollouts=5, debug=false, rollout_policy=TimingAwarePolicy(p)),   # 5 rollouts as in the 2026-10-21 finals
])
names = first.(configs)

# run_scenario plus opening years and wall time
function run_detail(policy, planning, evaluation, updater, seed)
    t0 = time()
    Random.seed!(seed)
    s0, b0 = create_initial_belief(planning)
    b, s = b0, s0
    disc = 0.0; explores = 0; opens = fill(0, planning.n_deposits)
    for t in 1:planning.time_horizon
        a = action(policy, b)
        explores += count(==(1), a.type)
        a_exec = LiPOMDPs.effective_action(evaluation, s.have_opened, a)
        for i in 1:planning.n_deposits
            a_exec.type[i] == 2 && !s.have_opened[i] && (opens[i] = t)
        end
        Random.seed!(hash((seed, t)))
        env = MersenneTwister(hash((seed, t, :env)))
        sp, o, r = gen(evaluation, s, a, env)
        disc += evaluation.γ^(t - 1) * r
        isterminal(evaluation, sp) && break
        b = update(updater, b, a, o)
        s = sp
    end
    return (disc=disc, explores=explores, opens=opens, mines=count(>(0), opens), secs=time()-t0)
end

jobs = [(pm, α, n, s) for n in names for (pm, α) in cells for s in seeds]
res = Dict(); lk = ReentrantLock(); t0 = time()
println("$(length(jobs)) episodes on $(Threads.nthreads()) threads, queries=$q, seeds=$seeds")
Threads.@threads :greedy for (pm, α, n, s) in jobs
    P = LiPOMDP(price_model_type=pm, alpha=α); E = LiPOMDP(price_model_type=eval_model, alpha=α)
    r = run_detail(Dict(configs)[n](P), P, E, LiBeliefUpdater(P=P), s)
    lock(lk) do; res[(pm, α, n, s)] = r; end
end
@printf("done in %.0f min\n", (time()-t0)/60)

open(joinpath(outdir, "structure_fixes_$(stage)$(stage in ("round4", "round5", "round7") ? "_p" * replace(ARGS[4], "," => "") : "")_q$(q)_n$(n_seeds).csv"), "w") do io
    println(io, "planning_model,alpha,policy,seed,discounted_reward,explorations,mines_opened,open_years,secs")
    for ((pm, α, n, s), r) in sort(collect(res), by=first)
        println(io, join([pm, α, "\"$n\"", s, r.disc, r.explores, r.mines, "\"$(r.opens)\"", round(r.secs, digits=1)], ","))
    end
end

se(d) = std(d) / sqrt(length(d))
for (pm, α) in cells
    println("\nplanning p$pm, α=$α  (seeds $(first(seeds))–$(last(seeds)); evaluation model $eval_model)")
    @printf("  %-20s %7s %14s %14s %6s %6s  %s\n", "config", "disc", "Δ vs TA", "Δ vs OSL-TA", "expl", "mines", "mean open year per site (0 = never)")
    ta  = [res[(pm, α, "TA", s)].disc for s in seeds]
    osl = [res[(pm, α, "OSL-TA", s)].disc for s in seeds]
    for n in names
        x = [res[(pm, α, n, s)].disc for s in seeds]
        oy = mean(hcat((res[(pm, α, n, s)].opens for s in seeds)...), dims=2)
        @printf("  %-20s %7.3f %+7.3f ±%5.3f %+7.3f ±%5.3f %6.1f %6.2f  %s  (%.0fs/ep)\n", n, mean(x),
                mean(x .- ta), se(x .- ta), mean(x .- osl), se(x .- osl),
                mean(res[(pm, α, n, s)].explores for s in seeds), mean(res[(pm, α, n, s)].mines for s in seeds),
                join([@sprintf("%4.1f", v) for v in oy], " "), mean(res[(pm, α, n, s)].secs for s in seeds))
    end
end
println("\nPOOLED over all cells × seeds (paired):")
for n in names
    x  = [res[(pm, α, n, s)].disc for (pm, α) in cells for s in seeds]
    ta = [res[(pm, α, "TA", s)].disc for (pm, α) in cells for s in seeds]
    os = [res[(pm, α, "OSL-TA", s)].disc for (pm, α) in cells for s in seeds]
    @printf("  %-20s mean %.3f   Δ vs TA %+.3f ± %.3f   Δ vs OSL-TA %+.3f ± %.3f\n", n, mean(x),
            mean(x .- ta), se(x .- ta), mean(x .- os), se(x .- os))
end
