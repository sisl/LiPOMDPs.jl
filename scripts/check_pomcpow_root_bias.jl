"""
Are POMCPOW's root Q-values biased? Brute-force check of the planner's own arithmetic.

At a belief reached by following TimingAware for `t0 - 1` years (GBM planning, α), run POMCPOW with
the final configuration and read its root Q for (i) the action it chose, (ii) "do nothing" and
(iii) TimingAware's action. Then estimate the SAME quantities without a tree: for each action, draw
N states from the belief, apply the action, and follow TimingAware (fully observed, like the tree's
rollout) to the horizon with the planning model's own stochastic dynamics. Common random numbers
across actions. The tree's Q should be ≥ the Monte-Carlo value (the tree optimises below the root),
but the RANKING of actions should agree; if the tree prefers "open now" while the brute-force
estimate prefers waiting by more than its standard error, the tree's root choice is biased.

  julia --project=. -t 8 scripts/check_pomcpow_root_bias.jl [planning_model] [alpha] [seed] [t0] [N]
"""

include(joinpath(@__DIR__, "pomcpow_tuning_harness.jl"))

pm   = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
α    = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.0
seed = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1001
t0   = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 12
N    = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : 2000

P = LiPOMDP(price_model_type=pm, alpha=α); E = LiPOMDP(price_model_type=6, alpha=α)
up = LiBeliefUpdater(P=P); ta = TimingAwarePolicy(P)
desc(a) = join(["$(i):$(("·","E","O","R")[t+1])" for (i, t) in enumerate(a.type)], " ")

# brute-force value of "a now, then TimingAware (fully observed) to the horizon", under the planning model
function mc_value(b, a, N)
    vals = Vector{Float64}(undef, N)
    Threads.@threads for k in 1:N
        rng = MersenneTwister(hash((seed, t0, k)))        # same scenario k for every action
        s = rand(rng, b)
        total, disc = 0.0, 1.0
        act = a
        while !isterminal(P, s)
            total += disc * reward(P, s, act)
            s = LiPOMDPs.sample_transition(P, s, act, rng)
            disc *= P.γ
            act = action(ta, s)
        end
        vals[k] = total
    end
    return mean(vals), std(vals) / sqrt(N), vals
end

Random.seed!(seed)
s, b = create_initial_belief(P)
for t in 1:(t0 - 1)
    a = action(ta, b)
    Random.seed!(hash((seed, t))); env = MersenneTwister(hash((seed, t, :env)))
    sp, o, r = gen(E, s, a, env)
    global b = update(up, b, a, o); global s = sp
end
println("planning p$pm α=$α seed=$seed, belief at t=$t0 (year $(P.start_year + t0 - 1)): price $(round(mean(b.price_distribution))), demand $(round(mean(b.demand_distribution))), open=$(b.have_opened)")

for (name, pl) in ["final config (TA rollout, c=4, d25)" => make_pomcpow(P; rollout=:timing, ucb_c=4.0, max_depth=25),
                   "same, 30k queries"                   => make_pomcpow(P; rollout=:timing, ucb_c=4.0, max_depth=25, tree_queries=30_000),
                   "final config + mean-path prior n0=10" => make_pomcpow(P; rollout=:timing, ucb_c=4.0, max_depth=25, init=:mean_path, init_n=10),
                   "final config + mean-path prior n0=30" => make_pomcpow(P; rollout=:timing, ucb_c=4.0, max_depth=25, init=:mean_path, init_n=30)]
    a_tree, info = action_info(pl, b; tree_in_info=true)
    tree = info[:tree]; root = tree.tried[1]
    q(a) = (k = findfirst(n -> tree.a_labels[n] == a, root); k === nothing ? (NaN, 0) : (tree.v[root[k]], tree.n[root[k]]))
    a_ta = action(ta, b)
    cands = unique([a_tree, DO_NOTHING, a_ta])
    println("\n== $name: chose ", desc(a_tree))
    @printf("  %-22s %10s %6s   %14s   %s\n", "action", "tree Q", "N", "brute-force V", "V − V(do nothing) ± SE (paired)")
    base = nothing
    for a in cands
        v, se, vals = mc_value(b, a, N)
        base === nothing && (base = vals)
        d = vals .- base
        tq, tn = q(a)
        @printf("  %-22s %10.3f %6d   %8.3f ±%.3f   %+8.3f ± %.3f\n", desc(a), tq, tn, v, se, mean(d), std(d) / sqrt(N))
    end
    # also the single best "open one site now" by brute force, in case the tree never tried it
    for i in 1:P.n_deposits
        b.have_opened[i] && continue
        a = Action(site=collect(1:4), type=[j == i ? 2 : 0 for j in 1:4])
        a in cands && continue
        LiPOMDPs.effective_action(P, b.have_opened, a) == a || continue
        v, se, vals = mc_value(b, a, N); tq, tn = q(a)
        @printf("  %-22s %10.3f %6d   %8.3f ±%.3f   %+8.3f ± %.3f\n", desc(a), tq, tn, v, se, mean(vals .- base), std(vals .- base) / sqrt(N))
    end
end
