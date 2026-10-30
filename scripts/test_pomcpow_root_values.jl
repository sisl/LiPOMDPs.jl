"""
How noisy are POMCPOW's root Q estimates? For one scenario, print the top root actions (Q, visits)
of the current configuration (stochastic TA rollouts, depth 25) next to the certainty-equivalent
leaf estimate (mean_path_value, depth 31), at the initial belief and after following TA for a
few years. Explains the "optimizer's curse" diagnosis: if the Q spread among the top actions is
smaller than the standard error of each estimate, MaxQ is picking by noise.

  julia --project=. -t 4 scripts/test_pomcpow_root_values.jl [planning_model] [alpha] [seed]
"""

include(joinpath(@__DIR__, "pomcpow_tuning_harness.jl"))

pm   = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4
α    = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 1.0
seed = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1001

P = LiPOMDP(price_model_type=pm, alpha=α); E = LiPOMDP(price_model_type=6, alpha=α)
up = LiBeliefUpdater(P=P); ta = TimingAwarePolicy(P)
planners = length(ARGS) >= 4 && ARGS[4] == "round2" ? [
    "mean-path, d31, TAfirst, k=5"  => make_pomcpow(P; rollout=:mean_path, ucb_c=4.0, max_depth=31, next_action=:timing_first),
    "mean-path, d31, TAfirst, k=1"  => make_pomcpow(P; rollout=:mean_path, ucb_c=4.0, max_depth=31, next_action=:timing_first, k_action=1.0),
    "mean-path, d31, TAfirst, k=1, 30k" => make_pomcpow(P; rollout=:mean_path, ucb_c=4.0, max_depth=31, next_action=:timing_first, k_action=1.0, tree_queries=30_000),
] : [
    "base: TA rollout, d25"       => make_pomcpow(P; rollout=:timing, ucb_c=4.0, max_depth=25),
    "mean-path leaf, d31"         => make_pomcpow(P; rollout=:mean_path, ucb_c=4.0, max_depth=31),
    "mean-path leaf, d31, TAfirst"=> make_pomcpow(P; rollout=:mean_path, ucb_c=4.0, max_depth=31, next_action=:timing_first),
]
desc(a) = join(["$(i):$(("·","E","O","R")[t+1])" for (i, t) in enumerate(a.type)], " ")

Random.seed!(seed)
s, b = create_initial_belief(P)
for t in 1:P.time_horizon
    if t in (1, 6, 12, 18)
        println("\n===== t=$t (year $(P.start_year + t - 1))  belief price $(round(mean(b.price_distribution))), demand $(round(mean(b.demand_distribution)))  open=$(b.have_opened)")
        println("TA action on belief: ", desc(action(ta, b)))
        for (name, pl) in planners
            t0 = time()
            a, info = action_info(pl, b; tree_in_info=true)
            tree = info[:tree]
            root = tree.tried[1]
            order = sortperm(tree.v[root], rev=true)
            println("  -- $name  ($(round(time()-t0, digits=1))s, $(length(root)) root actions)  chose ", desc(a))
            for k in order[1:min(8, end)]
                node = root[k]
                @printf("     Q=%8.3f  N=%5d  %s\n", tree.v[node], tree.n[node], desc(tree.a_labels[node]))
            end
            # the two reference actions
            for (lbl, ref) in (("do nothing", DO_NOTHING), ("TA action", action(ta, b)))
                k = findfirst(n -> tree.a_labels[n] == ref, root)
                k === nothing ? println("     $lbl: not tried") :
                    @printf("     %-10s Q=%8.3f  N=%5d\n", lbl, tree.v[root[k]], tree.n[root[k]])
            end
        end
    end
    a = action(ta, b)                                   # follow TA so all planners see the same beliefs
    Random.seed!(hash((seed, t))); env = MersenneTwister(hash((seed, t, :env)))
    sp, o, r = gen(E, s, a, env)
    isterminal(E, sp) && break
    global b = update(up, b, a, o); global s = sp
end
