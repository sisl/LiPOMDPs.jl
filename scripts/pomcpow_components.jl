"""
POMCPOW components shared by scripts/pomcpow_tuning_harness.jl and scripts/final_results_best_pomcpow.jl:
- mean_path_estimator: certainty-equivalent leaf value (TimingAwarePolicy on the expected path);
- valid_random_action_generator / timing_first_action_generator: progressive-widening proposers.
"""

using LiPOMDPs, POMDPs, POMCPOW, Random

const DO_NOTHING = Action(site=[1, 2, 3, 4], type=[0, 0, 0, 0])

"""
POMCPOW leaf value estimate: `mean_path_value` of the TimingAwarePolicy from the leaf state to the
end of the horizon (BasicPOMCP calls a Function estimator as `f(pomdp, s, h, steps)`).
"""
function mean_path_estimator(pomdp::LiPOMDP)
    ta = TimingAwarePolicy(pomdp)
    # Always value the leaf to the end of the horizon (`mean_path_value` stops at the terminal state):
    # `steps` = max_depth − depth is BasicPOMCP's rollout cap, which would hide the late years again.
    return (p, s, h, steps) -> mean_path_value(p, ta, s, p.time_horizon)
end

"""
Action proposer that draws uniformly from the actions that are VALID for a state sampled from the
node belief (`actions(p, s)`: no opening/exploring an open mine, no over-budget opens). POMCPOW's
default `RandomActionGenerator` does this only at the root; at every deeper node it calls
`actions(p, ::StateBelief)`, which falls back to all 256 joint actions, so most of the widening
budget there goes to actions that are no-ops in effect but distinct tree nodes.
"""
function valid_random_action_generator(pomdp::LiPOMDP; rng::AbstractRNG=MersenneTwister(11))
    return (p, b, h) -> rand(rng, actions(p, rand(rng, b)))
end

"""
Informed progressive widening: the first action tried at a node is the TimingAwarePolicy action
for a state sampled from the node belief, the second is "do nothing", the rest are valid random
actions (see valid_random_action_generator). Guarantees the rollout policy's own action and
waiting are always among the candidates, so sparsely visited nodes default to sensible choices.
"""
function timing_first_action_generator(pomdp::LiPOMDP; rng::AbstractRNG=MersenneTwister(11))
    ta = TimingAwarePolicy(pomdp)
    return function (p, b, h)
        s = rand(rng, b)
        tried = h.tree.tried[h.node]
        if isempty(tried)
            return action(ta, s)
        elseif length(tried) == 1 && h.tree.a_labels[tried[1]] != DO_NOTHING
            return DO_NOTHING
        else
            return rand(rng, actions(p, s))
        end
    end
end


"""
Weighted mean of the particles in a POMCPOW node belief (deposits, price, demand, recovery);
status flags are taken from the first particle (they follow the action history, not the sample).
"""
function node_mean_state(b::POMCPOW.StateBelief)
    d = b.sr_belief.dist
    w = diff(vcat(0.0, d.cdf)); w ./= sum(w)
    ss = first.(d.items)
    s1 = ss[1]
    return State(
        deposit        = sum(w[k] .* ss[k].deposit for k in eachindex(ss)),
        time           = s1.time,
        price          = sum(w[k] * ss[k].price for k in eachindex(ss)),
        company_demand = sum(w[k] * ss[k].company_demand for k in eachindex(ss)),
        total_mined    = s1.total_mined,
        have_opened    = copy(s1.have_opened),
        is_depleted    = copy(s1.is_depleted),
        restored_mines = copy(s1.restored_mines),
        recovery       = isempty(s1.recovery) ? Float64[] : sum(w[k] .* ss[k].recovery for k in eachindex(ss)),
    )
end

"""
Timing-tree proposer. At the root: the TimingAware action for a sampled state, "do nothing", then
valid random actions (every root action is still a candidate). Below the root the tree only
considers {TimingAware action on the node's mean particle state, do nothing}: the search is over
WHEN to open (a stochastic stopping tree on sampled price/deposit futures) instead of over
random joint actions, which removes the optimistic bias that noisy in-tree maximisation over
dozens of junk continuations gives to heavily visited branches (see test_pomcpow_root_values.jl).
"""
function timing_tree_action_generator(pomdp::LiPOMDP; rng::AbstractRNG=MersenneTwister(11))
    ta = TimingAwarePolicy(pomdp)
    return function (p, b, h)
        tried = h.tree.tried[h.node]
        if h.node == 1
            s = rand(rng, b)
            isempty(tried) && return action(ta, s)
            (length(tried) == 1 && h.tree.a_labels[tried[1]] != DO_NOTHING) && return DO_NOTHING
            return rand(rng, actions(p, s))
        end
        isempty(tried) && return action(ta, node_mean_state(b))
        any(h.tree.a_labels[n] == DO_NOTHING for n in tried) || return DO_NOTHING
        return action(ta, node_mean_state(b))    # repeats merge (check_repeat_act); a new mean may add one
    end
end

"""
Timing-tree proposer with a single-site root. At the root the candidates are, in order: the
TimingAware action, "do nothing", then every single-site action that is valid for the node's
mean state (explore i, open i within the budget, restore i); once those are exhausted the proposer
repeats "do nothing" (merged by check_repeat_act), so the root never holds more than ~10 actions
and each gets ~1000 of 10k samples instead of ~125 over 80 joint actions. Below the root it is the
timing tree: {TimingAware action on the mean state, do nothing}. Multi-site combinations are
reachable over consecutive years, which is how TimingAware and OneStepLookahead use them anyway.
"""
function timing_tree_single_action_generator(pomdp::LiPOMDP)
    ta = TimingAwarePolicy(pomdp)
    n = pomdp.n_deposits
    function single_site_actions(s::State)
        acts = Action[]
        for i in 1:n
            if !s.have_opened[i]
                push!(acts, Action(site=collect(1:n), type=[j == i ? 1 : 0 for j in 1:n]))
                a = Action(site=collect(1:n), type=[j == i ? 2 : 0 for j in 1:n])
                LiPOMDPs.effective_action(pomdp, s.have_opened, a) == a && push!(acts, a)
            elseif s.is_depleted[i] && !s.restored_mines[i]
                push!(acts, Action(site=collect(1:n), type=[j == i ? 3 : 0 for j in 1:n]))
            end
        end
        return acts
    end
    return function (p, b, h)
        tried = h.tree.tried[h.node]
        s = b isa LiBelief ? State(deposit=[mean(d) for d in b.deposits_distribution], time=b.time,
                                   price=max(mean(b.price_distribution), 100.0),
                                   company_demand=max(mean(b.demand_distribution), 0.0), total_mined=b.total_mined,
                                   have_opened=copy(b.have_opened), is_depleted=copy(b.is_depleted),
                                   restored_mines=copy(b.restored_mines),
                                   recovery=[mean(d) for d in b.recovery_distribution]) :
                               node_mean_state(b)
        isempty(tried) && return action(ta, s)
        labels = [h.tree.a_labels[k] for k in tried]
        DO_NOTHING in labels || return DO_NOTHING
        if h.node == 1
            for a in single_site_actions(s)
                a in labels || return a
            end
        end
        return DO_NOTHING
    end
end

"""
Heuristic prior for new action nodes (MCTS "domain knowledge" initialisation). When POMCPOW adds an
action `a` at node `h`, its value starts at the certainty-equivalent value of "a now, then
TimingAware to the horizon" from a state sampled from the node's belief, with `n0` pseudo-visits.
Motivation (scripts/check_pomcpow_root_bias.jl): with init_V = 0 an action's Q is the average of
whatever random continuations its few visits happened to try, so under-visited actions are biased
DOWN by up to 1.4 reward units and the root ranks actions by visit count rather than value.
"""
function mean_path_prior(pomdp::LiPOMDP; n0::Int=10, rng::AbstractRNG=MersenneTwister(13))
    ta = TimingAwarePolicy(pomdp)
    v = function (p, h, a)
        s = rand(rng, POMCPOW.belief(h))
        isterminal(p, s) && return 0.0
        a_eff = LiPOMDPs.effective_action(p, s.have_opened, a)
        return reward(p, s, a_eff) + p.γ * mean_path_value(p, ta, LiPOMDPs.mean_transition(p, s, a_eff), p.time_horizon)
    end
    n = (p, h, a) -> n0
    return v, n
end
