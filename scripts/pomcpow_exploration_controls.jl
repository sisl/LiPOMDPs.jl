"""
POMCPOW exploration controls, shared by scripts/pomcpow_tuning_harness.jl and
scripts/final_results_best_pomcpow.jl:
- info_gated_action_generator: progressive-widening proposer that only proposes exploring a site
  while its belief is wider than one exploration measurement;
- CheapestNearBest: final root-action criterion that drops explorations that do not change the
  value estimate beyond a tolerance.
"""

using LiPOMDPs, POMDPs, POMCPOW, Random, Statistics

"""
True while exploring site `i` can still teach something: the belief's spread in deposit size
or recovery (estimated from `n` samples of `b`) is larger than the noise of one exploration
measurement of that quantity. Once both are below it, another exploration barely moves the belief.
"""
function explore_informative(P::LiPOMDP, b, i::Int, rng::AbstractRNG; n::Int=30)
    ss = [rand(rng, b) for _ in 1:n]
    std([s.deposit[i] for s in ss]) > LiPOMDPs.deposit_obs_std(P, i, :explore) ||
        std([LiPOMDPs.site_recovery(P, s, i) for s in ss]) > P.σ_obs_recovery_explore
end

"""
Action proposer that keeps uniform random proposals but drops the "explore" flag on any site
whose belief is already tighter than one exploration measurement (see explore_informative).
"""
function info_gated_action_generator(pomdp::LiPOMDP; rng::AbstractRNG=MersenneTwister(11))
    return function (p, b, h)
        a = rand(rng, actions(p, rand(rng, b)))
        types = [t == 1 && !explore_informative(p, b, i, rng) ? 0 : t for (i, t) in enumerate(a.type)]
        return Action(site=copy(a.site), type=types)
    end
end

"""
Final root-action criterion: start from the highest-Q action and, among tried actions that make
the same open/restore decisions and score within `tol` (fraction of |Q|) of it, take the one with
the fewest explorations. An exploration costs ~0.003 in reward units, far below the noise of the
Q estimates, so MaxQ alone cannot tell "open 3" from "open 3 + explore 1" and pays for both.
"""
struct CheapestNearBest
    tol::Float64
end

function POMCPOW.select_best(crit::CheapestNearBest, h_node::POMCPOW.POWTreeObsNode, rng)
    tree, h = h_node.tree, h_node.node
    best = POMCPOW.select_best(POMCPOW.MaxQ(), h_node, rng)
    best_v = tree.v[best]
    core(a) = [t == 1 ? 0 : t for t in a.type]             # decisions other than exploring
    target = core(tree.a_labels[best])
    nexp(node) = count(==(1), tree.a_labels[node].type)
    choice = best
    for node in tree.tried[h]
        core(tree.a_labels[node]) == target || continue
        tree.v[node] >= best_v - crit.tol * abs(best_v) || continue
        if nexp(node) < nexp(choice) || (nexp(node) == nexp(choice) && tree.v[node] > tree.v[choice])
            choice = node
        end
    end
    return choice
end


"""
Robust root criterion for irreversible actions. Take the MaxQ action; among tried root actions that
open FEWER mines (explorations ignored) and whose Q is within `tol` (fraction of |Q|) of the best,
prefer the one with the fewest opens (ties: higher Q). Motivation (2026-10-21 in-model tests): the
tree's Q gap between "open now" and "wait" is of the order of its own estimation noise, and opening is
irreversible, so a yearly near-coin-flip ratchets into early opening; the heuristics never face this
because they compare actions on common random numbers. `tol` = 0 reduces to MaxQ.
"""
struct FewerOpensNearBest
    tol::Float64
end

function POMCPOW.select_best(crit::FewerOpensNearBest, h_node::POMCPOW.POWTreeObsNode, rng)
    tree, h = h_node.tree, h_node.node
    best = POMCPOW.select_best(POMCPOW.MaxQ(), h_node, rng)
    best_v = tree.v[best]
    nopen(node) = count(==(2), tree.a_labels[node].type)
    choice = best
    for node in tree.tried[h]
        tree.v[node] >= best_v - crit.tol * abs(best_v) || continue
        if nopen(node) < nopen(choice) || (nopen(node) == nopen(choice) && tree.v[node] > tree.v[choice])
            choice = node
        end
    end
    return choice
end
