"""
Expected value of perfect information (EVPI) for the "which sites to open in year t0" decision.

  V_prior   = max_S  E[ NPV(S) ]            choose the affordable site set S using only the prior
  V_perfect = E[ max_S NPV(S | true state) ]  choose S knowing each site's deposit and recovery
  EVPI      = V_perfect − V_prior ≥ 0

NPV(S) is the discounted profit (α = 1) from t0 to the horizon when the sites in S are opened at
t0 and nothing else is done, simulated with the model's own `reward` and `sample_transition` on
paired price/demand paths. Exploration only pays if it is worth more than it costs, so EVPI is
compared with the cost of exploring every site (an upper bound on what information can be bought).

Four variants separate what creates information value:
  tonnage only / + recovery uncertainty  ×  no capital budget / capital budget.

  julia --project=. scripts/value_of_information.jl [price_model] [gamma] [n_samples]
"""

using LiPOMDPs, POMDPs, Random, Statistics, Printf
import LiPOMDPs: sample_transition, committed_capex

price_model = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 6
γ           = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : LiPOMDP().γ
n_samples   = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 400

const OPEN_YEARS = [13, 17, 19, 21, 23]

variants = length(ARGS) >= 4 && ARGS[4] == "levers" ? [
    # 2026-10-21 model levers: current model vs higher recovery uncertainty and a tighter budget
    "current: rec sd 0.10, \$5B"            => (recovery_uncertainty = fill(0.10, 4),              capital_budget = 5.0e9),
    "rec sd 0.15, \$5B"                     => (recovery_uncertainty = fill(0.15, 4),              capital_budget = 5.0e9),
    "rec sd 0.15 DLE / 0.20 HR, \$5B"       => (recovery_uncertainty = [0.15, 0.15, 0.20, 0.20],  capital_budget = 5.0e9),
    "rec sd 0.10, \$3B"                     => (recovery_uncertainty = fill(0.10, 4),              capital_budget = 3.0e9),
    "rec sd 0.15, \$3B"                     => (recovery_uncertainty = fill(0.15, 4),              capital_budget = 3.0e9),
    "rec sd 0.15 DLE / 0.20 HR, \$3B"       => (recovery_uncertainty = [0.15, 0.15, 0.20, 0.20],  capital_budget = 3.0e9),
] : [
    "tonnage only, no budget"   => (recovery_uncertainty = zeros(4),         capital_budget = Inf),
    "+ recovery, no budget"     => (recovery_uncertainty = fill(0.10, 4),    capital_budget = Inf),
    "tonnage only, \$2.5B budget" => (recovery_uncertainty = zeros(4),       capital_budget = 2.5e9),
    "+ recovery, \$2.5B budget"   => (recovery_uncertainty = fill(0.10, 4),  capital_budget = 2.5e9),
    "+ recovery, \$4B budget"     => (recovery_uncertainty = fill(0.10, 4),  capital_budget = 4.0e9),
]

noop = Action(site=collect(1:4), type=zeros(Int, 4))

# Every affordable set of sites (as a Bool mask), including opening nothing
affordable_sets(P) = [m for m in (digits(k, base=2, pad=4) .== 1 for k in 0:15) if committed_capex(P, m) <= P.capital_budget]

function npv_of_set(P, s0, mask, t0, path_seed)
    s = deepcopy(s0); rng = MersenneTwister(path_seed)
    open_a = Action(site=collect(1:4), type=[m ? 2 : 0 for m in mask])
    v = 0.0
    for t in t0:P.time_horizon
        a = t == t0 ? open_a : noop
        v += P.γ^(t - t0) * reward(P, s, a) * P.reward_scale
        s = sample_transition(P, s, a, rng)
    end
    return v
end

function evpi(P, t0)
    sets = affordable_sets(P)
    npvs = zeros(n_samples, length(sets))
    for k in 1:n_samples
        Random.seed!(k)
        s = rand(POMDPs.initialstate(P))                # true deposits and recoveries
        rng = MersenneTwister(hash((k, :before)))
        for _ in 1:(t0 - 1)
            s = sample_transition(P, s, noop, rng)
        end
        for (j, m) in enumerate(sets)
            npvs[k, j] = npv_of_set(P, s, m, t0, hash((k, :after)))   # same price path for every set
        end
    end
    v_prior   = maximum(mean(npvs, dims=1))
    v_perfect = mean(maximum(npvs, dims=2))
    best_prior = sets[argmax(vec(mean(npvs, dims=1)))]
    # how often the perfectly informed choice differs from the prior choice
    differs = mean(sets[argmax(npvs[k, :])] != best_prior for k in 1:n_samples)
    return v_prior, v_perfect - v_prior, differs, best_prior
end

P0 = LiPOMDP(price_model_type=price_model, alpha=1.0, γ=γ)
explore_all = 2 * P0.exploration_cost_dle + 2 * P0.exploration_cost_hard_rock
@printf("EVPI for 'which sites to open at t0', price model %d, γ = %.3f, %d samples\n", price_model, γ, n_samples)
@printf("Cost of exploring all four sites once: \$%.1fM\n", explore_all / 1e6)
for (label, kw) in variants
    P = LiPOMDP(; price_model_type=price_model, alpha=1.0, γ=γ, kw...)
    println("\n$label")
    println("  open in  year   V_prior (\$M)   EVPI (\$M)   EVPI/V_prior   choice changes   best set with prior only")
    for t0 in OPEN_YEARS
        v, e, d, best = evpi(P, t0)
        @printf("  t=%2d     %d   %11.0f   %9.1f   %10.1f%%   %12.0f%%   %s\n", t0, P.start_year + t0 - 1,
                v / 1e6, e / 1e6, 100 * e / max(abs(v), 1), 100 * d, string(findall(best)))
    end
end
