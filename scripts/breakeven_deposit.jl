"""
Break-even deposit size per mine type and opening year.

NPV(D, t0) = E[ Σ_{t ≥ t0} γ^(t-t0) profit_t | open the mine at t0 ] − E[ same | never open ],
with the mine alone (no other sites open), α = 1, using the model's own `reward` and
`sample_transition` averaged over paired random price/demand paths. The break-even deposit D*
solves NPV(D*, t0) = 0. Information about a deposit is only worth gathering if D* falls inside
the prior, so this compares D* with the prior's 10/50/90% quantiles.

  julia --project=. scripts/breakeven_deposit.jl [price_model] [gamma]
"""

using LiPOMDPs, POMDPs, Random, Statistics, Distributions, Printf
import LiPOMDPs: sample_transition, lognormal_from_moments

price_model = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 6
γ           = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : LiPOMDP().γ
const N_PATHS = 200

P = LiPOMDP(price_model_type=price_model, alpha=1.0, γ=γ)
noop = Action(site=collect(1:4), type=zeros(Int, 4))

# Mean price/demand state at the start of year t0 (the transitions before t0 do not depend on the site)
function state_at(P, t0, site, D, rng)
    Random.seed!(rand(rng, UInt64))      # InitialStateDistribution only implements rand(d)
    s = rand(POMDPs.initialstate(P))
    s.deposit .= 1.0                     # other sites: never opened; > 0 so the episode is not terminal
    s.deposit[site] = D
    for _ in 1:(t0 - 1)
        s = sample_transition(P, s, noop, rng)
    end
    return s
end

function npv(P, site, D, t0)
    total = 0.0
    for k in 1:N_PATHS
        s0 = state_at(P, t0, site, D, MersenneTwister(k))
        v = 0.0
        for open in (true, false)
            s = deepcopy(s0); rng = MersenneTwister(hash((k, :after)))   # same path with and without the mine
            acc = 0.0
            for t in t0:P.time_horizon
                a = (open && t == t0) ? Action(site=collect(1:4), type=[i == site ? 2 : 0 for i in 1:4]) : noop
                acc += P.γ^(t - t0) * reward(P, s, a) * P.reward_scale     # α = 1 ⇒ reward·scale = profit in $
                s = sample_transition(P, s, a, rng)
            end
            v += open ? acc : -acc
        end
        total += v
    end
    return total / N_PATHS
end

function breakeven(P, site, t0; hi=1e8)
    npv(P, site, hi, t0) <= 0 && return Inf          # never pays back within the horizon
    lo_ = 0.0
    npv(P, site, lo_, t0) >= 0 && return 0.0
    for _ in 1:40
        mid = (lo_ + hi) / 2
        npv(P, site, mid, t0) > 0 ? (hi = mid) : (lo_ = mid)
        hi - lo_ < 500 && break
    end
    return (lo_ + hi) / 2
end

prior(site) = begin
    μ  = P.deposit_types[site] ? P.init_deposit_size_dle : P.init_deposit_size_mining
    cv = P.deposit_types[site] ? P.dle_uncertainty_factor : P.mining_uncertainty_factor
    lognormal_from_moments(μ, cv * μ)
end

@printf("Break-even in-situ deposit (t Li), price model %d, γ = %.3f, α = 1, %d paths\n", price_model, γ, N_PATHS)
for (site, label) in ((1, "DLE"), (3, "Hard rock"))
    d = prior(site)
    capex = P.deposit_types[site] ? P.dle_capex : P.mining_capex
    @printf("\n%s  (capex \$%.0fM; prior P10/P50/P90 = %.0f / %.0f / %.0f t)\n", label, capex / 1e6,
            quantile(d, 0.1), quantile(d, 0.5), quantile(d, 0.9))
    println("  open in  year   D* (t Li)    D*/P50   NPV at P50 (\$M)   capex making D* = P50 (\$M)")
    for t0 in 1:2:P.time_horizon
        Dstar = breakeven(P, site, t0)
        v50 = npv(P, site, quantile(d, 0.5), t0)
        @printf("  t=%2d     %d  %11s   %7s   %14.0f   %20.0f\n", t0, P.start_year + t0 - 1,
                isfinite(Dstar) ? @sprintf("%.0f", Dstar) : "never",
                isfinite(Dstar) ? @sprintf("%.2f", Dstar / quantile(d, 0.5)) : "—",
                v50 / 1e6, (capex + v50 * P.γ^0) / 1e6)
    end
end

# -----------------------------------------------------------------------------
# Break-even RECOVERY for a median deposit. Two cost models:
#   :per_lce  opex per tonne LCE sold (current model): recovery scales revenue and opex together
#   :per_ore  opex per tonne of ore processed: the cost per tonne LCE rises as recovery falls,
#             opex_eff = opex · r_nominal / r, so the nominal recovery reproduces the current costs
# -----------------------------------------------------------------------------
function npv_recovery(site, r, t0, cost_model)
    Q = deepcopy(P)
    r0 = Q.recovery[site]
    Q.recovery[site] = r
    if cost_model == :per_ore
        if Q.deposit_types[site]
            Q.dle_opex = round(Int, Q.dle_opex * r0 / r)
        else
            Q.mining_opex = round(Int, Q.mining_opex * r0 / r)
        end
    end
    return npv(Q, site, quantile(prior(site), 0.5), t0)
end

function breakeven_recovery(site, t0, cost_model)
    npv_recovery(site, 1.0, t0, cost_model) <= 0 && return Inf
    npv_recovery(site, 0.02, t0, cost_model) >= 0 && return 0.0
    lo_, hi = 0.02, 1.0
    for _ in 1:20
        mid = (lo_ + hi) / 2
        npv_recovery(site, mid, t0, cost_model) > 0 ? (hi = mid) : (lo_ = mid)
    end
    return (lo_ + hi) / 2
end

println("\n\nBreak-even recovery r* for a median deposit (NPV = 0); nominal recovery DLE 0.80, hard rock 0.70")
println("  open in  year     DLE per_lce  DLE per_ore   HR per_lce  HR per_ore")
for t0 in 1:2:P.time_horizon
    f(x) = isfinite(x) ? @sprintf("%11.2f", x) : @sprintf("%11s", "never")
    @printf("  t=%2d     %d  %s  %s  %s  %s\n", t0, P.start_year + t0 - 1,
            f(breakeven_recovery(1, t0, :per_lce)), f(breakeven_recovery(1, t0, :per_ore)),
            f(breakeven_recovery(3, t0, :per_lce)), f(breakeven_recovery(3, t0, :per_ore)))
end
