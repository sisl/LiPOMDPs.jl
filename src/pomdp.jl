#MAIN functions

"""
Return (in_situ_li, sellable_lce) for a site, limited by:
- site nameplate capacity (P.output in LCE),
- remaining deposit (in-situ Li),
- remaining demand (LCE),
- and recovery.
Assumes deposits are in Li, price/opex/emissions per LCE.

Centralized production calculation function to ensure consistency across all POMDP functions.
This function calculates how much can be produced from each site considering:
- Site capacity (P.output)
- Available deposit
- Current year's demand (not cumulative)
- Recovery rate

Note: Demand resets every year, so this is per-time-step production.
"""
function calculate_site_production(P::LiPOMDP, s::State, site::Int, current_demand_lce::Float64)
    if !s.have_opened[site] || s.deposit[site] <= 0.1 || s.is_depleted[site] || s.restored_mines[site]
        return 0.0, 0.0
    end
    # capacity (in-situ Li) from Li nameplate:
    # P.output is already in Li tonnes per year
    cap_in_situ_li = P.output

    # in-situ Li we can actually take this period:
    max_in_situ_li = min(cap_in_situ_li, s.deposit[site])

    # sellable LCE from that in-situ Li:
    r = site_recovery(P, s, site)
    max_sellable_lce = max_in_situ_li * r * P.LCE_per_lithium

    sellable_lce = min(max_sellable_lce, current_demand_lce)

    # recompute the in-situ actually needed for the sellable chosen
    in_situ_li = sellable_lce / (r * P.LCE_per_lithium)

    return in_situ_li, sellable_lce
end

site_capex(P::LiPOMDP, i::Int) = P.deposit_types[i] ? P.dle_capex : P.mining_capex
committed_capex(P::LiPOMDP, have_opened) = sum((site_capex(P, i) for i in 1:P.n_deposits if have_opened[i]); init=0.0)

"""
The action the environment actually executes: "open" requests that would push committed
capex past `P.capital_budget` are dropped, in site order. Used by `transition`, `reward`
and the belief updater, so all three agree whatever a policy proposes.
"""
function effective_action(P::LiPOMDP, have_opened, a::Action)
    spent = committed_capex(P, have_opened)
    types = copy(a.type)
    for i in 1:P.n_deposits
        if types[i] == 2 && !have_opened[i]
            if spent + site_capex(P, i) <= P.capital_budget
                spent += site_capex(P, i)
            else
                types[i] = 0
            end
        end
    end
    return types == a.type ? a : Action(site=copy(a.site), type=types)
end

"""
Calculate total production from all open mines for the current time step.
Returns total production and updated remaining demand.

Note: This tracks remaining demand within a time step (as mines produce).
Demand resets every year between time steps, but within a time step,
we need to track remaining demand to make decisions about opening new mines.
"""
function calculate_total_production(P::LiPOMDP, s::State)
    total_production_lce = 0.0
    remaining_demand_lce = s.company_demand
    for site in 1:P.n_deposits
        in_situ_li, sellable_lce = calculate_site_production(P, s, site, remaining_demand_lce)
        total_production_lce += sellable_lce
        remaining_demand_lce -= sellable_lce
    end
    return total_production_lce, remaining_demand_lce
end

function compute_emissions(P::LiPOMDP, s::State, a::Action)
    total_CO2_tons = 0.0
    remaining_demand_lce = s.company_demand

    # variable emissions proportional to SOLD LCE
    for site in 1:P.n_deposits
        if s.have_opened[site] && s.deposit[site] > 0.1 && !s.restored_mines[site]
            in_situ_li, sellable_lce = calculate_site_production(P, s, site, remaining_demand_lce)
            remaining_demand_lce -= sellable_lce
            if sellable_lce > 0
                ef_lce = P.deposit_types[site] ? P.dle_CO2_emissions : P.mining_CO2_emissions  # t CO2 per t LCE
                total_CO2_tons += sellable_lce * ef_lce
            end
        end
    end

    # Removed: No fixed emissions per open site
    # Removed: No startup emissions for newly opened sites
    # Removed: No restoration emissions
    # Emissions are ONLY based on production (sellable LCE)

    return - total_CO2_tons * P.CO2_cost  # negative penalty
end


function compute_costs(P::LiPOMDP, s::State, a::Action)
    a = effective_action(P, s.have_opened, a)
    costs = 0.0
    remaining_demand_lce = s.company_demand  # Track remaining demand across sites
    
    exploration_cost = 0.0  # Start with zero, don't add base cost
    for site in 1:P.n_deposits
        if a.type[site] == 1  # Exploring
            # Use deposit-type-specific exploration cost
            cost = site <= 2 ? P.exploration_cost_dle : P.exploration_cost_hard_rock
            exploration_cost += cost  # Add cost only when exploring
        end
    end
    
    # Operating costs for all open mines (automatic production)
    for site in 1:P.n_deposits
        if s.have_opened[site] && s.deposit[site] > 0.1 && !s.restored_mines[site]
            in_situ_li, site_production_lce = calculate_site_production(P, s, site, remaining_demand_lce)
            remaining_demand_lce -= site_production_lce  # Update remaining demand for next site
            opex = P.deposit_types[site] ? P.dle_opex : P.mining_opex
            # Opex is paid per tonne of ore processed. `opex` is quoted per t LCE at nominal
            # recovery, so a lower true recovery raises the cost per tonne actually sold.
            costs += opex * site_production_lce * P.recovery[site] / site_recovery(P, s, site)
        end
    end
    
    # Capital costs for newly opened mines
    for site in 1:P.n_deposits
        if a.type[site] == 2 && !s.have_opened[site]  # Open new mine
            capex = P.deposit_types[site] ? P.dle_capex : P.mining_capex
            # Note: You had this calculation but commented it out - you might want to use it
            #amt = min(P.output, s.deposit[site], s.company_demand) 
            #costs += capex * s.deposit[site]  # Capital cost based on capacity
            costs += capex  # Fixed capital cost regardless of size
        end
    end
    
    # Restoration costs for depleted mines
    for site in 1:P.n_deposits
        if a.type[site] == 3 && s.have_opened[site] && s.is_depleted[site]  # Restore depleted mine
            restoration_cost = P.deposit_types[site] ? P.dle_restoration_cost : P.mining_restoration_cost
            costs += restoration_cost
        end
    end

    total_cost = costs + exploration_cost
    
    return total_cost
end

"""
Calculate the maximum possible reward for the given POMDP parameters at a specific time step.
This is used for dynamic reward scaling to normalize rewards to [0, 1] range.
The calculation uses the current state's price and demand to determine the maximum possible reward.
"""
function POMDPs.reward(P::LiPOMDP, s::State, a::Action)

    # Calculate automatic production from all open mines (limited by demand)
    current_production, _ = calculate_total_production(P, s)
    
    # Market-based revenue (production equals sales since we limit by demand)
    revenue = current_production * s.price
    
    # Operating costs and emissions
    costs = compute_costs(P, s, a)  # Includes CAPEX for new mines and OPEX for all open mines
    emissions = compute_emissions(P, s, a)
    
    profit = revenue - costs
    
    # Calculate raw reward
    raw_reward = P.alpha*profit + (1-P.alpha) * (emissions * P.emission_scale_factor)
    
    # Scale reward by manual reward_scale parameter
    scaled_reward = raw_reward / P.reward_scale
    
    # Fix -0.0 display issue
    return scaled_reward == -0.0 ? 0.0 : scaled_reward
end


const PRICE_NOISE_FRAC = 0.1   # std of next-period price noise as a fraction of the base price (models 1, 2, 3, 6)
const PRICE_FLOOR = 1000.0

"""
Noise-free next-period price for the memoryless price models, given the current time `t`
(= s.time): 1 static, 2 linear, 3 exponential, 6 historical (5%/yr extrapolation past the data).
Returns `nothing` for GBM (4), whose next price depends on the current price.
"""
function price_base(P::LiPOMDP, t::Float64)
    p₀ = get_p₀(P)
    P.price_model_type == 1 && return p₀
    P.price_model_type == 2 && return p₀ + P.α_slope * t
    P.price_model_type == 3 && return p₀ * exp(P.λ_exp * t)
    P.price_model_type == 4 && return nothing
    if P.price_model_type == 6
        # Price for the NEXT period (time t + 1), i.e. calendar year start_year + t — the same
        # year `transition` uses for that period's demand (no one-year lag between price and demand)
        historical_year = P.start_year + Int(t)
        haskey(lithium_data, historical_year) && return lithium_data[historical_year][2]
        last_year = maximum(keys(lithium_data))
        return lithium_data[last_year][2] * 1.05^(historical_year - last_year)
    end
    return p₀   # fallback: deterministic
end

function update_price(P::LiPOMDP, s::State, a::Action, price_model_type=1; rng::AbstractRNG=Random.default_rng())
    Δt = 1.0  # step size -- is 1 year for simplicity
    if P.price_model_type == 4
        # One-step GBM: S(t+Δt) = S(t) * exp((μ - σ²/2)Δt + σ dW), dW ~ N(0, Δt)
        dW = rand(rng, Normal(0, sqrt(Δt)))
        new_price = s.price * exp((P.gbm_drift - P.gbm_volatility^2/2) * Δt + P.gbm_volatility * dW)
        return max(new_price, PRICE_FLOOR)
    end
    base = price_base(P, s.time)
    P.price_model_type in (1, 2, 3, 6) || return max(base, PRICE_FLOOR)
    return max(base + rand(rng, Normal(0, PRICE_NOISE_FRAC * abs(base))), PRICE_FLOOR)
end

"""
Prediction step of the price filter: belief over next period's price before observing it.
Memoryless models reset to the model's own predictive distribution; GBM propagates the
current belief with the exact first two moments of p·exp((μ - σ²/2) + σZ).
"""
function predict_price(P::LiPOMDP, b::LiBelief)
    μ, σ = mean(b.price_distribution), std(b.price_distribution)
    if P.price_model_type == 4
        m = μ * exp(P.gbm_drift)
        v = (μ^2 + σ^2) * exp(2 * P.gbm_drift + P.gbm_volatility^2) - m^2
        return Normal(m, sqrt(max(v, 1e-10)))
    end
    base = price_base(P, b.time)
    sd = P.price_model_type in (1, 2, 3, 6) ? PRICE_NOISE_FRAC * abs(base) : 0.0
    return Normal(base, max(sd, 1e-5))
end

"""
Prediction step of the demand filter. Next period's demand is market share × global production
for the year the transition uses for `sp`, with the transition's own forecast noise.
"""
function predict_demand(P::LiPOMDP, b::LiBelief)
    year = P.start_year + Int(floor(b.time + 1)) - 1
    gp_lce = get_historical_production(P, year) * P.LCE_per_lithium
    m = gp_lce * P.market_share_target
    return Normal(m, max(gp_lce * P.demand_uncertainty * P.market_share_target, 1e-5))
end

# Add function to calculate company's demand
function update_demand(P::LiPOMDP, global_production::Float64; rng::AbstractRNG=Random.default_rng())
    # Use global production instead of demand as basis for company target
    # Convert global production from Li metal to LCE
    global_production_lce = global_production * P.LCE_per_lithium
    
    # Add uncertainty to global production forecast
    production_noise = global_production_lce * P.demand_uncertainty
    uncertain_global_production = max(0.0, global_production_lce + rand(rng, Normal(0, production_noise)))
    
    # Calculate company's target demand based on market share of global production
    company_demand = uncertain_global_production * P.market_share_target
    
    return (company_demand)
end

#transition function, observation function, reward function


# Std of a deposit observation for site i. Shared by `observation` and the belief
# updater so the filter uses the same noise the environment generates.
function deposit_obs_std(P::LiPOMDP, i::Int, source::Symbol)
    σ_obs_base = i <= 2 ? P.σ_obs_d_dle : P.σ_obs_d_hard_rock
    source == :explore    && return σ_obs_base * 2.0
    source == :production && return σ_obs_base * 0.1  # production is more precise than exploration
    source == :none       && return σ_obs_base * 50.0
    error("unknown observation source $source")
end

#give signal of where you end up 
function POMDPs.observation(P::LiPOMDP, s::State,  a::Action, sp::State)
    # Generate observations for all sites based on their status
    deposit_estimate_dist = []
    for i in 1:length(sp.deposit)
        # Check if this site was explored in this action
        is_exploring = false
        for j in 1:length(a.site)
            if a.type[j] == 1 && a.site[j] == i
                is_exploring = true
                break
            end
        end
        
        # Check if this site is currently open (producing)
        is_open = sp.have_opened[i]
        
        if is_exploring
            # Direct exploration: high-quality observation
            # Use deposit-type-specific observation noise
            noise = deposit_obs_std(P, i, :explore)
            # Truncate to ensure non-negative observations
            mean_deposit = max(0.0, sp.deposit[i])
            push!(deposit_estimate_dist, Normal(mean_deposit, noise))
        elseif is_open
            # Production-based observation: learn from mining activity
            # Production provides VERY precise observations through actual extraction
            # This should be MORE precise than exploration
            # Use deposit-type-specific observation noise
            noise = deposit_obs_std(P, i, :production)
            # Truncate to ensure non-negative observations
            mean_deposit = max(0.0, sp.deposit[i])
            push!(deposit_estimate_dist, Normal(mean_deposit, noise))
        else
            # No information: unexplored and not producing
            # Use deposit-type-specific observation noise
            noise = deposit_obs_std(P, i, :none)
            # Truncate to ensure non-negative observations
            mean_deposit = max(0.0, sp.deposit[i])
            push!(deposit_estimate_dist, Normal(mean_deposit, noise))
        end
    end

    # Recovery: metallurgical test work when exploring, measured yield once open, else uninformative
    recovery_dists = [Normal(sp.recovery[i], a.type[i] == 1 ? P.σ_obs_recovery_explore :
                                             sp.have_opened[i] ? P.σ_obs_recovery_production : 1.0)
                      for i in 1:length(sp.recovery)]

    price_estimate_dist = Normal(sp.price, P.σ_obs_p)
    demand_estimate_dist = Normal(sp.company_demand, P.σ_obs_demand)
    
    # Combine all distributions: deposits, recoveries (if hidden), price, demand
    prod_dist = product_distribution(deposit_estimate_dist..., recovery_dists..., price_estimate_dist, demand_estimate_dist)
    return prod_dist
end

#have a POMDP that updates, takes the POMDP as the input and the updater
"""
Belief update: a predict-then-correct filter that mirrors `transition` and `observation`.

1. Status flags (opened / restored) follow the action exactly as in `transition`.
2. Prediction: producing mines lose their expected extraction, computed with
   `calculate_site_production` on the mean state (so the production order, capacity and
   demand limits match the true dynamics). Price and demand are propagated through the
   price model (`predict_price`) and the demand model (`predict_demand`).
3. Correction: Kalman (moment-matched Gaussian) updates for explored and producing sites,
   price and demand, using the same observation noise `observation` generates.
4. Depletion is detected from the production observation, with the transition's 5% threshold.
"""
function POMDPs.update(up::LiBeliefUpdater, b::LiBelief, a::Action, o::Observation)
    P = up.P
    n = P.n_deposits
    a = effective_action(P, b.have_opened, a)

    # 1. Status flags implied by the action
    updated_have_opened    = copy(b.have_opened)
    updated_is_depleted    = copy(b.is_depleted)
    updated_restored_mines = copy(b.restored_mines)
    for site in 1:n
        if a.type[site] == 2 && !b.have_opened[site]
            updated_have_opened[site] = true
        end
        if a.type[site] == 3 && b.have_opened[site] && b.is_depleted[site] && !b.restored_mines[site]
            updated_restored_mines[site] = true
        end
    end

    # 2. Prediction: expected depletion of producing mines (a mine opened this step is under construction)
    mean_state = State(
        deposit        = [mean(d) for d in b.deposits_distribution],
        time           = b.time,
        price          = max(mean(b.price_distribution), 100.0),
        company_demand = max(mean(b.demand_distribution), 0.0),   # transition produces against s.company_demand
        total_mined    = b.total_mined,
        have_opened    = b.have_opened,
        is_depleted    = updated_is_depleted,
        restored_mines = updated_restored_mines,
        recovery       = [mean(d) for d in b.recovery_distribution],
    )
    predicted = Vector{UnivariateDistribution}(undef, n)
    remaining_demand = mean_state.company_demand
    expected_production = 0.0
    for site in 1:n
        in_situ, sellable = calculate_site_production(P, mean_state, site, remaining_demand)
        remaining_demand -= sellable
        expected_production += sellable
        d = b.deposits_distribution[site]
        predicted[site] = in_situ > 0 ? Normal(max(0.0, mean(d) - in_situ), std(d)) : d
    end

    # 3. Correction for sites with an informative observation
    updated_deposits = Vector{UnivariateDistribution}(undef, n)
    for i in 1:n
        explored  = a.type[i] == 1
        producing = updated_have_opened[i]           # `observation` keys on sp.have_opened
        if explored || producing
            σ_obs = deposit_obs_std(P, i, explored ? :explore : :production)
            μ_prior, σ_prior = mean(predicted[i]), std(predicted[i])
            σ_obs2   = max(σ_obs^2, 1e-10)
            σ_prior2 = max(σ_prior^2, 1e-10)
            σ_post2  = 1 / (1/σ_prior2 + 1/σ_obs2)
            μ_post   = σ_post2 * (μ_prior/σ_prior2 + o.deposits[i]/σ_obs2)
            # uncertainty floor: 1% of the initial deposit size, so repeated observations never collapse the belief
            initial_deposit_size = i <= 2 ? P.init_deposit_size_dle : P.init_deposit_size_mining
            σ_post = max(sqrt(σ_post2), max(500.0, 0.01 * initial_deposit_size))
            updated_deposits[i] = truncated(Normal(μ_post, σ_post), 0.0, Inf)
        else
            updated_deposits[i] = predicted[i]
        end
    end

    # 4. Restoration zeroes the deposit; depletion is detected from the production observation
    for site in 1:n
        if updated_restored_mines[site] && !b.restored_mines[site]
            updated_deposits[site] = Normal(0.0, 0.1)
        end
        if updated_have_opened[site] && !updated_is_depleted[site] && !updated_restored_mines[site]
            initial_deposit_size = site <= 2 ? P.init_deposit_size_dle : P.init_deposit_size_mining
            if o.deposits[site] <= 0.05 * initial_deposit_size
                updated_is_depleted[site] = true
                updated_deposits[site] = Normal(0.0, 0.1)
            end
        end
    end

    # Recovery is static: correct with metallurgical tests (explore) or measured yield (open)
    updated_recovery = UnivariateDistribution[]
    for (i, d) in enumerate(b.recovery_distribution)
        σ_obs = a.type[i] == 1 ? P.σ_obs_recovery_explore :
                updated_have_opened[i] ? P.σ_obs_recovery_production : Inf
        if isfinite(σ_obs)
            μ0, σ0 = mean(d), std(d)
            σ_post2 = 1 / (1/σ0^2 + 1/σ_obs^2)
            μ_post = σ_post2 * (μ0/σ0^2 + o.recoveries[i]/σ_obs^2)
            push!(updated_recovery, truncated(Normal(μ_post, max(sqrt(σ_post2), 0.005)), 0.0, 1.0))
        else
            push!(updated_recovery, d)
        end
    end

    # Price: predict with the price model, then correct with the noisy price observation
    prior_p = predict_price(P, b)
    μp, σp = mean(prior_p), std(prior_p)
    σp2 = max(σp^2, 1e-10)
    σ_obs_p2 = max(P.σ_obs_p^2, 1e-10)
    σp_post2 = 1 / (1/σp2 + 1/σ_obs_p2)
    μp_post = σp_post2 * (μp/σp2 + o.observed_price/σ_obs_p2)
    updated_price = Normal(μp_post, sqrt(σp_post2))

    # Demand: predict from global production for next year, then correct
    prior_d = predict_demand(P, b)
    μd, σd = mean(prior_d), std(prior_d)
    σd2 = max(σd^2, 1e-10)
    σ_obs_demand2 = max(P.σ_obs_demand^2, 1e-10)
    σd_post2 = 1 / (1/σd2 + 1/σ_obs_demand2)
    μd_post = σd_post2 * (μd/σd2 + o.observed_demand/σ_obs_demand2)
    updated_demand = Normal(μd_post, sqrt(σd_post2))

    return LiBelief(
        deposits_distribution = updated_deposits,
        time                  = b.time + 1,
        price_distribution    = updated_price,
        demand_distribution   = updated_demand,
        total_mined           = b.total_mined + expected_production,
        have_opened           = updated_have_opened,
        is_depleted           = updated_is_depleted,
        restored_mines        = updated_restored_mines,
        recovery_distribution = updated_recovery,
    )
end


@with_kw struct InitialStateDistribution
    deposit_dists::Vector{LogNormal{Float64}} # one LogNormal per site
    price_dist::Normal                         
    demand_dist::Normal                        
    have_opened0::Vector{Bool}                 
    recovery_dists::Vector{Beta{Float64}} = Beta{Float64}[]  # empty = recovery known (nominal)
end


# rand should sample deposits, price, and demand
function Base.rand(d::InitialStateDistribution)
    deposits = [max(0.0, rand(nd)) for nd in d.deposit_dists]   # sample each deposit, clamp ≥ 0
    price    = max(100.0, rand(d.price_dist))                   # sample price, ensure minimum
    demand   = max(0.0, rand(d.demand_dist))                    # sample demand, ensure non-negative
    return State(
        deposit = deposits,
        time = 1.0,
        price = price,
        company_demand = demand,
        total_mined = 0.0,
        have_opened = copy(d.have_opened0),
        is_depleted = fill(false, length(deposits)),
        restored_mines = fill(false, length(deposits)),
        recovery = [clamp(rand(rd), 0.01, 1.0) for rd in d.recovery_dists],
    )
end


# LogNormal with the given mean and std (Distributions.jl parameterizes by log-space μ, σ)
function lognormal_from_moments(m::Float64, s::Float64)
    σ2 = log1p((s / m)^2)
    return LogNormal(log(m) - σ2 / 2, sqrt(σ2))
end

# Helper: choose per-site std (either explicit *_uncertainty_std or factor * mean)
@inline function site_init_std(P, i::Int, μ::Float64)
    if P.deposit_types[i]  # DLE
        hasproperty(P, :dle_uncertainty_std) ? P.dle_uncertainty_std :
        (P.dle_uncertainty_factor * μ)
    else                   # Mining
        hasproperty(P, :mining_uncertainty_std) ? P.mining_uncertainty_std :
        (P.mining_uncertainty_factor * μ)
    end
end


# Completing Initial State: return a container of distributions (not a concrete State)
function POMDPs.initialstate(P::LiPOMDP)
    # Per-site means and stds for deposit size
    means = Vector{Float64}(undef, P.n_deposits)
    stds  = Vector{Float64}(undef, P.n_deposits)
    @inbounds for i in 1:P.n_deposits
        μ = P.deposit_types[i] ? P.init_deposit_size_dle : P.init_deposit_size_mining
        means[i] = μ
        stds[i]  = site_init_std(P, i, μ)
    end

    # Anchors for price & demand (their means)
    price0 = get_p₀(P)
    global_production = get_historical_production(P, P.start_year)
    demand0 = update_demand(P, global_production)

    # Uncertainty for price & demand (now directly from P fields)
    price_std  = P.price_init_std
    demand_std = P.demand_init_std

    #return a Distribution that can sample different initial states
    #so randsample initial state from this distribution
    
    return InitialStateDistribution(
        deposit_dists = [lognormal_from_moments(means[i], stds[i]) for i in 1:P.n_deposits],
        price_dist    = Normal(price0,  price_std),
        demand_dist   = Normal(demand0, demand_std),
        have_opened0  = fill(false, P.n_deposits),
        recovery_dists = all(P.recovery_uncertainty .> 0) ?
            [beta_from_moments(P.recovery[i], P.recovery_uncertainty[i]) for i in 1:P.n_deposits] : Beta{Float64}[],
    )
end

function create_initial_belief(P::LiPOMDP)
    d = initialstate(P)  # your InitialStateDistribution
    belief = LiBelief(
        deposits_distribution = d.deposit_dists,
        time = 1.0,
        price_distribution    = d.price_dist,
        demand_distribution   = d.demand_dist,
        total_mined = 0.0,
        have_opened = copy(d.have_opened0),
        is_depleted = fill(false, P.n_deposits),
        restored_mines = fill(false, P.n_deposits),
        recovery_distribution = Vector{UnivariateDistribution}(d.recovery_dists),
    )
    true_state = rand(initialstate(P))  # Get the true initial state
    
    return true_state, belief  # Return both state and belief
end



#given a state s and action a it outputs the distribution of the next state sp
function POMDPs.transition(P::LiPOMDP, s::State, a::Action)
    return Deterministic(sample_transition(P, s, a, Random.default_rng()))
end

# Draws the next state using `rng` for price and demand noise.
function sample_transition(P::LiPOMDP, s::State, a::Action, rng::AbstractRNG)
    price  = update_price(P, s, a; rng=rng)
    # Company demand for the next period comes from that year's global production, with forecast noise
    demand = update_demand(P, get_historical_production(P, P.start_year + Int(floor(s.time + 1)) - 1); rng=rng)
    return transition_core(P, s, a, price, demand)
end

"""
Expected next-period price given the current state: the GBM conditional mean p·exp(μ) (the path
`TimingAwarePolicy` plans on) or the memoryless model's noise-free base price.
"""
function expected_price(P::LiPOMDP, s::State)
    P.price_model_type == 4 && return max(s.price * exp(P.gbm_drift), PRICE_FLOOR)
    return max(price_base(P, s.time), PRICE_FLOOR)
end

"""Expected company demand for the period starting at time `t`: market share × that year's global production."""
expected_demand(P::LiPOMDP, t::Real) =
    get_historical_production(P, P.start_year + Int(floor(t)) - 1) * P.LCE_per_lithium * P.market_share_target

"""
Noise-free (certainty-equivalent) transition: next period's price and demand at their conditional
means, everything else exactly as `sample_transition`.
"""
mean_transition(P::LiPOMDP, s::State, a::Action) =
    transition_core(P, s, a, expected_price(P, s), expected_demand(P, s.time + 1))

"""
Discounted α-weighted return of following `policy` from state `s` for up to `steps` periods along the
noise-free path (`mean_transition`). A certainty-equivalent value estimate with no Monte-Carlo
variance, used as POMCPOW's leaf value estimate (see `make_pomcpow(rollout=:mean_path)` in
scripts/pomcpow_tuning_harness.jl).
"""
function mean_path_value(P::LiPOMDP, policy::Policy, s::State, steps::Int)
    total, disc = 0.0, 1.0
    for _ in 1:steps
        isterminal(P, s) && break
        a = action(policy, s)
        total += disc * reward(P, s, a)
        s = mean_transition(P, s, a)
        disc *= P.γ
    end
    return total
end

# Deterministic part of the transition, given next period's price and company demand.
function transition_core(P::LiPOMDP, s::State, a::Action, price::Float64, demand::Float64)
    a = effective_action(P, s.have_opened, a)
    sp = deepcopy(s)
    sp.time += 1
    sp.price = price
    sp.company_demand = demand

    # Process each site's action
    for site in 1:P.n_deposits
        action_type = a.type[site]
        
        # Action: Do Nothing or Explore (type 0 and 1)
        if action_type in (0, 1)
            # No state change for these actions
            continue
        end
        
        # Action: Open Mine (type 2)
        if action_type == 2 && !s.have_opened[site]
            sp.have_opened[site] = true
        end
        
        # Action: Restore Mine (type 3)
        if action_type == 3 && s.have_opened[site] && s.is_depleted[site] && !s.restored_mines[site]
            # Keep is_depleted as true - mine stays depleted after restoration
            sp.restored_mines[site] = true
            # FIXED: Set deposits to 0 when restoring - mine is completely depleted
            sp.deposit[site] = 0.0
        end
    end

    # Automatic production from all open mines using centralized calculation
    # Use the same remaining demand logic as calculate_total_production
    # IMPORTANT: Use the OLD demand (s.company_demand) for production, not the new demand
    # A mine opened this step is under construction: it produces from the next step on,
    # matching `reward`, which pays for mines open at the start of the step (s.have_opened).
    total_mined_this_step = 0.0
    remaining_demand_lce = s.company_demand  # Use OLD demand for production calculation
    
    for site in 1:P.n_deposits
        if s.have_opened[site] && sp.deposit[site] > 0.1 && !sp.is_depleted[site] && !sp.restored_mines[site]
            in_situ_li, sellable_lce = calculate_site_production(P, sp, site, remaining_demand_lce)
            remaining_demand_lce -= sellable_lce  # Update remaining demand for next site
            sp.deposit[site] -= in_situ_li                # Li depletion
            total_mined_this_step += sellable_lce         # track SOLD LCE
            
            # Check if mine is depleted after this step (5% of initial deposit)
            initial_deposit_size = site <= 2 ? P.init_deposit_size_dle : P.init_deposit_size_mining
            depletion_threshold = 0.05 * initial_deposit_size
            if sp.deposit[site] <= depletion_threshold
                sp.is_depleted[site] = true
            end
        end
    end

    sp.total_mined = s.total_mined + total_mined_this_step  # Accumulate

    return sp
end

#if the process similar then use it as a nindex but if its different then create branch
#last line has to be deterministic 
#each action type will affect the state being updated 
#produce at capacity -- rate capacity

# Update the gen function to handle demand observations
function POMDPs.gen(P::LiPOMDP, s::State, a::Action, rng::AbstractRNG)
    
    sp = sample_transition(P, s, a, rng)

    o_array = rand(rng, observation(P, s, a, sp))
    n = length(sp.deposit)
    o = Observation(
        deposits = [max(0.0, x) for x in o_array[1:n]],   # truncated to ≥ 0
        recoveries = o_array[n+1:end-2],                   # empty when recovery is known
        observed_price = o_array[end-1],
        observed_demand = o_array[end],
    )

    r = reward(P, s, a)

    return (sp=sp, o=o, r=r)
end

# Additional POMDP interface methods
function POMDPs.isterminal(P::LiPOMDP, s::State)
    # Terminal when time exceeds horizon or all deposits are exhausted
    all_exhausted = all(d <= 0.0 for d in s.deposit)
    time_exceeded = s.time > P.time_horizon
    return all_exhausted || time_exceeded
end

function POMDPs.discount(P::LiPOMDP)
    return P.γ
end

# Define action space - now returns all possible combinations of actions
function POMDPs.actions(P::LiPOMDP)
    acts = Action[]
    
    possible_actions = [0, 1, 2, 3]

    for action_site_1 in possible_actions
        for action_site_2 in possible_actions
            for action_site_3 in possible_actions
                for action_site_4 in possible_actions
                    push!(acts, Action(site=[1, 2, 3, 4], type=[action_site_1, action_site_2, action_site_3, action_site_4]))
                end
            end
        end
    end

    # No need for special global do-nothing action - [1,2,3,4] with type=[0,0,0,0] already covers it
    return acts
end



# Helper function to check if an action is valid for a given state
function is_action_valid(action::Action, s::State, n_deposits::Int)
    for site in 1:n_deposits
        action_type = action.type[site]
        
        if action_type == 1 || action_type == 2  # explore or open
            if s.have_opened[site]  # Can't explore or open an already opened mine
                return false
            end
        elseif action_type == 3  # restore
            if !s.have_opened[site] || !s.is_depleted[site] || s.restored_mines[site]
                return false
            end
        end
    end
    return true
end

function POMDPs.actions(P::LiPOMDP, s::State)
    acts = actions(P)  # Start with all possible actions
    
    # Filter out invalid actions based on state
    valid_acts = Action[]
    
    for action in acts
        if is_action_valid(action, s, P.n_deposits) && effective_action(P, s.have_opened, action) == action
            push!(valid_acts, action)
        end
    end
    
    return valid_acts
end


# Actions for beliefs (same logic as states)
function POMDPs.actions(P::LiPOMDP, b::LiBelief)
    s = rand(b)  # Sample a representative state from the belief
    actions = POMDPs.actions(P, s)  

    return actions
end
    


function POMDPs.actionindex(P::LiPOMDP, a::Action)
    # For the new action space with all combinations
    # Each site can have action 0, 1, 2, or 3
    # Total combinations: 4^4 = 256 actions
    
    # Convert action combination to index using base-4 representation
    # action.type = [type1, type2, type3, type4] where each type is 0, 1, 2, or 3
    # Index = type1 * 4^3 + type2 * 4^2 + type3 * 4^1 + type4 * 4^0 + 1
    
    index = 1  # Start at 1 (Julia uses 1-based indexing)
    
    for i in 1:4
        index += a.type[i] * (4^(4-i))
    end
    
    return index
end

#SINGLE STEP ACTIONS
# # Define action space - now returns all possible combinations of actions
# function POMDPs.actions(P::LiPOMDP)
#     # For now, return a simplified action space with single actions
#     # This can be expanded to include multiple simultaneous actions
#     acts = Action[]
    
#     # Do nothing action
#     push!(acts, Action(site=[0, 0, 0, 0], type=[0, 0, 0, 0]))

#     # Single site actions
#     for site in 1:P.n_deposits
#         # Explore
#         site_vec = zeros(Int, 4)
#         type_vec = zeros(Int, 4)
#         # site_vec[1] = site
#         # type_vec[1] = 1
#         site_vec[site] = site
#         type_vec[site] = 1
#         push!(acts, Action(site=site_vec, type=type_vec))
        
#         # Open mine
#         site_vec = zeros(Int, 4)
#         type_vec = zeros(Int, 4)
#         # site_vec[1] = site
#         # type_vec[1] = 2
#         site_vec[site] = site
#         type_vec[site] = 2
#         push!(acts, Action(site=site_vec, type=type_vec))
        
#         # Restore mine
#         site_vec = zeros(Int, 4)
#         type_vec = zeros(Int, 4)
#         site_vec[site] = site
#         type_vec[site] = 3
#         push!(acts, Action(site=site_vec, type=type_vec))
#     end
    
#     # Could add combined actions here if needed
#     return acts
# end



# # Stateful action generator
# function POMDPs.actions(P::LiPOMDP, s::State)

#     acts = Action[]
    
#     # Do nothing action
#     push!(acts, Action(site=[0, 0, 0, 0], type=[0, 0, 0, 0]))
    
#     # Single site actions based on state
#     for site in 1:P.n_deposits
#         if !s.have_opened[site]
#            # println("DEBUG: Site $site not opened, adding explore/open actions")

#             # Can explore unopened sites
#             site_vec = zeros(Int, 4)
#             type_vec = zeros(Int, 4)
#             # site_vec[1] = site
#             # type_vec[1] = 1
#             site_vec[site] = site
#             type_vec[site] = 1
#             push!(acts, Action(site=site_vec, type=type_vec))
            
#             # Can open unopened sites
#             site_vec = zeros(Int, 4)
#             type_vec = zeros(Int, 4)
#             # site_vec[1] = site
#             # type_vec[1] = 2
#             site_vec[site] = site
#             type_vec[site] = 2
#             push!(acts, Action(site=site_vec, type=type_vec))
#         elseif s.have_opened[site] && s.is_depleted[site] && !s.restored_mines[site]
#             # Can restore depleted mines that haven't been restored yet
#             site_vec = zeros(Int, 4)
#             type_vec = zeros(Int, 4)
#             site_vec[site] = site
#             type_vec[site] = 3  # Restore action
#             push!(acts, Action(site=site_vec, type=type_vec))
#         end
#         # No operate action needed - mines operate automatically
#     end
#     #println("DEBUG: Generated $(length(acts)) total actions")
#     return acts
# end


# # Actions for beliefs (same logic as states)
# function POMDPs.actions(P::LiPOMDP, b::LiBelief)
#     s = rand(b)  # Sample a representative state from the belief
#     actions = POMDPs.actions(P, s)  

#     return actions
# end
    


# function POMDPs.actionindex(P::LiPOMDP, a::Action)
#     # Simple indexing for single-action vectors
#     # Do nothing action
#     if all(a.site .== 0) && all(a.type .== 0)
#         return 1
#     end
    
#     # Find first non-zero action
#     for i in 1:length(a.site)
#         if a.site[i] > 0 && a.type[i] > 0
#             site = a.site[i]
#             type = a.type[i]
#             # 2 actions per site (explore, open)
#             return 1 + (site - 1) * 2 + type
#         end
#     end
    
#     return 1  # Default to do nothing
# end

# Helper functions for policy evaluation
"""
Calculate expected profit if opening site i now.
Returns the expected profit over the mine's lifetime.
"""
function expected_profit_site(P::LiPOMDP, s::State, i::Int)
    if s.have_opened[i] || s.deposit[i] <= 0.1 || s.is_depleted[i] || s.restored_mines[i]
        return 0.0
    end
    
    remaining_time = P.time_horizon - s.time
    if remaining_time <= 0
        return 0.0
    end
    
    # Calculate expected production over lifetime
    max_possible_in_situ = min(s.deposit[i], P.output * remaining_time)
    max_possible_sellable = max_possible_in_situ * site_recovery(P, s, i) * P.LCE_per_lithium
    
    # Calculate realistic production considering demand constraints
    avg_demand_per_period = s.company_demand
    realistic_lifetime_demand = avg_demand_per_period * remaining_time
    realistic_sellable = min(max_possible_sellable, realistic_lifetime_demand)
    
    # Calculate lifetime revenue using current price (simplified)
    # In practice, you might want to use price forecasting like in other policies
    lifetime_revenue = realistic_sellable * s.price
    
    # Calculate lifetime costs
    capex = P.deposit_types[i] ? P.dle_capex : P.mining_capex
    opex = P.deposit_types[i] ? P.dle_opex : P.mining_opex
    lifetime_opex = opex * realistic_sellable
    
    # Calculate lifetime profit
    total_profit = lifetime_revenue - lifetime_opex - capex
    
    return total_profit
end

"""
Calculate expected emissions if opening site i now.
Returns the expected emissions over the mine's lifetime.
"""
function expected_emission_site(P::LiPOMDP, s::State, i::Int)
    if s.have_opened[i] || s.deposit[i] <= 0.1 || s.is_depleted[i] || s.restored_mines[i]
        return 0.0
    end
    
    remaining_time = P.time_horizon - s.time
    if remaining_time <= 0
        return 0.0
    end
    
    # Calculate expected production over lifetime
    max_possible_in_situ = min(s.deposit[i], P.output * remaining_time)
    max_possible_sellable = max_possible_in_situ * site_recovery(P, s, i) * P.LCE_per_lithium
    
    # Calculate realistic production considering demand constraints
    avg_demand_per_period = s.company_demand
    realistic_lifetime_demand = avg_demand_per_period * remaining_time
    realistic_sellable = min(max_possible_sellable, realistic_lifetime_demand)
    
    # Calculate lifetime emissions
    # Variable emissions (per tonne LCE produced)
    variable_emissions_per_tonne = P.deposit_types[i] ? P.dle_CO2_emissions : P.mining_CO2_emissions
    lifetime_variable_emissions = realistic_sellable * variable_emissions_per_tonne
    
    # Removed: No fixed emissions
    # Removed: No startup emissions
    # Emissions are ONLY based on production
    
    # Total lifetime emissions (production only)
    total_lifetime_emissions = lifetime_variable_emissions  # t CO2
    
    return total_lifetime_emissions
end

# Initial observation distribution
function POMDPs.initialobs(P::LiPOMDP, s::State)
    # Use the same observation model as regular observations
    # Create a dummy action for initial observation
    dummy_action = Action(site=[0, 0, 0, 0], type=[0, 0, 0, 0])
    return observation(P, s, dummy_action, s)
end

# Observation weight for POMCPOW
function POMDPTools.obs_weight(P::LiPOMDP, s::State, a::Action, sp::State, o::Observation)
    # Calculate the probability density of observation o given state sp
    # Get the observation distribution
    o_dist = observation(P, s, a, sp)
    
    # Create observation vector from Observation struct
    o_vec = vcat(o.deposits, o.recoveries, o.observed_price, o.observed_demand)
    
    # Calculate pdf
    return pdf(o_dist, o_vec)
end
