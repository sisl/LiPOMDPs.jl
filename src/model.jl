using Parameters
using POMDPs
using POMDPTools
using Distributions
using Statistics


@with_kw mutable struct State
    deposit::Vector{Float64} # [v₁, v₂, v₃, v₄] in-situ Li (t Li)
    time::Float64 = 1.0  # current time, year
    price::Float64 = 1_000.0  #$/t LCE 
    company_demand::Float64 = 0.0  # t LCE / year (company target based on market share * global production)
    total_mined::Float64   # cumulative sold LCE (t)
    have_opened::Vector{Bool}  # whether each site has been opened
    is_depleted::Vector{Bool}  # whether each mine is depleted
    restored_mines::Vector{Bool}  # whether each depleted mine has been restored
    recovery::Vector{Float64} = Float64[]  # true recovery per site (hidden); empty = nominal P.recovery
end
Base.hash(s::State, h::UInt) = hash(Tuple(getproperty(s, p) for p in propertynames(s)), h)
Base.isequal(s1::State, s2::State) = all(isequal(getproperty(s1, p), getproperty(s2, p)) for p in propertynames(s1))
Base.:(==)(s1::State, s2::State) = isequal(s1, s2)

@with_kw mutable struct Action
    site::Vector{Int64} = [0, 0, 0, 0]   # site: 0=DoNothing, 1-2=DLE, 3-4=Hard Rock Mining
    type::Vector{Int64} = [0, 0, 0, 0]    # type: 0=DoNothing, 1=explore, 2=open, 3=restore (open mines produce automatically)
end
Base.hash(a::Action, h::UInt) = hash(Tuple(getproperty(a, p) for p in propertynames(a)), h)
Base.isequal(a1::Action, a2::Action) = all(isequal(getproperty(a1, p), getproperty(a2, p)) for p in propertynames(a1))
Base.:(==)(a1::Action, a2::Action) = isequal(a1, a2)

@with_kw mutable struct Observation
    deposits::Vector{Float64} # [v₁, v₂, v₃, v₄] in-situ Li (t Li)
    observed_price::Float64  # $/t LCE
    observed_demand::Float64  # t LCE / year
    recoveries::Vector{Float64} = Float64[]  # noisy recovery per site (metallurgical tests / production)
end
# Structural hashing and equality, so identical observations compare equal
Base.hash(o::Observation, h::UInt) = hash(Tuple(getproperty(o, p) for p in propertynames(o)), h)
Base.isequal(o1::Observation, o2::Observation) = all(isequal(getproperty(o1, p), getproperty(o2, p)) for p in propertynames(o1))
Base.:(==)(o1::Observation, o2::Observation) = isequal(o1, o2)

# Hidden quantities are distributions; status flags are known exactly
@with_kw mutable struct LiBelief{T<:UnivariateDistribution} 
    deposits_distribution::Vector{UnivariateDistribution} # [v₁, v₂, v₃, v₄]; LogNormal prior, truncated Normal after updates
    time::Float64 = 1.0  # current time, year
    price_distribution::T  # belief over price ($/t LCE)
    demand_distribution::T  # belief over company demand (t LCE / year)
    total_mined::Float64  # expected cumulative sold LCE (t)
    production_this_step::Float64 = 0.0  # production in the current time step
    have_opened::Vector{Bool}  # whether each site has been opened
    is_depleted::Vector{Bool}  # whether each mine is depleted
    restored_mines::Vector{Bool}  # whether each depleted mine has been restored
    recovery_distribution::Vector{UnivariateDistribution} = UnivariateDistribution[]  # empty = recovery known (nominal)
end


@with_kw mutable struct LiPOMDP <: POMDP{State,Action,Observation}
    # Observation parameters
    σ_obs_d_dle::Float64 = 20_000.0 # Observation noise for DLE deposits (sites 1&2)
    σ_obs_d_hard_rock::Float64 = 90_000.0 # Observation noise for Hard Rock deposits (sites 3&4)
    σ_obs_p::Float64 = 500.0 # Std of the price observation noise ($/t LCE)
    σ_obs_demand::Float64 = 100.0  # Std of the demand observation noise (t LCE)

    price_init_std::Float64 = 1000.0   # Std of the initial price distribution ($/t LCE)
    demand_init_std::Float64 = 500.0   # Std of the initial demand distribution (t LCE)

    γ::Float64 = 0.926 # = 1/1.08: 8% real discount rate, typical of mining feasibility studies
    time_horizon::Int64 = 31 # years 1995–2025 (start_year + time_horizon − 1)
    n_deposits::Int64 = 4 # number of deposits
    alpha::Float64 = 0.5 # reward weight on profit; 1 - alpha weights monetized emissions
    exploration_cost_dle::Float64 = 3_421_000.0  # $ per exploration of a DLE site
    exploration_cost_hard_rock::Float64 = 1_762_000.0  # $ per exploration of a hard rock site
    LCE_per_lithium::Float64 = 5.323  # 1 t Li ≈ 5.323 t LCE

    # Reward scaling
    emission_scale_factor::Float64 = 1.0  # extra weight on monetized emissions in the reward (1 = CO2_cost as is)
    reward_scale::Float64 = 1_000_000_000.0  # rewards are divided by this ($1B = 1 reward unit)

    # Market parameters
    market_share_target::Float64 = 0.30  # company demand = 30% of global production
    demand_uncertainty::Float64 = 0.05    # std of the global production forecast, as a fraction of its value
    
    # Production parameters (deposits are in-situ Li metal tonnes)
    output::Float64 = 25_000.0  # nameplate capacity per mine (t in-situ Li / year)
    # In-situ Li (t Li): LCE = deposit × recovery × LCE_per_lithium
    init_deposit_size_dle::Float64 = 124_753   # → 531,250 t LCE sellable (recovery 0.80)
    init_deposit_size_mining::Float64 = 587_336 # → 2,188,474 t LCE sellable (recovery 0.70)
    # Coefficient of variation (std/mean) of the LogNormal prior on in-situ Li.
    # 0.25-0.45 matches an Inferred resource (annual output within ±50% at 90%, Parker's criteria);
    # brine (DLE) estimates are less certain than hard rock (porosity/specific-yield heterogeneity).
    dle_uncertainty_factor::Float64 = 0.45
    mining_uncertainty_factor::Float64 = 0.25

    # Mining method for each deposit (true = DLE, false = hard rock mining)
    deposit_types::Vector{Bool} = [true, true, false, false]  # sites 1-2 DLE, sites 3-4 hard rock
    recovery::Vector{Float64} = [0.80, 0.80, 0.70, 0.70]    # Nominal (prior mean) fraction of in-situ Li recovered as saleable product
    # Std of the Beta prior on each site's true recovery (0 for all sites = recovery known).
    # DLE pilots report 84–97% but commercial performance is unproven (50–90%); spodumene
    # recoveries range ~40–90% with ore liberation.
    # Beta(2.4, 0.6) for DLE and Beta(1.65, 0.71) for hard rock (second parameter < 1: density rises toward 100%).
    recovery_uncertainty::Vector{Float64} = [0.20, 0.20, 0.25, 0.25]
    σ_obs_recovery_explore::Float64 = 0.15      # metallurgical test work during exploration
    σ_obs_recovery_production::Float64 = 0.01   # measured yield once producing
    # Total capex the company can commit over the horizon (Inf = no limit). $5B allows any three mines, not all four
    # mines (the dearest, both hard-rock mines plus one DLE, cost $4.55B) but not all four ($5.78B).
    capital_budget::Float64 = 5.0e9
    CO2_cost::Int64 = 190  # $/t CO2: EPA social cost of carbon

    # Hard Rock mining parameters
    mining_CO2_emissions::Float64 = 6.25  # t CO2 per t LCE (= kg CO2e/kg LCE, production only)
    mining_capex::Int64 = 1_661_000_000  # $ per mine
    mining_opex::Int64 = 4_818           # $/t LCE at nominal recovery

    # DLE parameters
    dle_CO2_emissions::Float64 = 4.1   # t CO2 per t LCE (= kg CO2e/kg LCE, production only)
    dle_capex::Int64 = 1_230_000_000  # $ per mine
    dle_opex::Int64 = 5_435           # $/t LCE at nominal recovery

    # Restoration parameters
    dle_restoration_cost::Int64 = 10_000_000  # $ to restore a depleted DLE site
    mining_restoration_cost::Int64 = 20_000_000  # $ to restore a depleted hard rock mine

    # Price model parameters
    price_model_type::Int64 = 1  # 1 static, 2 linear, 3 exponential, 4 GBM, 6 historical
    hist_data::Dict{Int64, Tuple{Float64, Float64}} = lithium_data  # (production, price) by year
    start_year::Int64 = 1995  # calendar year of t = 1
    # Starting prices ($/t LCE); GBM and historical start at the observed start_year price (see get_p₀)
    p₀_static::Float64 = 10_355.33    # constant price for the static model
    p₀_linear::Float64 = 2_381.38     # intercept of the linear model
    p₀_exponential::Float64 = 917.36  # intercept of the exponential model
    p₀_fallback::Float64 = 10_000.0   # used for an unrecognized price_model_type

    α_slope::Float64 = 878.39   # linear model slope ($/t LCE per year)
    λ_exp::Float64 = 0.1261     # exponential model growth rate (per year)

    # GBM (Geometric Brownian Motion) parameters
    # Fitted to lithium_data by scripts/fit_gbm.jl: σ from 5%-winsorized annual log-returns,
    # μ = mean log-return + σ²/2 so the median path matches the observed 1995–2025 growth
    gbm_drift::Float64 = 0.0606
    gbm_volatility::Float64 = 0.2287
end

# Helper function to get the appropriate starting price based on model type
function get_p₀(P::LiPOMDP)
    if P.price_model_type == 1
        return P.p₀_static
    elseif P.price_model_type == 2
        return P.p₀_linear
    elseif P.price_model_type == 3
        return P.p₀_exponential
    elseif P.price_model_type in (4, 6)
        # GBM is a random walk from the current price, so like the historical model it starts at
        # the observed first-year price rather than a fitted intercept
        return lithium_data[P.start_year][2]
    else
        return P.p₀_fallback
    end
end

@with_kw mutable struct LiBeliefUpdater <: Updater
    P::LiPOMDP
end

# True recovery of a site; states without a recovery vector use the nominal value
site_recovery(P::LiPOMDP, s, i::Int) = isempty(s.recovery) ? P.recovery[i] : s.recovery[i]

# Beta distribution with the given mean and std
function beta_from_moments(m::Float64, s::Float64)
    k = m * (1 - m) / s^2 - 1
    k > 0 || error("recovery std $s too large for mean $m")
    return Beta(m * k, (1 - m) * k)
end

# Sample a state from the belief (used by POMCPOW and the belief-based policies)
function Base.rand(rng::AbstractRNG, b::LiBelief)
    sampled_deposits = [max(0.0, rand(rng, d)) for d in b.deposits_distribution]  # clamp ≥ 0
    sampled_price = max(100.0, rand(rng, b.price_distribution))  # floor at $100/t
    sampled_demand = max(0.0, rand(rng, b.demand_distribution))  # clamp ≥ 0

    # Status flags are known, so they are copied from the belief
    return State(
        deposit = sampled_deposits,
        time = b.time,
        price = sampled_price,
        company_demand = sampled_demand,
        total_mined = b.total_mined,
        have_opened = copy(b.have_opened),
        is_depleted = copy(b.is_depleted),
        restored_mines = copy(b.restored_mines),
        recovery = [clamp(rand(rng, d), 0.01, 1.0) for d in b.recovery_distribution],
    )
end
