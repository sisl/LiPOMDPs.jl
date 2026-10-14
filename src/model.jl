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
    total_mined::Float64   # track cumulative SOLD LCE
    have_opened::Vector{Bool}  # Boolean value to represent whether or not we have opened a mine
    is_depleted::Vector{Bool}  # Boolean value to represent whether a mine is depleted
    restored_mines::Vector{Bool}  # Boolean value to represent whether a mine has been restored
end
Base.hash(s::State, h::UInt) = hash(Tuple(getproperty(s, p) for p in propertynames(s)), h)
Base.isequal(s1::State, s2::State) = all(isequal(getproperty(s1, p), getproperty(s2, p)) for p in propertynames(s1))
Base.:(==)(s1::State, s2::State) = isequal(s1, s2)

@with_kw mutable struct Action
    site::Vector{Int64} = [0, 0, 0, 0]   # site: 0=DoNothing, 1-2=DLE, 3-4=Hard Rock Mining
    type::Vector{Int64} = [0, 0, 0, 0]    # type: 0=DoNothing, 1=explore, 2=open, 3=restore (no operate - mines produce automatically after opening)
end
Base.hash(a::Action, h::UInt) = hash(Tuple(getproperty(a, p) for p in propertynames(a)), h)
Base.isequal(a1::Action, a2::Action) = all(isequal(getproperty(a1, p), getproperty(a2, p)) for p in propertynames(a1))
Base.:(==)(a1::Action, a2::Action) = isequal(a1, a2)

@with_kw mutable struct Observation
    deposits::Vector{Float64} # [v₁, v₂, v₃, v₄]
    observed_price::Float64 
    observed_demand::Float64 
end
#makes sures there are no duplicates
Base.hash(o::Observation, h::UInt) = hash(Tuple(getproperty(o, p) for p in propertynames(o)), h)
Base.isequal(o1::Observation, o2::Observation) = all(isequal(getproperty(o1, p), getproperty(o2, p)) for p in propertynames(o1))
Base.:(==)(o1::Observation, o2::Observation) = isequal(o1, o2)

#only the unobservable has to be a distribution
@with_kw mutable struct LiBelief{T<:UnivariateDistribution} 
    deposits_distribution::Vector{UnivariateDistribution} # [v₁, v₂, v₃, v₄] # allow mixed LogNormal, etc
    time::Float64 = 1.0  # current time, year
    price_distribution::T  # Price per thousand
    demand_distribution::T  # Demand distribution
    total_mined::Float64  # track how much has been mined so far
    production_this_step::Float64 = 0.0  # track production for current time step
    have_opened::Vector{Bool}  # Boolean value to represent whether or not we have opened a mine
    is_depleted::Vector{Bool}  # Boolean value to represent whether a mine is depleted
    restored_mines::Vector{Bool}  # Boolean value to represent whether a mine has been restored
end
    

@with_kw mutable struct LiPOMDP <: POMDP{State,Action,Observation}
    #Observation parameters
    σ_obs_d_dle::Float64 = 60_000.0 # Observation noise for DLE deposits (sites 1&2)
    σ_obs_d_hard_rock::Float64 = 150_000.0 # Observation noise for Hard Rock deposits (sites 3&4)
    σ_obs_p::Float64 = 500.0 # Standard deviation of the observation noise for price
    σ_obs_demand::Float64 = 100.0  # Standard deviation for demand observation noise (reduced for lower error)

    price_init_std::Float64 = 1000.0   # Initial price uncertainty
    demand_init_std::Float64 = 500.0   # Initial demand uncertainty (reduced for lower error)

    γ::Float64 = 0.97 #discounted reward
    time_horizon::Int64 = 29 #time horizon
    n_deposits::Int64 = 4 # number of deposits
    alpha::Float64 = 0.5 # Parameter to control tradeoff between emissions and volume
    exploration_cost_dle::Float64 = 1_200_000.0  # Exploration cost for DLE sites
    exploration_cost_hard_rock::Float64 = 400_000.0  # Exploration cost for Hard Rock sites
    LCE_per_lithium = 5.323  # 1 t Li ≈ 5.323 t LCE

    # FIXED: Add emission scaling factor to balance scales between emissions and profit
    emission_scale_factor::Float64 = 1000 # Scale emissions up to balance with profit scale (increased from 3.0)
    reward_scale::Float64 = 10_000_000_000.0  # Manual reward scaling parameter (divide by this to normalize rewards)

    # Add market parameters
    market_share_target::Float64 = 0.30  # Target 30% of global market
    demand_uncertainty::Float64 = 0.05    # 5% std dev in demand forecasts (reduced for lower error)
    
    # Production parameters (deposits are in-situ Li metal tonnes)
    # output is NAMEPLATE Li t/yr per mine
    output::Float64 = 25_000.0  # Annual output per mine of Li
    # In-situ Li (t Li): LCE = deposit × recovery × LCE_per_lithium
    init_deposit_size_dle::Float64 = 160_165   # → 682,050 t LCE sellable (recovery 0.80)
    init_deposit_size_mining::Float64 = 1_237_427 # → 4,610,780 t LCE sellable (recovery 0.70)
    dle_uncertainty_factor::Float64 = 0.3  # Reduced to prevent negative deposits
    mining_uncertainty_factor::Float64 = 0.2 # Reduced to prevent negative deposits

    # Mining method for each deposit (true = DLE, false = hard rock mining)
    deposit_types::Vector{Bool} = [true, true, false, false]  # First two are DLE, last two Hard Rock Mining
    recovery::Vector{Float64} = [0.80, 0.80, 0.70, 0.70]    # Fraction of in‑situ Li recovered as saleable product
    CO2_cost::Int64 = 190 #according to EPA, average cost of CO2 capture and storage is $190/t CO2

    # Hard Rock mining parameters
    mining_CO2_emissions::Float64 = 6.25  # kg CO2 per tonne LCE (production only)
    mining_capex::Int64 = 3_670_000_000    # Higher initial investment
    mining_opex::Int64 = 5_173     #  Similar opeartinal costs per ton 

    # DLE parameters
    dle_CO2_emissions::Float64 = 4.1   # kg CO2 per tonne LCE (production only)
    dle_capex::Int64 = 1_209_000_000       # Lower initial investment
    dle_opex::Int64 = 5_694        # Similar operating costs per ton
    
    # Restoration parameters
    dle_restoration_cost::Int64 = 10_000_000  # Cost to restore depleted DLE mine
    mining_restoration_cost::Int64 = 20_000_000  # Cost to restore depleted mining mine

    #pricing model parameters
    price_model_type::Int64 = 1  # 1 static, 2 linear, 3 exponential, 4 GBM, 6 historical
    hist_data::Dict{Int64, Tuple{Float64, Float64}} = lithium_data  # Historical data as (production, price) tuples by year
    start_year::Int64 = 1995  # Starting year for historical data
    # Starting prices for each pricing model
    p₀_static::Float64 = 10_355.33    # Starting value for static pricing
    p₀_linear::Float64 = 2_381.38     # Starting value for linear pricing  
    p₀_exponential::Float64 = 917.36  # Starting value for exponential pricing
    p₀_gbm::Float64 = 10_000.0        # Starting value for GBM pricing
    p₀_fallback::Float64 = 10_000.0   # Fallback value
    
    α_slope::Float64 = 878.39   # linear slope (for linear pricing), increasing 10,000 dollars every year 
    λ_exp::Float64 = 0.1261     # exponential growth rate, changing to 0.1261
    
    # GBM (Geometric Brownian Motion) parameters
    gbm_drift::Float64 = 0.033      
    gbm_volatility::Float64 = 0.238 
end

# Helper function to get the appropriate starting price based on model type
function get_p₀(P::LiPOMDP)
    if P.price_model_type == 1
        return P.p₀_static
    elseif P.price_model_type == 2
        return P.p₀_linear
    elseif P.price_model_type == 3
        return P.p₀_exponential
    elseif P.price_model_type == 4
        return P.p₀_gbm
    else
        return P.p₀_fallback
    end
end

@with_kw mutable struct LiBeliefUpdater <: Updater #TODO understand this
    P::LiPOMDP
end

# Allow sampling states from beliefs for POMCPOW
function Base.rand(rng::AbstractRNG, b::LiBelief)
    # Sample deposit volumes
    sampled_deposits = [max(0.0, rand(rng, d)) for d in b.deposits_distribution]  # clamp ≥ 0
    
    # Sample price and demand
    sampled_price = max(100.0, rand(rng, b.price_distribution))  # Ensure minimum price
    sampled_demand = max(0.0, rand(rng, b.demand_distribution))  # Ensure non-negative demand
    
    # Create state with sampled values and deterministic values from belief
    return State(
        deposit = sampled_deposits,
        time = b.time,
        price = sampled_price,
        company_demand = sampled_demand,
        total_mined = b.total_mined,
        have_opened = copy(b.have_opened),
        is_depleted = copy(b.is_depleted),
        restored_mines = copy(b.restored_mines)
    )
end
