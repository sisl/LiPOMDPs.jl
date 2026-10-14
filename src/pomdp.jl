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
    max_sellable_lce = max_in_situ_li * P.recovery[site] * P.LCE_per_lithium

    sellable_lce = min(max_sellable_lce, current_demand_lce)

    # recompute the in-situ actually needed for the sellable chosen
    in_situ_li = sellable_lce / (P.recovery[site] * P.LCE_per_lithium)

    return in_situ_li, sellable_lce
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
                ef_lce = P.deposit_types[site] ? P.dle_CO2_emissions : P.mining_CO2_emissions  # kg CO2 per t LCE
                total_CO2_tons += sellable_lce * ef_lce / 1000.0  # Convert kg to tonnes CO2
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
            costs += opex * site_production_lce
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


function update_price(P::LiPOMDP, s::State, a::Action, price_model_type=1)
    t = s.time
    Δt = 1.0  # step size -- is 1 year for simplicity
    # Get the appropriate starting price based on model type
    p₀ = get_p₀(P)
    
    # 1. Static Pricing
    if P.price_model_type == 1
        # Use proportional noise for static pricing
        noise = rand(Normal(0, p₀ * 0.1))  # 5% noise relative to base price
        new_price = p₀ + noise
        return max(new_price, 1000.0)  # Ensure minimum price
    end

    # 2. Linear Pricing
    if P.price_model_type == 2
        base_price = p₀ + P.α_slope * t  # Start from p₀ and add slope * time
        # Use proportional noise relative to current price level
        noise = rand(Normal(0, abs(base_price) * 0.1))  # 10% noise
        new_price = base_price + noise
        return max(new_price, 1000.0)  # Ensure minimum price
    end

    # 3. Exponential Pricing
    if P.price_model_type == 3
        base_price = p₀ * exp(P.λ_exp * t)  # Start from p₀ and grow exponentially
        
        # Make noise proportional to the base price (e.g., 10% standard deviation)
        noise_std = base_price * 0.1  # 10% of base price
        noise = rand(Normal(0, noise_std))
        
        new_price = base_price + noise
        return max(new_price, 1000.0)  # Ensure minimum price
    end
    
    # 4. Geometric Brownian Motion (GBM) Pricing
    if P.price_model_type == 4
        # One-step GBM: S(t+Δt) = S(t) * exp((μ - σ²/2)Δt + σ dW), dW ~ N(0, Δt)
        # Use current price in state; p₀ is only for initial price (initialstate)
        dW = rand(Normal(0, sqrt(Δt)))
        new_price = s.price * exp((P.gbm_drift - P.gbm_volatility^2/2) * Δt + P.gbm_volatility * dW)
        
        return max(new_price, 1000.0)  # Ensure minimum price
    end

    # 6. Historical Pricing
    if P.price_model_type == 6
        # Use historical data from historical_data.jl
        # Map simulation time to historical years (starting from 1995)
        historical_year = 1995 + Int(t) - 1
        
        # Get historical price for this year
        if haskey(lithium_data, historical_year)
            historical_price = lithium_data[historical_year][2]  # Price is the second element
            # Add some noise to historical price for realism
            noise = rand(Normal(0, historical_price * 0.1))  # 10% noise
            new_price = historical_price + noise
            return max(new_price, 1000.0)  # Ensure minimum price
        else
            # If year is beyond historical data, use last available price with trend
            last_year = maximum(keys(lithium_data))
            last_price = lithium_data[last_year][2]
            # Simple extrapolation with some growth
            years_beyond = historical_year - last_year
            growth_factor = 1.05^years_beyond  # 5% annual growth
            extrapolated_price = last_price * growth_factor
            noise = rand(Normal(0, extrapolated_price * 0.1))  # 10% noise for extrapolation
            new_price = extrapolated_price + noise
            return max(new_price, 1000.0)
        end
    end
    
    # Fallback case
    return max(p₀, 1000.0)
end

# Add function to calculate company's demand
function update_demand(P::LiPOMDP, global_production::Float64)
    # Use global production instead of demand as basis for company target
    # Convert global production from Li metal to LCE
    global_production_lce = global_production * P.LCE_per_lithium
    
    # Add uncertainty to global production forecast
    production_noise = global_production_lce * P.demand_uncertainty
    uncertain_global_production = max(0.0, global_production_lce + rand(Normal(0, production_noise)))
    
    # Calculate company's target demand based on market share of global production
    company_demand = uncertain_global_production * P.market_share_target
    
    return (company_demand)
end

#transition function, observation function, reward function


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
            σ_obs_base = i <= 2 ? P.σ_obs_d_dle : P.σ_obs_d_hard_rock
            noise = σ_obs_base * 2.0  
            # Truncate to ensure non-negative observations
            mean_deposit = max(0.0, sp.deposit[i])
            push!(deposit_estimate_dist, Normal(mean_deposit, noise))
        elseif is_open
            # Production-based observation: learn from mining activity
            # Production provides VERY precise observations through actual extraction
            # This should be MORE precise than exploration
            # Use deposit-type-specific observation noise
            σ_obs_base = i <= 2 ? P.σ_obs_d_dle : P.σ_obs_d_hard_rock
            noise = σ_obs_base * 0.1  # 10x LESS noise - production is more precise than exploration
            # Truncate to ensure non-negative observations
            mean_deposit = max(0.0, sp.deposit[i])
            push!(deposit_estimate_dist, Normal(mean_deposit, noise))
        else
            # No information: unexplored and not producing
            # Use deposit-type-specific observation noise
            σ_obs_base = i <= 2 ? P.σ_obs_d_dle : P.σ_obs_d_hard_rock
            noise = σ_obs_base * 50.0  
            # Truncate to ensure non-negative observations
            mean_deposit = max(0.0, sp.deposit[i])
            push!(deposit_estimate_dist, Normal(mean_deposit, noise))
        end
    end

    price_estimate_dist = Normal(sp.price, P.σ_obs_p)
    demand_estimate_dist = Normal(sp.company_demand, P.σ_obs_demand)
    
    # Combine all distributions
    prod_dist = product_distribution(deposit_estimate_dist..., price_estimate_dist, demand_estimate_dist)
    return prod_dist
end

#have a POMDP that updates, takes the POMDP as the input and the updater
#original belief, action you take and observation
function POMDPs.update(up::LiBeliefUpdater, b::LiBelief, a::Action, o::Observation)
    P = up.P  # Access the POMDP parameters

    # Initialize new vector of updated distributions for deposits
    updated_deposits = Vector{UnivariateDistribution}(undef, length(o.deposits))

    # Loop through each deposit site to update its belief
    for i in 1:length(o.deposits)
        # Check if this site was explored in the action
        was_explored = false
        for j in 1:length(a.site)
            if a.type[j] == 1 && a.site[j] == i
                was_explored = true
                break
            end
        end
        
        # Check if this site is currently open (producing)
        is_open = b.have_opened[i] || any(a.type[j] == 2 && a.site[j] == i for j in 1:length(a.site))
        
        if was_explored
            # Direct exploration: high-quality observation using Kalman filter
            prior = b.deposits_distribution[i]      # prior belief (Normal) for site i
            μ_prior, σ_prior = mean(prior), std(prior)
            z = o.deposits[i]                        # observed noisy deposit value for site i

            # Use different noise levels based on deposit type for balanced convergence
            # DLE sites (1&2): lower noise for faster convergence
            # Hard Rock sites (3&4): higher noise for realistic uncertainty
            σ_obs_base = i <= 2 ? P.σ_obs_d_dle : P.σ_obs_d_hard_rock  # DLE vs Hard Rock observation noise
            σ_obs2 = max((σ_obs_base * 1.5)^2, 1e-10)  # 1x base noise for exploration
            σ_prior2 = max(σ_prior^2, 1e-10)  # Prevent division by zero
            σ_post2 = 1 / (1/σ_prior2 + 1/σ_obs2)     # updated variance
            μ_post = σ_post2 * (μ_prior/σ_prior2 + z/σ_obs2)  # updated mean (weighted avg)

            # Ensure minimum uncertainty floor based on initial deposit size (not current estimate)
            # This prevents uncertainty from going to zero even with many observations
            initial_deposit_size = i <= 2 ? P.init_deposit_size_dle : P.init_deposit_size_mining
            min_uncertainty = max(500.0, 0.01 * initial_deposit_size)  # 1% of initial deposit
            σ_post_final = max(sqrt(σ_post2), min_uncertainty)

            # Store the updated Normal distribution with truncation to prevent negatives
            updated_deposits[i] = truncated(Normal(μ_post, σ_post_final), 0.0, Inf)
        elseif is_open
            # Production-based observation: learn from mining activity
            prior = b.deposits_distribution[i]      # prior belief (Normal) for site i
            μ_prior, σ_prior = mean(prior), std(prior)
            z = o.deposits[i]                        # observed noisy deposit value for site i

            # Use low noise for production observations (production is more precise than exploration)
            # Use same deposit-type-specific base noise as exploration
            σ_obs_base = i <= 2 ? P.σ_obs_d_dle : P.σ_obs_d_hard_rock  # DLE vs Hard Rock observation noise
            σ_obs2 = max((σ_obs_base * 0.1)^2, 1e-10)  # 10x LESS noise for production
            σ_prior2 = max(σ_prior^2, 1e-10)  # Prevent division by zero
            σ_post2 = 1 / (1/σ_prior2 + 1/σ_obs2)     # updated variance
            μ_post = σ_post2 * (μ_prior/σ_prior2 + z/σ_obs2)  # updated mean (weighted avg)

            # Ensure minimum uncertainty floor based on initial deposit size (not current estimate)
            # This prevents uncertainty from going to zero even with many observations
            initial_deposit_size = i <= 2 ? P.init_deposit_size_dle : P.init_deposit_size_mining
            min_uncertainty = max(500.0, 0.01 * initial_deposit_size)  # 1% of initial deposit
            σ_post_final = max(sqrt(σ_post2), min_uncertainty)

            # Store the updated Normal distribution with truncation to prevent negatives
            updated_deposits[i] = truncated(Normal(μ_post, σ_post_final), 0.0, Inf)
        else
            # Keep prior belief unchanged for unexplored and non-producing sites
            updated_deposits[i] = b.deposits_distribution[i]
        end
    end

    #PRICE
    #Update the price belief
    prior_p = b.price_distribution                   # prior belief over price
    μp, σp = mean(prior_p), std(prior_p)
    zp = o.observed_price                                 # observed noisy price

    # Prevent division by zero in price belief update
    σp2 = max(σp^2, 1e-10)
    σ_obs_p2 = max(P.σ_obs_p^2, 1e-10)
    σp_post2 = 1 / (1/σp2 + 1/σ_obs_p2)         # updated variance for price
    μp_post = σp_post2 * (μp/σp2 + zp/σ_obs_p2) # updated mean for price
    updated_price = Normal(μp_post, sqrt(σp_post2))

    #DEMAND
    # Update the demand belief
    prior_d = b.demand_distribution                   # prior belief over demand
    μd, σd = mean(prior_d), std(prior_d)
    zd = o.observed_demand                       # observed noisy demand

    # Prevent division by zero in demand belief update
    σd2 = max(σd^2, 1e-10)
    σ_obs_demand2 = max(P.σ_obs_demand^2, 1e-10) #TODO why does demand shrink
    σd_post2 = 1 / (1/σd2 + 1/σ_obs_demand2)  # updated variance for demand
    μd_post = σd_post2 * (μd/σd2 + zd/σ_obs_demand2) # updated mean for demand
    updated_demand = Normal(μd_post, sqrt(σd_post2))

    # update have_opened status based on action
    updated_have_opened = copy(b.have_opened)

    for site in 1:P.n_deposits
        if a.type[site] == 2 && !b.have_opened[site]
            updated_have_opened[site] = true
        end
    end

    # Update depleted status based on actions and production
    updated_is_depleted = copy(b.is_depleted)
    updated_restored_mines = copy(b.restored_mines)
    
    # Check for restoration actions
    for site in 1:length(a.site)
        if a.type[site] == 3 && a.site[site] != 0  # Restore action
            # Keep is_depleted as true - mine stays depleted after restoration
            updated_restored_mines[site] = true  # Mark as restored
            # FIXED: Set deposit distribution to 0 when restoring - mine is completely depleted
            updated_deposits[site] = Normal(0.0, 0.1)  # Very small distribution around 0
        end
    end
    
    # Check for depletion based on observations
    # For opened mines, check if they should be depleted based on the observed deposit
    for site in 1:length(updated_deposits)
        if updated_have_opened[site] && !updated_is_depleted[site] && !updated_restored_mines[site]
            # Check if the observed deposit is below depletion threshold (5% of initial size)
            observed_deposit = o.deposits[site]
            initial_deposit_size = site <= 2 ? P.init_deposit_size_dle : P.init_deposit_size_mining
            depletion_threshold = 0.05 * initial_deposit_size  # 5% of initial deposit
            if observed_deposit <= depletion_threshold  # Marks depleted when observed deposit is essentially zero
                updated_is_depleted[site] = true
                # Update the deposit distribution to reflect depletion
                updated_deposits[site] = Normal(0.0, 0.1)  # Very small distribution around 0
            end
        end
    end

 # FIXED: Use centralized production calculation with proper demand tracking
    # Create a temporary state-like object for production calculation
    temp_deposits = [mean(updated_deposits[i]) for i in 1:P.n_deposits]
    expected_demand = mean(updated_demand)
    remaining_demand = expected_demand
    
    expected_production_this_step = 0.0
    
    # Process each site in order (same as calculate_total_production)
    for site in 1:P.n_deposits
        if updated_have_opened[site] && temp_deposits[site] > 0
            # Use PRIOR deposit belief (before observation update) for production calculation
            prior_deposit_mean = mean(b.deposits_distribution[site])
            
            # Create temporary state for this site
            temp_state = (
                deposit = temp_deposits,
                have_opened = updated_have_opened,
                restored_mines = updated_restored_mines
            )
            
            # Use the centralized production calculation
            in_situ, sellable = calculate_site_production_for_belief(
                P, temp_state, site, remaining_demand, prior_deposit_mean
            )
            
            expected_production_this_step += sellable
            remaining_demand -= sellable  # Track remaining demand
               
            # Update the deposit belief to account for mining depletion
            # Reduce the mean by the amount mined (in_situ, not sellable)
            current_mean = mean(updated_deposits[site])
            current_std = std(updated_deposits[site])
            depleted_mean = max(0.0, current_mean - in_situ)  # FIXED: subtract in_situ
            updated_deposits[site] = Normal(depleted_mean, current_std)
        end
    end

    updated_total_mined = b.total_mined + expected_production_this_step

    # Construct and return the updated belief 
    return LiBelief(
        deposits_distribution = updated_deposits,  # new beliefs over each site's deposit
        time = b.time + 1,                        # increment time
        price_distribution = updated_price,       # updated price belief
        demand_distribution = updated_demand,     # updated demand distribution
        total_mined = updated_total_mined,
        have_opened = updated_have_opened,        # track which mines are opened
        is_depleted = updated_is_depleted,        # track which mines are depleted
        restored_mines = updated_restored_mines   # track which mines have been restored
    )
end

# Helper function for belief updates - avoids creating full State objects
function calculate_site_production_for_belief(P::LiPOMDP, temp_state, site::Int, current_demand_lce::Float64, prior_deposit_li::Float64)
    if !temp_state.have_opened[site] || temp_state.deposit[site] <= 0.1 || prior_deposit_li <= 0.1 || temp_state.restored_mines[site]
        return 0.0, 0.0
    end
    
    # capacity (in-situ Li) from Li nameplate:
    # P.output is already in Li tonnes per year
    cap_in_situ_li = P.output

    # in-situ Li we can actually take this period:
    max_in_situ_li = min(cap_in_situ_li, prior_deposit_li)

    # sellable LCE from that in-situ Li:
    max_sellable_lce = max_in_situ_li * P.recovery[site] * P.LCE_per_lithium

    sellable_lce = min(max_sellable_lce, current_demand_lce)

    # recompute the in-situ actually needed for the sellable chosen
    in_situ_li = sellable_lce / (P.recovery[site] * P.LCE_per_lithium)
    
    return in_situ_li, sellable_lce
end


@with_kw struct InitialStateDistribution
    deposit_dists::Vector{Normal} # one Normal per site
    price_dist::Normal                         
    demand_dist::Normal                        
    have_opened0::Vector{Bool}                 
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
        restored_mines = fill(false, length(deposits))
    )
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
        deposit_dists = [Normal(means[i], stds[i]) for i in 1:P.n_deposits],
        price_dist    = Normal(price0,  price_std),
        demand_dist   = Normal(demand0, demand_std),
        have_opened0  = fill(false, P.n_deposits)
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
        restored_mines = fill(false, P.n_deposits)
    )
    true_state = rand(initialstate(P))  # Get the true initial state
    
    return true_state, belief  # Return both state and belief
end



#given a state s and action a it outputs the distribution of the next state sp
function POMDPs.transition(P::LiPOMDP, s::State, a::Action)
    sp = deepcopy(s)
    sp.time += 1
    
    # Update price
    current_year = P.start_year + Int(floor(sp.time)) - 1
    sp.price = update_price(P, s, a)
    
    # Get production data (using production instead of demand for company target)
    global_production = get_historical_production(P, current_year)
    
    # Update company demand based on global production with uncertainty
    sp.company_demand = update_demand(P, global_production)

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
    total_mined_this_step = 0.0
    remaining_demand_lce = s.company_demand  # Use OLD demand for production calculation
    
    for site in 1:P.n_deposits
        if sp.have_opened[site] && sp.deposit[site] > 0.1 && !sp.is_depleted[site] && !sp.restored_mines[site]
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

    return Deterministic(sp)
end

#if the process similar then use it as a nindex but if its different then create branch
#last line has to be deterministic 
#each action type will affect the state being updated 
#produce at capacity -- rate capacity

# Update the gen function to handle demand observations
function POMDPs.gen(P::LiPOMDP, s::State, a::Action, rng::AbstractRNG)
    
    sp = rand(transition(P, s, a))

    o_array = rand(observation(P, s, a, sp))
    # Truncate deposit observations to ensure they're non-negative
    deposits_obs = [max(0.0, obs) for obs in o_array[1:end-2]]
    o = Observation(
        deposits=deposits_obs,  # All but last two elements are deposits, truncated to ≥ 0
        observed_price=o_array[end-1],  # Second to last is price
        observed_demand=o_array[end]    # Last element is demand
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
        if is_action_valid(action, s, P.n_deposits)
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
    max_possible_sellable = max_possible_in_situ * P.recovery[i] * P.LCE_per_lithium
    
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
    max_possible_sellable = max_possible_in_situ * P.recovery[i] * P.LCE_per_lithium
    
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
    total_lifetime_emissions = lifetime_variable_emissions / 1000.0  # Convert kg to tonnes
    
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
    o_vec = vcat(o.deposits, o.observed_price, o.observed_demand)
    
    # Calculate pdf
    return pdf(o_dist, o_vec)
end
