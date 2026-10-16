""" Select top k sites to open at period times based on profit maximization
"""

"""
Calculate expected average price over a mine's lifetime using the exact same logic as update_price.
This accounts for price appreciation over time and includes the same noise model.
"""
function calculate_lifetime_average_price(P::LiPOMDP, current_time::Float64, lifetime_years::Int)
    p₀ = get_p₀(P)
    total_price = 0.0
    
    for year in 1:lifetime_years
        t = current_time + year - 1
        
        if P.price_model_type == 1  # Static
            base_price = p₀
            noise_std = p₀ * 0.05  # 5% noise relative to base price
        elseif P.price_model_type == 2  # Linear
            base_price = p₀ + P.α_slope * t
            noise_std = abs(base_price) * 0.1  # 10% noise
        elseif P.price_model_type == 3  # Exponential
            base_price = p₀ * exp(P.λ_exp * t)
            noise_std = base_price * 0.1  # 10% of base price
        elseif P.price_model_type == 6  # Historical
            historical_year = 1995 + Int(t) - 1
            if haskey(lithium_data, historical_year)
                base_price = lithium_data[historical_year][2]  # Historical price
            else
                # Extrapolate beyond historical data
                last_year = maximum(keys(lithium_data))
                last_price = lithium_data[last_year][2]
                years_beyond = historical_year - last_year
                growth_factor = 1.05^years_beyond
                base_price = last_price * growth_factor
            end
            noise_std = base_price * 0.1  # 10% of base price
        else
            base_price = p₀  # Fallback
            noise_std = p₀ * 0.05
        end
        
        # Use deterministic price (no noise) for consistent policy decisions
        year_price = base_price
        year_price = max(year_price, 1000.0)  # Ensure minimum price (same as update_price)
        total_price += year_price
    end
    
    return total_price / lifetime_years  # Average base price over lifetime
end


@with_kw struct ProfitMaximizerPolicy <: Policy
    pomdp::LiPOMDP
    debug::Bool
    top_k::Int=1 # number of sites to open
    times::Vector{Int}=[1] # vector of times to open the sites
end

ProfitMaximizerPolicy(pomdp::LiPOMDP; debug=false, top_k=1, times=[1]) = 
    ProfitMaximizerPolicy(pomdp, debug, top_k, times)

function POMDPs.action(policy::ProfitMaximizerPolicy, s::State)
    if policy.debug
        println("\n[DEBUG] ProfitMaximizerPolicy evaluating state:")
        println("  Time: $(s.time)")
        println("  Price: $(round(s.price, digits=2))")
        println("  Deposits: $(round.(s.deposit, digits=1))")
        println("  Have opened: $(s.have_opened)")
    end
    
    # Check for restoration opportunities first (outside of time restrictions)
    opened_sites = zeros(Int, policy.pomdp.n_deposits)
    opened_types = zeros(Int, policy.pomdp.n_deposits)

    for site in 1:policy.pomdp.n_deposits
        if s.have_opened[site] && s.is_depleted[site] && !s.restored_mines[site]
            # Restore depleted mine
            opened_sites[site] = site
            opened_types[site] = 3  # Restore action
            
            if policy.debug
                println("  [DEBUG] Restoring site $site: Mine is depleted and not yet restored")
            end
        end
    end

    # Only proceed with opening new mines if we're at an allowed time
    if s.time in policy.times
        # Find the most profitable mine to open
        profits = zeros(policy.pomdp.n_deposits)
        remaining_time = policy.pomdp.time_horizon - s.time
        for site in 1:policy.pomdp.n_deposits
            if !s.have_opened[site] && s.deposit[site] > 0
                # Calculate realistic lifetime profit for opening this site now
                # Consider both deposit size AND demand constraints
                max_possible_in_situ = min(s.deposit[site], policy.pomdp.output * remaining_time)
                max_possible_sellable = max_possible_in_situ * policy.pomdp.recovery[site]
                
                # But production is also limited by demand - calculate realistic production
                # Assume we can capture a reasonable share of demand over time
                avg_demand_per_period = s.company_demand  # Current period demand
                realistic_lifetime_demand = avg_demand_per_period * remaining_time
                realistic_sellable = min(max_possible_sellable, realistic_lifetime_demand)
                
                # Calculate lifetime revenue using dynamic price forecasting
                # Account for price appreciation over the mine's lifetime
                lifetime_years = Int(remaining_time)  # Mine operates for remaining time horizon
                
                # Calculate average expected price over mine lifetime
                avg_lifetime_price = calculate_lifetime_average_price(policy.pomdp, s.time, Int(lifetime_years))
                
                # Use the exact price model without discount factor
                lifetime_revenue = realistic_sellable * avg_lifetime_price
                
                # Calculate lifetime costs to match actual cost structure
                capex = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_capex : policy.pomdp.mining_capex
                opex = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_opex : policy.pomdp.mining_opex
                lifetime_opex = opex * realistic_sellable  # OPEX on realistic sellable production
                
                # Calculate lifetime profit
                total_profit = lifetime_revenue - lifetime_opex - capex
                
                profits[site] = total_profit
                
                if policy.debug
                    println("  [DEBUG] Site $site: current_price=$(round(s.price, digits=2)), avg_lifetime_price=$(round(avg_lifetime_price, digits=2)), lifetime_years=$lifetime_years, realistic_sellable=$(round(realistic_sellable, digits=1)), revenue=$(round(lifetime_revenue/1e6, digits=2))M, opex=$(round(lifetime_opex/1e6, digits=2))M, capex=$(round(capex/1e6, digits=2))M, profit=$(round(total_profit/1e6, digits=2))M")
                end
            end
        end
    
        ranked_sites = sortperm(profits, rev=true)
        
        for i in 1:policy.top_k
            opened_sites[ranked_sites[i]] = ranked_sites[i]  # Set to actual site number
            opened_types[ranked_sites[i]] = 2
        end

        return Action(site=opened_sites, type=opened_types)
    else
        if policy.debug
            println("  [DEBUG] ProfitMaximizerPolicy not activated at time $(s.time), but checking for restoration")
        end
        # Even if not at an allowed time, return any restoration actions that were set
        return Action(site=opened_sites, type=opened_types)
    end
    
end

function POMDPs.action(policy::ProfitMaximizerPolicy, b::LiBelief)
    if policy.debug
        println("\n[DEBUG] ProfitMaximizerPolicy evaluating belief")
    end
    
    # Sample from belief distributions like POMCPOW does for consistency
    sampled_deposits = [rand(d) for d in b.deposits_distribution]  # Sample from distributions
    sampled_price = rand(b.price_distribution)  # Sample from distributions
    sampled_demand = rand(b.demand_distribution)  # Sample from distributions
    
    # Create a representative state
    s = State(
        deposit = sampled_deposits,
        time = b.time,
        price = sampled_price,
        company_demand = sampled_demand,
        total_mined = b.total_mined,
        have_opened = copy(b.have_opened),
        is_depleted = copy(b.is_depleted),
        restored_mines = copy(b.restored_mines)
    )
    
    return POMDPs.action(policy, s)
end


"""EmissionMinimizerPolicy - A policy that minimizes emissions
"""

@with_kw struct EmissionMinimizerPolicy <: Policy
    pomdp::LiPOMDP
    debug::Bool
    top_k::Int=1 # number of sites to open
    times::Vector{Int}=[1] # vector of times to open the sites
end

EmissionMinimizerPolicy(pomdp::LiPOMDP; debug=false, top_k=1, times=[1]) = EmissionMinimizerPolicy(pomdp, debug, top_k, times)

function POMDPs.action(policy::EmissionMinimizerPolicy, s::State)
    if policy.debug
        println("\n[DEBUG] EmissionMinimizerPolicy evaluating state:")
        println("  Time: $(s.time)")
        println("  Price: $(round(s.price, digits=2))")
        println("  Deposits: $(round.(s.deposit, digits=1))")
        println("  Have opened: $(s.have_opened)")
    end

    # Check for restoration opportunities first (outside of time restrictions)
    opened_sites = zeros(Int, policy.pomdp.n_deposits)
    opened_types = zeros(Int, policy.pomdp.n_deposits)

    for site in 1:policy.pomdp.n_deposits
        if s.have_opened[site] && s.is_depleted[site] && !s.restored_mines[site]
            # Restore depleted mine
            opened_sites[site] = site
            opened_types[site] = 3  # Restore action
            
            if policy.debug
                println("  [DEBUG] Restoring site $site: Mine is depleted and not yet restored")
            end
        end
    end

    # Only proceed with opening new mines if we're at an allowed time
    if s.time in policy.times
        # Find the least emitting mine to open
        emissions = zeros(policy.pomdp.n_deposits)
        remaining_time = policy.pomdp.time_horizon - s.time
        for site in 1:policy.pomdp.n_deposits
            if !s.have_opened[site] && s.deposit[site] > 0
                # Calculate lifetime emissions for opening this site now
                remaining_time = max(policy.pomdp.time_horizon - s.time, 0)
                max_in_situ = min(s.deposit[site], policy.pomdp.output * remaining_time)
                max_sellable = max_in_situ * policy.pomdp.recovery[site]
                
                # Note: Emissions don't depend on price, so we don't need price forecasting here
                # But we could add price-dependent emission factors in the future if needed
                
                # Variable emissions (per in-situ ton produced over lifetime)
                var_emissions_per_ton = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_CO2_emissions : policy.pomdp.mining_CO2_emissions
                lifetime_var_emissions = max_in_situ * var_emissions_per_ton
                
        
                # Total lifetime emissions (production only)
                total_lifetime_emissions = lifetime_var_emissions / 1000.0  # Convert kg to tonnes
                
                emissions[site] = total_lifetime_emissions
                
                if policy.debug
                    println("  [DEBUG] Site $site: lifetime emissions=$(round(total_lifetime_emissions, digits=2)) tCO2 (production-only)")
                end
            end
        end


        ranked_sites = sortperm(emissions)
        
        for i in 1:policy.top_k
            opened_sites[ranked_sites[i]] = ranked_sites[i]  # Set to actual site number
            opened_types[ranked_sites[i]] = 2
        end

        if policy.debug
            for i in 1:policy.pomdp.n_deposits
                if opened_types[i] == 2
                    println("  [DEBUG] Opening site $i with emissions $(round(emissions[i], digits=2)) tCO2")
                end
            end
        end

        return Action(site=opened_sites, type=opened_types)
    else
        if policy.debug
            println("  [DEBUG] EmissionMinimizerPolicy not activated at time $(s.time), but checking for restoration")
        end
        # Even if not at an allowed time, return any restoration actions that were set
        return Action(site=opened_sites, type=opened_types)
    end
end

function POMDPs.action(policy::EmissionMinimizerPolicy, b::LiBelief)
    if policy.debug
        println("\n[DEBUG] EmissionMinimizerPolicy evaluating belief")
    end
    
    # Use mean values from belief for stable decisions
    # Sample from belief distributions like POMCPOW does for consistency
    sampled_deposits = [max(0.0, rand(d)) for d in b.deposits_distribution]  # Sample from distributions
    sampled_price = rand(b.price_distribution)  # Sample from distributions
    sampled_demand = rand(b.demand_distribution)  # Sample from distributions
    
    # Create a representative state
    s = State(
        deposit = sampled_deposits,
        time = b.time,
        price = sampled_price,
        company_demand = sampled_demand,
        total_mined = b.total_mined,
        have_opened = copy(b.have_opened),
        is_depleted = copy(b.is_depleted),
        restored_mines = copy(b.restored_mines)
    )
    
    return POMDPs.action(policy, s)
end

"""
ExploreOnlyPolicy - A heuristic policy that prioritizes exploration before opening
"bad policy"

Priority order:
1. Explore unexplored sites (to gather information)
2. Open explored sites that aren't opened
3. Do nothing if all sites are opened

This policy focuses on information gathering before making opening decisions.
"""
struct ExploreOnlyPolicy <: Policy
    pomdp::LiPOMDP
    debug::Bool
end

ExploreOnlyPolicy(pomdp::LiPOMDP) = ExploreOnlyPolicy(pomdp, false)

function POMDPs.action(policy::ExploreOnlyPolicy, s::State)
    if policy.debug
        println("\n[DEBUG] ExploreOnlyPolicy evaluating state:")
        println("  Time: $(s.time)")
        println("  Have opened: $(s.have_opened)")
        println("  Deposits: $(round.(s.deposit, digits=1))")
    end
    
    # Check for restoration opportunities first
    explore_actions = zeros(Int, 4)
    explore_sites = zeros(Int, 4)

    for i in 1:policy.pomdp.n_deposits
        if s.have_opened[i] && s.is_depleted[i] && !s.restored_mines[i]
            # Restore depleted mine
            explore_actions[i] = 3  # Restore action
            explore_sites[i] = i
            
            if policy.debug
                println("  [DEBUG] Restoring site $i: Mine is depleted and not yet restored")
            end
        elseif !s.have_opened[i]
            if policy.debug
                println("  [DEBUG] Choosing to explore site $i (unopened)")
            end            
            explore_actions[i] = 1
            explore_sites[i] = i
        end
    end

    if sum(explore_actions) > 0
        return Action(site=explore_sites, type=explore_actions)
    end
    
    # Fallback: do nothing if all sites are opened
    if policy.debug
        println("  [DEBUG] All sites opened, doing nothing")
    end
    return Action(site=[1, 2, 3, 4], type=[0, 0, 0, 0])
end

function POMDPs.action(policy::ExploreOnlyPolicy, b::LiBelief)
    if policy.debug
        println("\n[DEBUG] ExploreOnlyPolicy evaluating belief")
    end
    
    # Sample from belief distributions for consistency with other policies
    sampled_deposits = [rand(d) for d in b.deposits_distribution]  # Sample from distributions
    sampled_price = rand(b.price_distribution)  # Sample from distributions
    sampled_demand = rand(b.demand_distribution)  # Sample from distributions
    
    # Create a representative state
    s = State(
        deposit = sampled_deposits,
        time = b.time,
        price = sampled_price,
        company_demand = sampled_demand,
        total_mined = b.total_mined,
        have_opened = copy(b.have_opened),
        is_depleted = copy(b.is_depleted),
        restored_mines = copy(b.restored_mines)
    )
    
    return action(policy, s)
end

"""
RandomHeuristicPolicy - Baseline random policy for comparison
This policy makes random decisions for each mine independently at each time step.
"""
struct RandomHeuristicPolicy <: Policy
    pomdp::LiPOMDP
    rng::AbstractRNG
    debug::Bool
end

RandomHeuristicPolicy(pomdp::LiPOMDP; rng=Random.GLOBAL_RNG, debug=false) = RandomHeuristicPolicy(pomdp, rng, debug)

function POMDPs.action(policy::RandomHeuristicPolicy, s::State)
    if policy.debug
        println("\n[DEBUG] RandomHeuristicPolicy evaluating state:")
        println("  Time: $(s.time)")
        println("  Have opened: $(s.have_opened)")
        println("  Deposits: $(round.(s.deposit, digits=1))")
    end
    
    n_sites = policy.pomdp.n_deposits
    site_actions = zeros(Int, n_sites)
    site_types = zeros(Int, n_sites)
    
    for i in 1:n_sites
        if s.have_opened[i] && s.is_depleted[i]
            # Always restore depleted mines
            site_actions[i] = i
            site_types[i] = 3  # Restore action
            
            if policy.debug
                println("  [DEBUG] Restoring site $i: Mine is depleted")
            end
        elseif !s.have_opened[i]
            # Randomly choose action for each unopened mine:
            # 0 = do nothing, 1 = explore, 2 = open
            action_choice = rand(policy.rng, 0:2)
            
            if action_choice > 0  # If we're doing something (explore or open)
                site_actions[i] = i  # Set site identifier to actual site number
                site_types[i] = action_choice  # Set action type (1=explore, 2=open)
            end
            # If action_choice == 0, both arrays remain 0 (do nothing)
        end
        # If mine is already opened but not depleted, keep action as 0 (do nothing)
    end
    
    if policy.debug
        for i in 1:n_sites
            if site_actions[i] > 0
                action_name = site_actions[i] == 1 ? "explore" : "open"
                println("  [DEBUG] Randomly chose to $action_name site $i")
            end
        end
    end
    
    return Action(site=site_actions, type=site_types)
end

function POMDPs.action(policy::RandomHeuristicPolicy, b::LiBelief)
    if policy.debug
        println("\n[DEBUG] RandomHeuristicPolicy evaluating belief state")
    end
    
    # Sample a state from the belief
    s = rand(policy.rng, b)
    return action(policy, s)
end

"""
DynamicProfitMaximizerPolicy - A policy that maximizes profit at each time step
This policy can take actions on all mines and chooses actions that lead to more profit.
For each mine, it can explore, open, or do nothing based on expected profitability.
FIXED: Now includes better demand management for optimal profit maximization.
"""
struct DynamicProfitMaximizerPolicy <: Policy
    pomdp::LiPOMDP
    debug::Bool
end

DynamicProfitMaximizerPolicy(pomdp::LiPOMDP; debug=false) = DynamicProfitMaximizerPolicy(pomdp, debug)

function POMDPs.action(policy::DynamicProfitMaximizerPolicy, s::State)
    if policy.debug
        println("\n[DEBUG] DynamicProfitMaximizerPolicy evaluating state:")
        println("  Time: $(s.time)")
        println("  Price: $(round(s.price, digits=2))")
        println("  Deposits: $(round.(s.deposit, digits=1))")
        println("  Have opened: $(s.have_opened)")
        println("  Company demand: $(s.company_demand)")
    end
    
    n_sites = policy.pomdp.n_deposits
    site_actions = zeros(Int, n_sites)
    site_types = zeros(Int, n_sites)
    
    # Calculate profitability for each action on each mine
    remaining_time = policy.pomdp.time_horizon - s.time
    
    # Calculate current production capacity from already opened mines
    current_capacity = 0.0
    for site in 1:n_sites
        if s.have_opened[site] && s.deposit[site] > 0
            # Calculate this mine's capacity
            mine_capacity = policy.pomdp.output * policy.pomdp.recovery[site] * policy.pomdp.LCE_per_lithium
            current_capacity += mine_capacity
        end
    end
    
    if policy.debug
        println("  [DEBUG] Current capacity from opened mines: $(round(current_capacity, digits=1))")
        println("  [DEBUG] Current demand: $(round(s.company_demand, digits=1))")
    end
    
    # Calculate remaining demand
    remaining_demand = max(0.0, s.company_demand - current_capacity)
    
    # evaluate all sites first to see which can contribute
    site_evaluations = []
    
    for site in 1:n_sites
        if !s.have_opened[site]
            # Calculate expected production if we open this mine
            max_possible_production = min(s.deposit[site], policy.pomdp.output * remaining_time)
            expected_recovery = max_possible_production * policy.pomdp.recovery[site]
            
            if expected_recovery > 0
                # FIXED: Calculate realistic production based on deposit depletion
                # Calculate how many years this mine can operate before depletion
                years_to_depletion = s.deposit[site] / policy.pomdp.output
                
                # Calculate realistic lifetime production (limited by deposit size AND remaining demand)
                realistic_sellable = min(
                    expected_recovery,  # Max possible recovery
                    policy.pomdp.output * policy.pomdp.recovery[site] * policy.pomdp.LCE_per_lithium * years_to_depletion,  # Deposit-limited
                    remaining_demand * remaining_time  # Demand-limited over lifetime
                )
                
                # FIXED: Use lifetime average price to account for price changes over time
                lifetime_years = Int(remaining_time)
                avg_lifetime_price = calculate_lifetime_average_price(policy.pomdp, s.time, lifetime_years)
                expected_revenue = realistic_sellable * avg_lifetime_price
                
                # Calculate costs
                capex = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_capex : policy.pomdp.mining_capex
                opex = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_opex : policy.pomdp.mining_opex
                lifetime_opex = opex * realistic_sellable  # OPEX on realistic sellable production
                
                # FIXED: Calculate emission costs
                # Variable emissions (per tonne LCE produced)
                variable_emissions_per_tonne = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_CO2_emissions : policy.pomdp.mining_CO2_emissions
                lifetime_variable_emissions = realistic_sellable * variable_emissions_per_tonne
                
                # Removed: No fixed emissions
                # Removed: No startup emissions
                # Emissions are ONLY based on production
                
                # Total emissions cost (production only, convert kg to tonnes)
                total_emissions_cost = (lifetime_variable_emissions / 1000.0) * policy.pomdp.CO2_cost
                
                # FIXED: Calculate total profit including environmental costs
                # Total profit = revenue - operational costs - environmental costs
                total_profit = expected_revenue - lifetime_opex - capex - total_emissions_cost
                
                # Store evaluation for later decision making
                push!(site_evaluations, (site, total_profit, expected_recovery, expected_revenue))
                
                if policy.debug
                    println("  [DEBUG] Site $site: current_price=$(round(s.price, digits=2)), avg_lifetime_price=$(round(avg_lifetime_price, digits=2)), realistic_production=$(round(realistic_sellable, digits=1)), years_to_depletion=$(round(years_to_depletion, digits=1))")
                    println("    Remaining demand: $(round(remaining_demand, digits=1)), Demand-limited production: $(round(remaining_demand * remaining_time, digits=1))")
                    println("    Revenue: \$$(round(expected_revenue/1e6, digits=2))M, OPEX: \$$(round(lifetime_opex/1e6, digits=2))M, CAPEX: \$$(round(capex/1e6, digits=2))M")
                    println("    Emissions cost: \$$(round(total_emissions_cost/1e6, digits=2))M, Total Profit: \$$(round(total_profit/1e6, digits=2))M")
                end
            else
                # If no production possible, do nothing
                site_actions[site] = 0  # Do nothing
                site_types[site] = 0  # Do nothing
                
                if policy.debug
                    println("  [DEBUG] Site $site: Doing nothing (no production possible)")
                end
            end
        end
        # If mine is already opened, do nothing (action = 0)
    end
    
    # Check for restoration opportunities first - use sampled state deposits
    for site in 1:policy.pomdp.n_deposits
        if s.have_opened[site] && s.is_depleted[site] && !s.restored_mines[site] && s.deposit[site] <= 0.1
            # Restore depleted mine - only if sampled deposits are actually zero
            site_actions[site] = site
            site_types[site] = 3  # Restore action
            
            if policy.debug
                println("  [DEBUG] Restoring site $site: Mine is depleted and not yet restored (deposits: $(s.deposit[site]))")
            end
        end
    end
    
    # FIXED: Now make decisions based on all available information
    if !isempty(site_evaluations)
        # Sort by profitability (highest first)
        sort!(site_evaluations, by=x->x[2], rev=true)
        
        # FIXED: Open the most profitable mine regardless of threshold
        best_site, best_profit, best_recovery, best_revenue = site_evaluations[1]
        
        # Open the most profitable mine
        site_actions[best_site] = best_site
        site_types[best_site] = 2  # Open mine
        
        if policy.debug
            println("  [DEBUG] Opening site $best_site: Profit = \$$(round(best_profit/1e6, digits=2))M (most profitable)")
        end
        
        # Explore other promising sites (restore exploration) - but be selective
        for i in 2:length(site_evaluations)
            site, profit, recovery, revenue = site_evaluations[i]
            # Only explore if the site is reasonably profitable (within 70% of best) AND has good potential
            if profit > best_profit * 0.7 && recovery > 0
                site_actions[site] = site
                site_types[site] = 1  # Explore
                if policy.debug
                    println("  [DEBUG] Exploring site $site: Profit = \$$(round(profit/1e6, digits=2))M (promising)")
                end
            else
                site_actions[site] = 0  # Do nothing
                site_types[site] = 0  # Do nothing
                if policy.debug
                    println("  [DEBUG] Site $site: Doing nothing (not profitable enough or no recovery potential)")
                end
            end
        end
    end
    

    return Action(site=site_actions, type=site_types)
end

function POMDPs.action(policy::DynamicProfitMaximizerPolicy, b::LiBelief)
    # Sample from belief distributions for consistency with other policies
    sampled_deposits = [rand(d) for d in b.deposits_distribution]  # Sample from distributions
    sampled_price = rand(b.price_distribution)  # Sample from distributions
    sampled_demand = rand(b.demand_distribution)  # Sample from distributions
    
    # Create a representative state
    s = State(
        deposit = sampled_deposits,
        time = b.time,
        price = sampled_price,
        company_demand = sampled_demand,
        total_mined = b.total_mined,
        have_opened = copy(b.have_opened),
        is_depleted = copy(b.is_depleted),
        restored_mines = copy(b.restored_mines)
    )
    
    return POMDPs.action(policy, s)
end

"""
DynamicEmissionMinimizerPolicy - A policy that minimizes emissions at each time step
This policy finds the single lowest emission choice and can take actions on all mines.
For each mine, it can explore, open, or do nothing based on emission intensity.
FIXED: Now implements true emission minimization by being highly selective about mine selection.
"""
struct DynamicEmissionMinimizerPolicy <: Policy
    pomdp::LiPOMDP
    debug::Bool
end

DynamicEmissionMinimizerPolicy(pomdp::LiPOMDP; debug=false) = DynamicEmissionMinimizerPolicy(pomdp, debug)

function POMDPs.action(policy::DynamicEmissionMinimizerPolicy, s::State)
    if policy.debug
        println("\n[DEBUG] DynamicEmissionMinimizerPolicy evaluating state:")
        println("  Time: $(s.time)")
        println("  Deposits: $(round.(s.deposit, digits=1))")
        println("  Have opened: $(s.have_opened)")
        println("  Company demand: $(s.company_demand)")
    end
    
    n_sites = policy.pomdp.n_deposits
    site_actions = zeros(Int, n_sites)
    site_types = zeros(Int, n_sites)
    
    # FIXED: True emission minimization strategy
    # The goal is to minimize TOTAL emissions, not just emission intensity
    
    # Calculate remaining time
    remaining_time = policy.pomdp.time_horizon - s.time
    
    # Strategy: Only open mines when absolutely necessary to meet demand
    # and prefer to open them later when demand is higher
    
    # Calculate current production capacity from already opened mines
    current_capacity = 0.0
    for site in 1:n_sites
        if s.have_opened[site] && s.deposit[site] > 0
            # Calculate this mine's capacity
            mine_capacity = policy.pomdp.output * policy.pomdp.recovery[site] * policy.pomdp.LCE_per_lithium
            current_capacity += mine_capacity
        end
    end
    
    if policy.debug
        println("  [DEBUG] Current capacity from opened mines: $(round(current_capacity, digits=1))")
        println("  [DEBUG] Current demand: $(round(s.company_demand, digits=1))")
    end
    
    # Only open new mines if current capacity cannot meet demand
    if current_capacity < s.company_demand
        remaining_demand = s.company_demand - current_capacity
        
        # Evaluate unopened mines for total emission impact
        site_evaluations = []
        
        for site in 1:n_sites
            if !s.have_opened[site] && s.deposit[site] > 0
                # Calculate expected production
                max_possible_production = min(s.deposit[site], policy.pomdp.output * remaining_time)
                expected_recovery = max_possible_production * policy.pomdp.recovery[site]
                
                # Calculate realistic production per period
                mine_capacity_per_period = policy.pomdp.output * policy.pomdp.recovery[site] * policy.pomdp.LCE_per_lithium
                realistic_production_per_period = min(mine_capacity_per_period, remaining_demand)
                
                # Calculate total emissions for this mine (production only)
                emission_factor = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_CO2_emissions : policy.pomdp.mining_CO2_emissions
                
                # Total emissions = variable production emissions only (convert kg to tonnes)
                total_emissions = (realistic_production_per_period * emission_factor * remaining_time) / 1000.0
                
                # Store evaluation
                push!(site_evaluations, (site, total_emissions, realistic_production_per_period, emission_factor))
                
                if policy.debug
                    println("  [DEBUG] Site $site: Total emissions = $(round(total_emissions, digits=1)), Production = $(round(realistic_production_per_period, digits=1))")
                end
            end
        end
        
        # Sort by total emissions (lowest first)
        sort!(site_evaluations, by=x->x[2])
        
        # FIXED: Only open the minimum number of mines needed
        # This is the key to true emission minimization
        mines_to_open = []
        temp_remaining_demand = remaining_demand
        
        for (site, total_emissions, production_per_period, emission_factor) in site_evaluations
            if temp_remaining_demand > 0
                # Only open this mine if it significantly reduces unmet demand
                if production_per_period >= temp_remaining_demand * 0.3  # Must cover at least 30% of remaining demand
                    push!(mines_to_open, site)
                    temp_remaining_demand -= production_per_period
                    
                    if policy.debug
                        println("  [DEBUG] Selected site $site: Total emissions = $(round(total_emissions, digits=1)), Production = $(round(production_per_period, digits=1))")
                    end
                    
                    # Stop if we have enough capacity
                    if temp_remaining_demand <= 0
                        break
                    end
                end
            end
        end
        
        # Check for restoration opportunities first - use sampled state deposits
        for site in 1:policy.pomdp.n_deposits
            if s.have_opened[site] && s.is_depleted[site] && !s.restored_mines[site] && s.deposit[site] <= 0.1
                # Restore depleted mine - only if sampled deposits are actually zero
                site_actions[site] = site
                site_types[site] = 3  # Restore action
                
                if policy.debug
                    println("  [DEBUG] Restoring site $site: Mine is depleted and not yet restored (deposits: $(s.deposit[site]))")
                end
            end
        end
        
        # Apply the decisions
        for site in mines_to_open
            site_actions[site] = site
            site_types[site] = 2  # Open mine
        end
        
        # Explore other sites that weren't selected for opening (restore exploration) - but be selective
        for (site, total_emissions, production_per_period, emission_factor) in site_evaluations
            if !(site in mines_to_open) && production_per_period > 0
                # Only explore if the site has significant production potential (at least 20% of demand)
                if production_per_period >= s.company_demand * 0.2
                    site_actions[site] = site
                    site_types[site] = 1  # Explore
                    if policy.debug
                        println("  [DEBUG] Exploring site $site: Total emissions = $(round(total_emissions, digits=1)), Production = $(round(production_per_period, digits=1))")
                    end
                else
                    site_actions[site] = 0  # Do nothing
                    site_types[site] = 0  # Do nothing
                    if policy.debug
                        println("  [DEBUG] Site $site: Doing nothing (insufficient production potential)")
                    end
                end
            end
        end
    end
    
    if policy.debug
        println("  [DEBUG] Final action: $(site_actions)")
        println("  [DEBUG] Final types: $(site_types)")
    end
    
    return Action(site=site_actions, type=site_types)
end

function POMDPs.action(policy::DynamicEmissionMinimizerPolicy, b::LiBelief)
    if policy.debug
        println("\n[DEBUG] DynamicEmissionMinimizerPolicy evaluating belief")
    end
    
    # Sample from belief distributions for realistic uncertainty
    sampled_deposits = [max(0.0, rand(d)) for d in b.deposits_distribution]  # Sample from distributions
    sampled_price = rand(b.price_distribution)  # Sample from distributions
    sampled_demand = rand(b.demand_distribution)  # Sample from distributions
    
    # Create a representative state
    s = State(
        deposit = sampled_deposits,
        time = b.time,
        price = sampled_price,
        company_demand = sampled_demand,
        total_mined = b.total_mined,
        have_opened = copy(b.have_opened),
        is_depleted = copy(b.is_depleted),
        restored_mines = copy(b.restored_mines)
    )
    
    return POMDPs.action(policy, s)
end

# =============================================================================
# Alpha-Aware Rollout Policy for OneStepLookahead
# =============================================================================

"""
AlphaAwareRolloutPolicy - A policy that balances profit and emissions using alpha.
This is similar to DynamicProfitMaximizerPolicy but uses the POMDP's alpha parameter
to trade off between profit maximization and emission minimization.

When alpha = 1.0: Maximizes profit (like DynamicProfitMaximizerPolicy)
When alpha = 0.0: Minimizes emissions (like DynamicEmissionMinimizerPolicy)
When alpha = 0.5: Balances both equally
"""
struct AlphaAwareRolloutPolicy <: Policy
    pomdp::LiPOMDP
    debug::Bool
end

AlphaAwareRolloutPolicy(pomdp::LiPOMDP; debug=false) = AlphaAwareRolloutPolicy(pomdp, debug)

function POMDPs.action(policy::AlphaAwareRolloutPolicy, s::State)
    if policy.debug
        println("\n[DEBUG] AlphaAwareRolloutPolicy evaluating state (α=$(policy.pomdp.alpha)):")
        println("  Time: $(s.time)")
        println("  Price: $(round(s.price, digits=2))")
        println("  Deposits: $(round.(s.deposit, digits=1))")
        println("  Have opened: $(s.have_opened)")
    end
    
    n_sites = policy.pomdp.n_deposits
    site_actions = zeros(Int, n_sites)
    site_types = zeros(Int, n_sites)
    
    remaining_time = policy.pomdp.time_horizon - s.time
    
    # Calculate current production capacity
    current_capacity = 0.0
    for site in 1:n_sites
        if s.have_opened[site] && s.deposit[site] > 0
            mine_capacity = policy.pomdp.output * policy.pomdp.recovery[site] * policy.pomdp.LCE_per_lithium
            current_capacity += mine_capacity
        end
    end
    
    remaining_demand = max(0.0, s.company_demand - current_capacity)
    
    # Evaluate all unopened sites
    site_evaluations = []
    
    for site in 1:n_sites
        if !s.have_opened[site]
            max_possible_production = min(s.deposit[site], policy.pomdp.output * remaining_time)
            expected_recovery = max_possible_production * policy.pomdp.recovery[site]
            
            if expected_recovery > 0
                years_to_depletion = s.deposit[site] / policy.pomdp.output
                
                realistic_sellable = min(
                    expected_recovery,
                    policy.pomdp.output * policy.pomdp.recovery[site] * policy.pomdp.LCE_per_lithium * years_to_depletion,
                    remaining_demand * remaining_time
                )
                
                # Calculate profit
                lifetime_years = Int(remaining_time)
                avg_lifetime_price = calculate_lifetime_average_price(policy.pomdp, s.time, lifetime_years)
                expected_revenue = realistic_sellable * avg_lifetime_price
                
                capex = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_capex : policy.pomdp.mining_capex
                opex = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_opex : policy.pomdp.mining_opex
                lifetime_opex = opex * realistic_sellable
                
                raw_profit = expected_revenue - lifetime_opex - capex
                
                # Calculate emissions (as negative cost in reward units)
                variable_emissions_per_tonne = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_CO2_emissions : policy.pomdp.mining_CO2_emissions
                lifetime_variable_emissions = realistic_sellable * variable_emissions_per_tonne
                total_emissions_cost = (lifetime_variable_emissions / 1000.0) * policy.pomdp.CO2_cost
                
                # Use ALPHA to balance profit vs emissions (matching reward function)
                # Note: emissions are negative in the reward function, so we apply emission_scale_factor
                alpha_weighted_value = policy.pomdp.alpha * raw_profit + (1 - policy.pomdp.alpha) * (-total_emissions_cost * policy.pomdp.emission_scale_factor)
                
                push!(site_evaluations, (site, alpha_weighted_value, expected_recovery, expected_revenue))
                
                if policy.debug
                    println("  [DEBUG] Site $site: profit=$(round(raw_profit/1e6, digits=2))M, emissions_cost=$(round(total_emissions_cost/1e6, digits=2))M")
                    println("    Alpha-weighted value: $(round(alpha_weighted_value/1e6, digits=2))M")
                end
            end
        end
    end
    
    # Handle restoration
    for site in 1:policy.pomdp.n_deposits
        if s.have_opened[site] && s.is_depleted[site] && !s.restored_mines[site] && s.deposit[site] <= 0.1
            site_actions[site] = site
            site_types[site] = 3  # Restore action
            
            if policy.debug
                println("  [DEBUG] Restoring site $site")
            end
        end
    end
    
    # Make decisions based on alpha-weighted values
    if !isempty(site_evaluations)
        # Sort by alpha-weighted value (highest first)
        sort!(site_evaluations, by=x->x[2], rev=true)
        
        # Open the best site
        best_site, best_value, best_recovery, best_revenue = site_evaluations[1]
        
        site_actions[best_site] = best_site
        site_types[best_site] = 2  # Open mine
        
        if policy.debug
            println("  [DEBUG] Opening site $best_site: Alpha-weighted value = $(round(best_value/1e6, digits=2))M")
        end
        
        # Explore other promising sites (within 70% of best)
        for i in 2:length(site_evaluations)
            site, value, recovery, revenue = site_evaluations[i]
            if value > best_value * 0.7 && recovery > 0
                site_actions[site] = site
                site_types[site] = 1  # Explore
                if policy.debug
                    println("  [DEBUG] Exploring site $site: Alpha-weighted value = $(round(value/1e6, digits=2))M")
                end
            end
        end
    end
    
    return Action(site=site_actions, type=site_types)
end

function POMDPs.action(policy::AlphaAwareRolloutPolicy, b::LiBelief)
    sampled_deposits = [rand(d) for d in b.deposits_distribution]
    sampled_price = rand(b.price_distribution)
    sampled_demand = rand(b.demand_distribution)

    s = State(
        deposit = sampled_deposits,
        time = b.time,
        price = sampled_price,
        company_demand = sampled_demand,
        total_mined = b.total_mined,
        have_opened = copy(b.have_opened),
        is_depleted = copy(b.is_depleted),
        restored_mines = copy(b.restored_mines),
    )

    return POMDPs.action(policy, s)
end

# =============================================================================
# One-Step Lookahead Policy with Rollouts
# =============================================================================

"""
One-step lookahead heuristic policy.

At each time step:
1. Consider all possible actions
2. For each action, simulate forward using AlphaAwareRolloutPolicy
3. Run multiple simulations (default 30) to get average reward
4. Choose the action with highest expected reward

This is essentially a rollout-based policy using AlphaAwareRolloutPolicy as the rollout policy,
which respects the POMDP's alpha parameter for balancing profit vs emissions.
"""
struct OneStepLookaheadPolicy <: Policy
    pomdp::LiPOMDP
    rollout_policy::AlphaAwareRolloutPolicy
    n_rollouts::Int
    debug::Bool
end

function OneStepLookaheadPolicy(pomdp::LiPOMDP; n_rollouts=30, debug=false)
    rollout_policy = AlphaAwareRolloutPolicy(pomdp, debug=false)
    return OneStepLookaheadPolicy(pomdp, rollout_policy, n_rollouts, debug)
end

"""
Generate all feasible actions for a given state.
Returns a vector of possible actions to consider.
This generates ALL possible action combinations (4^n_sites = 256 for 4 sites),
filtered by state constraints.
"""
function generate_feasible_actions(policy::OneStepLookaheadPolicy, s::State)
    n_sites = policy.pomdp.n_deposits
    actions = Action[]
    
    # Generate all possible combinations of actions across all sites
    # Action types: 0=nothing, 1=explore, 2=open, 3=restore
    n_combinations = 4^n_sites  # 256 for 4 sites
    
    for combo_idx in 0:(n_combinations-1)
        types = zeros(Int, n_sites)
        
        # Convert combination index to base-4 representation
        temp_idx = combo_idx
        for site in 1:n_sites
            action_type = temp_idx % 4
            types[site] = action_type
            temp_idx = div(temp_idx, 4)
        end
        
        # Check if this action combination is valid
        valid = true
        for site in 1:n_sites
            if types[site] == 1  # EXPLORE
                # Can only explore unopened sites
                if s.have_opened[site]
                    valid = false
                    break
                end
            elseif types[site] == 2  # OPEN
                # Can only open unopened sites
                if s.have_opened[site]
                    valid = false
                    break
                end
            elseif types[site] == 3  # RESTORE
                # Can only restore depleted mines that haven't been restored
                if !s.have_opened[site] || !s.is_depleted[site] || s.restored_mines[site]
                    valid = false
                    break
                end
            end
        end
        
        # Add valid action to list
        if valid
            push!(actions, Action(site=collect(1:n_sites), type=types))
        end
    end
    
    return actions
end

"""
Simulate one rollout from the given state with the given initial action,
then following the rollout policy for all subsequent steps.
Returns the total discounted reward.
"""
function simulate_rollout(policy::OneStepLookaheadPolicy, s::State, a::Action, rng::AbstractRNG)
    total_reward = 0.0
    discount_factor = 1.0  # First reward is not discounted (γ^0 = 1.0)
    
    # Execute the first action
    sp, o, r = @gen(:sp, :o, :r)(policy.pomdp, s, a, rng)
    total_reward += discount_factor * r
    
    # Check if terminal
    if isterminal(policy.pomdp, sp)
        return total_reward
    end
    
    # Continue with rollout policy for remaining time steps
    current_state = sp
    remaining_steps = Int(policy.pomdp.time_horizon - current_state.time + 1)
    
    for t in 1:remaining_steps
        discount_factor *= policy.pomdp.γ
        
        # Get action from rollout policy
        rollout_action = action(policy.rollout_policy, current_state)
        
        # Execute action
        sp, o, r = @gen(:sp, :o, :r)(policy.pomdp, current_state, rollout_action, rng)
        total_reward += discount_factor * r
        
        # Check if terminal
        if isterminal(policy.pomdp, sp)
            break
        end
        
        current_state = sp
    end
    
    return total_reward
end

"""
Action selection: evaluate all feasible actions and choose the best one.
"""
function POMDPs.action(policy::OneStepLookaheadPolicy, s::State)
    if policy.debug
        println("\n[DEBUG] OneStepLookaheadPolicy evaluating state:")
        println("  Time: $(s.time)")
        println("  Price: $(round(s.price, digits=2))")
    end
    
    # Generate all feasible actions
    feasible_actions = generate_feasible_actions(policy, s)
    
    if policy.debug
        println("  Evaluating $(length(feasible_actions)) feasible actions")
    end
    
    # Evaluate each action with multiple rollouts
    best_action = feasible_actions[1]
    best_avg_reward = -Inf
    
    for (idx, a) in enumerate(feasible_actions)
        # Run multiple rollout simulations for this action
        rewards = Float64[]
        
        for sim in 1:policy.n_rollouts
            rng = Random.MersenneTwister(hash((s.time, idx, sim)))
            reward = simulate_rollout(policy, s, a, rng)
            push!(rewards, reward)
        end
        
        avg_reward = mean(rewards)
        
        if policy.debug && any(a.type .!= 0)  # Only print non-NO-OP actions
            action_desc = join(["Site $i: $(a.type[i])" for i in 1:length(a.type) if a.type[i] != 0], ", ")
            println("  Action $(idx): $action_desc")
            println("    Avg reward: $(round(avg_reward, digits=2)) (std: $(round(std(rewards), digits=2)))")
        end
        
        # Update best action
        if avg_reward > best_avg_reward
            best_avg_reward = avg_reward
            best_action = a
        end
    end
    
    if policy.debug
        best_action_desc = join(["Site $i: $(best_action.type[i])" for i in 1:length(best_action.type) if best_action.type[i] != 0], ", ")
        println("  ✓ Selected action: $best_action_desc (avg reward: $(round(best_avg_reward, digits=2)))")
    end
    
    return best_action
end

"""
Rollout when the deposit is hidden. `s` is used only by the simulator.
Later actions come from `action(rollout_policy, b)` on the updated belief,
not from the true deposit.
"""
function simulate_rollout_belief(policy::OneStepLookaheadPolicy, b0::LiBelief, s::State, a::Action, updater, rng::AbstractRNG)
    total = 0.0
    disc = 1.0
    sp, o, r = @gen(:sp, :o, :r)(policy.pomdp, s, a, rng)
    total += r
    isterminal(policy.pomdp, sp) && return total
    b = update(updater, b0, a, o)
    s = sp
    remaining = Int(policy.pomdp.time_horizon - s.time + 1)
    for _ in 1:remaining
        disc *= policy.pomdp.γ
        a = action(policy.rollout_policy, b)
        sp, o, r = @gen(:sp, :o, :r)(policy.pomdp, s, a, rng)
        total += disc * r
        isterminal(policy.pomdp, sp) && break
        b = update(updater, b, a, o)
        s = sp
    end
    return total
end

"""
Action from a belief. Feasible actions use opened/depleted flags only.
Each rollout draws a hidden state for the simulator; the rollout policy
never sees that state.
"""
function POMDPs.action(policy::OneStepLookaheadPolicy, b::LiBelief)
    n = policy.pomdp.n_deposits
    s_flags = State(
        deposit = zeros(n),
        time = b.time,
        price = max(mean(b.price_distribution), 100.0),
        company_demand = max(mean(b.demand_distribution), 0.0),
        total_mined = b.total_mined,
        have_opened = copy(b.have_opened),
        is_depleted = copy(b.is_depleted),
        restored_mines = copy(b.restored_mines),
    )
    feasible = generate_feasible_actions(policy, s_flags)
    updater = LiBeliefUpdater(P=policy.pomdp)
    best_action = feasible[1]
    best_avg = -Inf
    for (idx, a) in enumerate(feasible)
        total = 0.0
        for sim in 1:policy.n_rollouts
            rng = Random.MersenneTwister(hash((b.time, idx, sim, :onestep)))
            s = rand(rng, b)
            total += simulate_rollout_belief(policy, b, s, a, updater, rng)
        end
        avg = total / policy.n_rollouts
        if avg > best_avg
            best_avg = avg
            best_action = a
        end
    end
    return best_action
end
