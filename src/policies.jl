"""
Average noise-free price over the next `lifetime_years` years under the price model, floored at
\$1,000/t as in `update_price`. GBM (model 4) falls back to the constant p₀.
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
            historical_year = P.start_year + Int(t) - 1
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
        
        # Deterministic price (noise_std is not applied) for consistent policy decisions
        year_price = base_price
        year_price = max(year_price, 1000.0)  # same floor as update_price
        total_price += year_price
    end
    
    return total_price / lifetime_years  # Average base price over lifetime
end


"""
ProfitMaximizerPolicy - at each time in `times`, open the `top_k` unopened sites with the highest
undiscounted lifetime profit (average forecast price, opex and capex). Depleted mines are restored.
"""
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
    
    # Restore depleted mines at any time, not only at the opening times
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

    # Open new mines only at the configured times
    if s.time in policy.times
        # Rank unopened sites by lifetime profit
        profits = zeros(policy.pomdp.n_deposits)
        remaining_time = policy.pomdp.time_horizon - s.time
        for site in 1:policy.pomdp.n_deposits
            if !s.have_opened[site] && s.deposit[site] > 0
                # Lifetime production, limited by deposit, capacity and demand
                max_possible_in_situ = min(s.deposit[site], policy.pomdp.output * remaining_time)
                max_possible_sellable = max_possible_in_situ * policy.pomdp.recovery[site]
                
                # Assume current demand holds for the rest of the horizon
                avg_demand_per_period = s.company_demand
                realistic_lifetime_demand = avg_demand_per_period * remaining_time
                realistic_sellable = min(max_possible_sellable, realistic_lifetime_demand)
                
                # Revenue at the average forecast price over the rest of the horizon
                lifetime_years = Int(remaining_time)
                avg_lifetime_price = calculate_lifetime_average_price(policy.pomdp, s.time, Int(lifetime_years))
                
                # Undiscounted
                lifetime_revenue = realistic_sellable * avg_lifetime_price
                
                # Lifetime costs
                capex = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_capex : policy.pomdp.mining_capex
                opex = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_opex : policy.pomdp.mining_opex
                lifetime_opex = opex * realistic_sellable  # OPEX on realistic sellable production
                
                total_profit = lifetime_revenue - lifetime_opex - capex
                profits[site] = total_profit
                
                if policy.debug
                    println("  [DEBUG] Site $site: current_price=$(round(s.price, digits=2)), avg_lifetime_price=$(round(avg_lifetime_price, digits=2)), lifetime_years=$lifetime_years, realistic_sellable=$(round(realistic_sellable, digits=1)), revenue=$(round(lifetime_revenue/1e6, digits=2))M, opex=$(round(lifetime_opex/1e6, digits=2))M, capex=$(round(capex/1e6, digits=2))M, profit=$(round(total_profit/1e6, digits=2))M")
                end
            end
        end
    
        ranked_sites = sortperm(profits, rev=true)
        
        for i in 1:policy.top_k
            opened_sites[ranked_sites[i]] = ranked_sites[i]
            opened_types[ranked_sites[i]] = 2
        end

        return Action(site=opened_sites, type=opened_types)
    else
        if policy.debug
            println("  [DEBUG] ProfitMaximizerPolicy not activated at time $(s.time), but checking for restoration")
        end
        # Outside the opening times, return only the restoration actions
        return Action(site=opened_sites, type=opened_types)
    end
    
end

function POMDPs.action(policy::ProfitMaximizerPolicy, b::LiBelief)
    if policy.debug
        println("\n[DEBUG] ProfitMaximizerPolicy evaluating belief")
    end
    
    # Act on a single state sampled from the belief
    sampled_deposits = [rand(d) for d in b.deposits_distribution]
    sampled_price = rand(b.price_distribution)
    sampled_demand = rand(b.demand_distribution)
    
    # Status flags are known, so they are copied from the belief
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
EmissionMinimizerPolicy - at each time in `times`, open the `top_k` unopened sites with the lowest
lifetime emissions. Depleted mines are restored.
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

    # Restore depleted mines at any time, not only at the opening times
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

    # Open new mines only at the configured times
    if s.time in policy.times
        # Rank unopened sites by lifetime emissions
        emissions = zeros(policy.pomdp.n_deposits)
        remaining_time = policy.pomdp.time_horizon - s.time
        for site in 1:policy.pomdp.n_deposits
            if !s.have_opened[site] && s.deposit[site] > 0
                # Lifetime emissions if the site is opened now
                remaining_time = max(policy.pomdp.time_horizon - s.time, 0)
                max_in_situ = min(s.deposit[site], policy.pomdp.output * remaining_time)
                max_sellable = max_in_situ * policy.pomdp.recovery[site]
                
                # Emission factor × lifetime in-situ extraction (no price forecast needed)
                var_emissions_per_ton = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_CO2_emissions : policy.pomdp.mining_CO2_emissions
                lifetime_var_emissions = max_in_situ * var_emissions_per_ton
                total_lifetime_emissions = lifetime_var_emissions
                
                emissions[site] = total_lifetime_emissions
                
                if policy.debug
                    println("  [DEBUG] Site $site: lifetime emissions=$(round(total_lifetime_emissions, digits=2)) tCO2 (production-only)")
                end
            end
        end


        ranked_sites = sortperm(emissions)
        
        for i in 1:policy.top_k
            opened_sites[ranked_sites[i]] = ranked_sites[i]
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
        # Outside the opening times, return only the restoration actions
        return Action(site=opened_sites, type=opened_types)
    end
end

function POMDPs.action(policy::EmissionMinimizerPolicy, b::LiBelief)
    if policy.debug
        println("\n[DEBUG] EmissionMinimizerPolicy evaluating belief")
    end
    
    # Act on a single state sampled from the belief
    sampled_deposits = [max(0.0, rand(d)) for d in b.deposits_distribution]
    sampled_price = rand(b.price_distribution)
    sampled_demand = rand(b.demand_distribution)
    
    # Status flags are known, so they are copied from the belief
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
ExploreOnlyPolicy - baseline that explores every unopened site at every step and never opens a
mine (depleted mines are restored). Measures the cost of exploration with no production.
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
    
    # Restore depleted mines; explore every unopened site
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
    
    # Do nothing if every site is opened
    if policy.debug
        println("  [DEBUG] All sites opened, doing nothing")
    end
    return Action(site=[1, 2, 3, 4], type=[0, 0, 0, 0])
end

function POMDPs.action(policy::ExploreOnlyPolicy, b::LiBelief)
    if policy.debug
        println("\n[DEBUG] ExploreOnlyPolicy evaluating belief")
    end
    
    # Act on a single state sampled from the belief
    sampled_deposits = [rand(d) for d in b.deposits_distribution]
    sampled_price = rand(b.price_distribution)
    sampled_demand = rand(b.demand_distribution)
    
    # Status flags are known, so they are copied from the belief
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
RandomHeuristicPolicy - baseline that, at each step, independently picks do nothing / explore /
open uniformly for every unopened site, and restores depleted mines.
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
            
            if action_choice > 0
                site_actions[i] = i
                site_types[i] = action_choice  # 1=explore, 2=open
            end
        end
        # Open, non-depleted mines: do nothing
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
DynamicProfitMaximizerPolicy - at every step, rank unopened sites by undiscounted lifetime profit
(revenue at the average forecast price, minus opex, capex and emissions at CO2_cost), limited by
the demand not already covered by open mines. Opens the best site (with no profitability
threshold), explores the others within 70% of the best, and restores depleted mines.
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
    
    remaining_time = policy.pomdp.time_horizon - s.time
    
    # Production capacity (t LCE / year) of mines already open
    current_capacity = 0.0
    for site in 1:n_sites
        if s.have_opened[site] && s.deposit[site] > 0
            mine_capacity = policy.pomdp.output * policy.pomdp.recovery[site] * policy.pomdp.LCE_per_lithium
            current_capacity += mine_capacity
        end
    end
    
    if policy.debug
        println("  [DEBUG] Current capacity from opened mines: $(round(current_capacity, digits=1))")
        println("  [DEBUG] Current demand: $(round(s.company_demand, digits=1))")
    end
    
    # Demand not covered by open mines
    remaining_demand = max(0.0, s.company_demand - current_capacity)
    
    # Evaluate every unopened site
    site_evaluations = []
    
    for site in 1:n_sites
        if !s.have_opened[site]
            max_possible_production = min(s.deposit[site], policy.pomdp.output * remaining_time)
            expected_recovery = max_possible_production * policy.pomdp.recovery[site]
            
            if expected_recovery > 0
                # Years the deposit lasts at nameplate capacity
                years_to_depletion = s.deposit[site] / policy.pomdp.output
                
                # Lifetime sellable production, limited by deposit size and uncovered demand
                realistic_sellable = min(
                    expected_recovery,  # Max possible recovery
                    policy.pomdp.output * policy.pomdp.recovery[site] * policy.pomdp.LCE_per_lithium * years_to_depletion,  # Deposit-limited
                    remaining_demand * remaining_time  # Demand-limited over lifetime
                )
                
                # Revenue at the average forecast price over the rest of the horizon
                lifetime_years = Int(remaining_time)
                avg_lifetime_price = calculate_lifetime_average_price(policy.pomdp, s.time, lifetime_years)
                expected_revenue = realistic_sellable * avg_lifetime_price
                
                capex = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_capex : policy.pomdp.mining_capex
                opex = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_opex : policy.pomdp.mining_opex
                lifetime_opex = opex * realistic_sellable  # OPEX on realistic sellable production
                
                # Emissions (t CO2 per t LCE produced)
                variable_emissions_per_tonne = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_CO2_emissions : policy.pomdp.mining_CO2_emissions
                lifetime_variable_emissions = realistic_sellable * variable_emissions_per_tonne
                
                # Emissions cost (production only) at CO2_cost
                total_emissions_cost = lifetime_variable_emissions * policy.pomdp.CO2_cost
                
                # Profit net of emissions cost
                total_profit = expected_revenue - lifetime_opex - capex - total_emissions_cost
                
                                push!(site_evaluations, (site, total_profit, expected_recovery, expected_revenue))
                
                if policy.debug
                    println("  [DEBUG] Site $site: current_price=$(round(s.price, digits=2)), avg_lifetime_price=$(round(avg_lifetime_price, digits=2)), realistic_production=$(round(realistic_sellable, digits=1)), years_to_depletion=$(round(years_to_depletion, digits=1))")
                    println("    Remaining demand: $(round(remaining_demand, digits=1)), Demand-limited production: $(round(remaining_demand * remaining_time, digits=1))")
                    println("    Revenue: \$$(round(expected_revenue/1e6, digits=2))M, OPEX: \$$(round(lifetime_opex/1e6, digits=2))M, CAPEX: \$$(round(capex/1e6, digits=2))M")
                    println("    Emissions cost: \$$(round(total_emissions_cost/1e6, digits=2))M, Total Profit: \$$(round(total_profit/1e6, digits=2))M")
                end
            else
                # No production possible: do nothing
                site_actions[site] = 0
                site_types[site] = 0
                
                if policy.debug
                    println("  [DEBUG] Site $site: Doing nothing (no production possible)")
                end
            end
        end
        # Opened mines: do nothing
    end
    
    # Restore depleted mines whose (sampled) deposit is exhausted
    for site in 1:policy.pomdp.n_deposits
        if s.have_opened[site] && s.is_depleted[site] && !s.restored_mines[site] && s.deposit[site] <= 0.1
            site_actions[site] = site
            site_types[site] = 3  # Restore action
            
            if policy.debug
                println("  [DEBUG] Restoring site $site: Mine is depleted and not yet restored (deposits: $(s.deposit[site]))")
            end
        end
    end
    
    if !isempty(site_evaluations)
        # Sort by profitability (highest first)
        sort!(site_evaluations, by=x->x[2], rev=true)
        
        # Open the most profitable mine, with no profitability threshold
        best_site, best_profit, best_recovery, best_revenue = site_evaluations[1]
        site_actions[best_site] = best_site
        site_types[best_site] = 2  # Open mine
        
        if policy.debug
            println("  [DEBUG] Opening site $best_site: Profit = \$$(round(best_profit/1e6, digits=2))M (most profitable)")
        end
        
        # Explore the other sites whose profit is within 70% of the best
        for i in 2:length(site_evaluations)
            site, profit, recovery, revenue = site_evaluations[i]
            if profit > best_profit * 0.7 && recovery > 0
                site_actions[site] = site
                site_types[site] = 1  # Explore
                if policy.debug
                    println("  [DEBUG] Exploring site $site: Profit = \$$(round(profit/1e6, digits=2))M (promising)")
                end
            else
                site_actions[site] = 0
                site_types[site] = 0
                if policy.debug
                    println("  [DEBUG] Site $site: Doing nothing (not profitable enough or no recovery potential)")
                end
            end
        end
    end
    

    return Action(site=site_actions, type=site_types)
end

function POMDPs.action(policy::DynamicProfitMaximizerPolicy, b::LiBelief)
    # Act on a single state sampled from the belief
    sampled_deposits = [rand(d) for d in b.deposits_distribution]
    sampled_price = rand(b.price_distribution)
    sampled_demand = rand(b.demand_distribution)
    
    # Status flags are known, so they are copied from the belief
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
DynamicEmissionMinimizerPolicy - opens mines only when open capacity falls short of demand. Unopened
sites are taken in order of lowest emissions over the rest of the horizon; each is opened only
if it covers at least 30% of the remaining shortfall, until the shortfall is met. In the same
steps, other sites that could supply at least 20% of demand are explored and depleted mines are
restored; with no shortfall it does nothing.
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
    
    remaining_time = policy.pomdp.time_horizon - s.time
    
    # Production capacity (t LCE / year) of mines already open
    current_capacity = 0.0
    for site in 1:n_sites
        if s.have_opened[site] && s.deposit[site] > 0
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
        
        # Total emissions of each unopened site over the rest of the horizon
        site_evaluations = []
        
        for site in 1:n_sites
            if !s.have_opened[site] && s.deposit[site] > 0
                max_possible_production = min(s.deposit[site], policy.pomdp.output * remaining_time)
                expected_recovery = max_possible_production * policy.pomdp.recovery[site]
                
                # Production per period, limited by the shortfall
                mine_capacity_per_period = policy.pomdp.output * policy.pomdp.recovery[site] * policy.pomdp.LCE_per_lithium
                realistic_production_per_period = min(mine_capacity_per_period, remaining_demand)
                
                # Production emissions only (t CO2 per t LCE)
                emission_factor = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_CO2_emissions : policy.pomdp.mining_CO2_emissions
                total_emissions = realistic_production_per_period * emission_factor * remaining_time  # t CO2
                
                push!(site_evaluations, (site, total_emissions, realistic_production_per_period, emission_factor))
                
                if policy.debug
                    println("  [DEBUG] Site $site: Total emissions = $(round(total_emissions, digits=1)), Production = $(round(realistic_production_per_period, digits=1))")
                end
            end
        end
        
        # Sort by total emissions (lowest first)
        sort!(site_evaluations, by=x->x[2])
        
        # Open the fewest mines that cover the shortfall
        mines_to_open = []
        temp_remaining_demand = remaining_demand
        
        for (site, total_emissions, production_per_period, emission_factor) in site_evaluations
            if temp_remaining_demand > 0
                if production_per_period >= temp_remaining_demand * 0.3  # must cover at least 30% of the remaining shortfall
                    push!(mines_to_open, site)
                    temp_remaining_demand -= production_per_period
                    
                    if policy.debug
                        println("  [DEBUG] Selected site $site: Total emissions = $(round(total_emissions, digits=1)), Production = $(round(production_per_period, digits=1))")
                    end
                    
                    # Stop once the shortfall is covered
                    if temp_remaining_demand <= 0
                        break
                    end
                end
            end
        end
        
        # Restore depleted mines whose (sampled) deposit is exhausted
        for site in 1:policy.pomdp.n_deposits
            if s.have_opened[site] && s.is_depleted[site] && !s.restored_mines[site] && s.deposit[site] <= 0.1
                site_actions[site] = site
                site_types[site] = 3  # Restore action
                
                if policy.debug
                    println("  [DEBUG] Restoring site $site: Mine is depleted and not yet restored (deposits: $(s.deposit[site]))")
                end
            end
        end
        
        # Open the selected mines
        for site in mines_to_open
            site_actions[site] = site
            site_types[site] = 2  # Open mine
        end
        
        # Explore unselected sites that could supply at least 20% of demand
        for (site, total_emissions, production_per_period, emission_factor) in site_evaluations
            if !(site in mines_to_open) && production_per_period > 0
                if production_per_period >= s.company_demand * 0.2
                    site_actions[site] = site
                    site_types[site] = 1  # Explore
                    if policy.debug
                        println("  [DEBUG] Exploring site $site: Total emissions = $(round(total_emissions, digits=1)), Production = $(round(production_per_period, digits=1))")
                    end
                else
                    site_actions[site] = 0
                    site_types[site] = 0
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
    
    # Act on a single state sampled from the belief
    sampled_deposits = [max(0.0, rand(d)) for d in b.deposits_distribution]
    sampled_price = rand(b.price_distribution)
    sampled_demand = rand(b.demand_distribution)
    
    # Status flags are known, so they are copied from the belief
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
AlphaAwareRolloutPolicy - DynamicProfitMaximizerPolicy with sites ranked by the reward's
alpha-weighted value, alpha × profit - (1 - alpha) × emissions cost × emission_scale_factor,
instead of profit net of emissions. alpha = 1 ranks by profit alone, alpha = 0 by emissions alone.
Opens the best site, explores the others within 70% of the best, and restores depleted mines.
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
    
    # Production capacity (t LCE / year) of mines already open
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
                
                # Profit at the average forecast price over the rest of the horizon
                lifetime_years = Int(remaining_time)
                avg_lifetime_price = calculate_lifetime_average_price(policy.pomdp, s.time, lifetime_years)
                expected_revenue = realistic_sellable * avg_lifetime_price
                
                capex = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_capex : policy.pomdp.mining_capex
                opex = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_opex : policy.pomdp.mining_opex
                lifetime_opex = opex * realistic_sellable
                
                raw_profit = expected_revenue - lifetime_opex - capex
                
                # Emissions cost at CO2_cost ($)
                variable_emissions_per_tonne = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_CO2_emissions : policy.pomdp.mining_CO2_emissions
                lifetime_variable_emissions = realistic_sellable * variable_emissions_per_tonne
                total_emissions_cost = lifetime_variable_emissions * policy.pomdp.CO2_cost
                
                # Weight profit and emissions with alpha, as in `reward`
                alpha_weighted_value = policy.pomdp.alpha * raw_profit + (1 - policy.pomdp.alpha) * (-total_emissions_cost * policy.pomdp.emission_scale_factor)
                
                push!(site_evaluations, (site, alpha_weighted_value, expected_recovery, expected_revenue))
                
                if policy.debug
                    println("  [DEBUG] Site $site: profit=$(round(raw_profit/1e6, digits=2))M, emissions_cost=$(round(total_emissions_cost/1e6, digits=2))M")
                    println("    Alpha-weighted value: $(round(alpha_weighted_value/1e6, digits=2))M")
                end
            end
        end
    end
    
    # Restore depleted mines whose (sampled) deposit is exhausted
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

POMDPs.action(policy::AlphaAwareRolloutPolicy, b::LiBelief) = action_from_belief(policy, b, Random.default_rng())

# Samples a state from `b` with `rng` and acts on it.
function action_from_belief(policy::AlphaAwareRolloutPolicy, b::LiBelief, rng::AbstractRNG)
    sampled_deposits = [rand(rng, d) for d in b.deposits_distribution]
    sampled_price = rand(rng, b.price_distribution)
    sampled_demand = rand(rng, b.demand_distribution)

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

# Base policies without a sampling step act on the belief directly
action_from_belief(policy::Policy, b::LiBelief, rng::AbstractRNG) = POMDPs.action(policy, b)

# =============================================================================
# Timing-Aware Policy (optimal stopping on the expected path)
# =============================================================================

"""
TimingAwarePolicy - open the best affordable site at the best time on the expected path.

For each unopened, affordable site it computes the α-weighted discounted value of opening it in
every remaining year, on the planning model's noise-free price path and expected demand, with
the site's expected deposit and recovery (construction year, per-ore opex, emissions and
depletion as in the model). It opens the best site now only if now is that site's best opening
year and the value is positive: optimal stopping on the mean path. It never explores, so the
value of information is left to the planner that uses it as a rollout or base policy.
"""
struct TimingAwarePolicy <: Policy
    pomdp::LiPOMDP
end

# Expected price and demand at each future time τ (index τ), as `transition` would generate them
function expected_paths(P::LiPOMDP, t::Int, price_now::Float64)
    H = P.time_horizon
    price  = zeros(H + 1)
    demand = zeros(H + 1)
    for τ in (t + 1):H
        base = price_base(P, Float64(τ - 1))
        price[τ]  = base === nothing ? price_now * exp(P.gbm_drift * (τ - t)) : max(base, PRICE_FLOOR)
        demand[τ] = get_historical_production(P, P.start_year + τ - 1) * P.LCE_per_lithium * P.market_share_target
    end
    return price, demand
end

# α-weighted value, discounted to t_now, of opening `site` at t_open (production starts t_open + 1)
function opening_value(P::LiPOMDP, site::Int, t_open::Int, t_now::Int, deposit::Float64, r::Float64,
                       price::Vector{Float64}, demand_left::Vector{Float64})
    opex = P.deposit_types[site] ? P.dle_opex : P.mining_opex
    ef   = P.deposit_types[site] ? P.dle_CO2_emissions : P.mining_CO2_emissions
    threshold = 0.05 * (P.deposit_types[site] ? P.init_deposit_size_dle : P.init_deposit_size_mining)
    v = -P.alpha * site_capex(P, site) * P.γ^(t_open - t_now)
    d = deposit
    for τ in (t_open + 1):P.time_horizon
        d <= max(threshold, 0.1) && break
        sold = min(min(P.output, d) * r * P.LCE_per_lithium, max(demand_left[τ], 0.0))
        d -= sold / (r * P.LCE_per_lithium)
        profit = sold * (price[τ] - opex * P.recovery[site] / r)
        emis   = sold * ef * P.CO2_cost
        v += P.γ^(τ - t_now) * (P.alpha * profit - (1 - P.alpha) * emis * P.emission_scale_factor)
    end
    return v
end

function timing_action(policy::TimingAwarePolicy, t::Int, deposits, recovery, price_now::Float64,
                       have_opened, is_depleted, restored_mines)
    P = policy.pomdp
    n = P.n_deposits
    price, demand = expected_paths(P, t, price_now)
    # demand left for a new mine after the capacity of mines already open
    used = sum((P.output * recovery[i] * P.LCE_per_lithium for i in 1:n
                if have_opened[i] && !is_depleted[i] && !restored_mines[i]); init=0.0)
    demand_left = demand .- used
    best_site, best_v = 0, 0.0
    for i in 1:n
        have_opened[i] && continue
        committed_capex(P, have_opened) + site_capex(P, i) > P.capital_budget && continue
        v_now = opening_value(P, i, t, t, deposits[i], recovery[i], price, demand_left)
        v_later = maximum((opening_value(P, i, t2, t, deposits[i], recovery[i], price, demand_left)
                           for t2 in (t + 1):P.time_horizon); init=-Inf)
        if v_now > 0 && v_now >= v_later && v_now > best_v
            best_site, best_v = i, v_now
        end
    end
    types = zeros(Int, n)
    best_site > 0 && (types[best_site] = 2)
    return Action(site=collect(1:n), type=types)
end

function POMDPs.action(policy::TimingAwarePolicy, s::State)
    P = policy.pomdp
    return timing_action(policy, Int(s.time), s.deposit, [site_recovery(P, s, i) for i in 1:P.n_deposits],
                         s.price, s.have_opened, s.is_depleted, s.restored_mines)
end

function POMDPs.action(policy::TimingAwarePolicy, b::LiBelief)
    P = policy.pomdp
    recovery = isempty(b.recovery_distribution) ? copy(P.recovery) : [mean(d) for d in b.recovery_distribution]
    return timing_action(policy, Int(b.time), [mean(d) for d in b.deposits_distribution], recovery,
                         mean(b.price_distribution), b.have_opened, b.is_depleted, b.restored_mines)
end

# =============================================================================
# One-Step Lookahead Policy with Rollouts
# =============================================================================

"""
One-step lookahead heuristic policy.

At each time step:
1. Consider every feasible action within the capital budget
2. For each action, simulate forward with `rollout_policy` (default AlphaAwareRolloutPolicy)
3. Average the discounted return over `n_rollouts` simulations (default 30), with common random numbers
4. Choose the action with the highest average return
"""
struct OneStepLookaheadPolicy <: Policy
    pomdp::LiPOMDP
    rollout_policy::Policy
    n_rollouts::Int
    debug::Bool
    rng::AbstractRNG
end

function OneStepLookaheadPolicy(pomdp::LiPOMDP; n_rollouts=30, debug=false, rng::AbstractRNG=Random.default_rng(),
                                rollout_policy::Policy=AlphaAwareRolloutPolicy(pomdp, debug=false))
    return OneStepLookaheadPolicy(pomdp, rollout_policy, n_rollouts, debug, rng)
end

"""
All action combinations (4^n_sites = 256 for 4 sites) that are valid in state `s`.
The capital budget is not checked here.
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
        
        # Valid if every site's action is allowed in s
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
    
    # Generate all feasible actions (within the capital budget)
    feasible_actions = filter(a -> effective_action(policy.pomdp, s.have_opened, a) == a,
                              generate_feasible_actions(policy, s))
    
    if policy.debug
        println("  Evaluating $(length(feasible_actions)) feasible actions")
    end
    
    # Evaluate each action with multiple rollouts
    best_action = feasible_actions[1]
    best_avg_reward = -Inf
    # Common random numbers: rollout `sim` uses the same seed for every action
    base_seed = rand(policy.rng, UInt64)
    
    for (idx, a) in enumerate(feasible_actions)
        # Run multiple rollout simulations for this action
        rewards = Float64[]
        
        for sim in 1:policy.n_rollouts
            rng = Random.MersenneTwister(hash((base_seed, sim)))
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
        a = action_from_belief(policy.rollout_policy, b, rng)
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
    feasible = filter(a -> effective_action(policy.pomdp, s_flags.have_opened, a) == a,
                      generate_feasible_actions(policy, s_flags))
    updater = LiBeliefUpdater(P=policy.pomdp)
    best_action = feasible[1]
    best_avg = -Inf
    # Common random numbers: rollout `sim` draws the same hidden state and
    # noise for every action, so actions are compared on the same scenarios.
    base_seed = rand(policy.rng, UInt64)
    for a in feasible
        total = 0.0
        for sim in 1:policy.n_rollouts
            rng = Random.MersenneTwister(hash((base_seed, sim)))
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
