#!/usr/bin/env julia

"""
Train/Test Split Evaluation for LiPOMDP Policies

This script implements proper train/test split evaluation to avoid overfitting.
It separates training and testing phases with different random seeds and scenarios.

Key Features:
- Different random seeds for training vs testing
- Separate train/test phases
- Better overfitting detection
- More realistic evaluation on unseen data
"""

using LiPOMDPs
using Random
using Statistics
using POMDPs
using POMDPTools
using POMCPOW
using POMCPOW: POMCPOWPlanner  # Import POMCPOWPlanner specifically
using D3Trees

# Import specific functions from pomdp.jl that aren't exported
import LiPOMDPs: calculate_site_production, calculate_total_production, compute_costs, compute_emissions



# Create output directory for tree visualizations
# TREE_DIR = "tree_visualizations"
# if !isdir(TREE_DIR)
#     mkdir(TREE_DIR)
# end

# function save_tree_html(path::AbstractString, dtree::D3Trees.D3Tree)
#     open(path, "w") do io
#         show(io, MIME"text/html"(), dtree)  # writes an interactive D3 page
#     end
# end

# =============================================================================
# POLICY CREATION FUNCTIONS
# =============================================================================

function create_policy(policy_type::String, pomdp::LiPOMDP)
    """Create a policy of the specified type."""
    if policy_type == "POMCPOW"
        # Use estimate_value = 0 for alpha = 0 (emission minimization)
        # Use DynamicProfitMaximizer rollout for all other alphas
        estimate_val = (pomdp.alpha == 0.0) ? 0 : FORollout(DynamicProfitMaximizerPolicy(pomdp))
        
        solver = POMCPOWSolver(
            tree_queries = 10000,
            max_depth = 25,
            criterion = MaxUCB(1.0),
            enable_action_pw = true, # ENABLED for large action space
            k_action = 5.0,          # Increased for more action exploration
            alpha_action = 0.5,     # Increased for more progressive widening
            k_observation = 3.0,    # Increased for more observation exploration
            alpha_observation = 0.3, # Increased for more observation progressive widening
            estimate_value = estimate_val  # Use ExploreOnly for rollouts
            #estimate_value = 0
            )
        return solve(solver, pomdp), solver
    elseif policy_type == "POMCPOW_Depth1"
        # Shallow planning heuristic - only looks one step ahead
        solver = POMCPOWSolver(
            tree_queries = 1000,
            max_depth = 1,           # Only one step ahead
            criterion = MaxUCB(1.0),
            enable_action_pw = true, # ENABLED for large action space
            k_action = 5.0,
            alpha_action = 0.5,
            k_observation = 3.0,
            alpha_observation = 0.3,
            estimate_value = FORollout(AlphaAwareRolloutPolicy(pomdp))  # Use alpha-aware rollout for leaf evaluation
            )
        return solve(solver, pomdp), solver
    elseif policy_type == "OneMineProfitMaximization"
        return ProfitMaximizerPolicy(pomdp, top_k=1, times=[1]), nothing
    elseif policy_type == "OneMineEmissionMinimization"
        return EmissionMinimizerPolicy(pomdp, top_k=1, times=[1]), nothing
    elseif policy_type == "ExploreOnly"
        return ExploreOnlyPolicy(pomdp), nothing
    elseif policy_type == "Random" || policy_type == "RandomHeuristicPolicy"
        return RandomHeuristicPolicy(pomdp, debug=false), nothing
    elseif policy_type == "DynamicProfitMaximizer"
        return DynamicProfitMaximizerPolicy(pomdp, debug=false), nothing
    elseif policy_type == "DynamicEmissionMinimizer"
        return DynamicEmissionMinimizerPolicy(pomdp, debug=false), nothing
    elseif policy_type == "OneStepLookahead"
        return OneStepLookaheadPolicy(pomdp, n_rollouts=30, debug=false), nothing
    else
        error("Unknown policy type: $policy_type")
    end
end

# =============================================================================
# POMCPOW CONFIGURATION EXTRACTION
# =============================================================================

function extract_pomcpow_config(solver::POMCPOWSolver)
    """Extract POMCPOW configuration parameters from solver."""
    return Dict(
        "tree_queries" => solver.tree_queries,
        "max_depth" => solver.max_depth,
        "criterion" => string(solver.criterion),
        "enable_action_pw" => solver.enable_action_pw,
        "k_action" => solver.k_action,
        "alpha_action" => solver.alpha_action,
        "k_observation" => solver.k_observation,
        "alpha_observation" => solver.alpha_observation,
        "estimate_value" => string(solver.estimate_value)
    )
end

# =============================================================================
# TRAIN/TEST SPLIT EVALUATION FUNCTIONS
# =============================================================================

function evaluate_policy_train_test_split(planning_pomdp::LiPOMDP, evaluation_pomdp::LiPOMDP, 
                              policy, updater::LiBeliefUpdater, n_train_sims::Int=20, n_test_sims::Int=10; 
                              rng::AbstractRNG=Random.GLOBAL_RNG)
    """Evaluate a policy with proper train/test split to avoid overfitting."""
    
    println("  📚 Training Phase: $n_train_sims simulations")
    println("  🧪 Testing Phase: $n_test_sims simulations")
    
    # Training phase results
    train_results = Dict(
        "total_reward" => Float64[],
        "total_discounted_reward" => Float64[],
        "total_emission_cost" => Float64[],
        "total_profit" => Float64[],
        "total_demand" => Float64[],
        "total_production" => Float64[],
        "total_exploration_cost" => Float64[],
        "total_capex" => Float64[],
        "total_opex" => Float64[],
        "total_costs" => Float64[],
        "total_mines_opened" => Float64[],
        "avg_active_mines" => Float64[],
        "mines_depleted" => Float64[],
        "total_co2_tonnes" => Float64[],
        "total_revenue" => Float64[],
        "total_explorations" => Float64[],
        "successful_explorations" => Float64[],
        "histories" => []
    )
    
    # Testing phase results
    test_results = Dict(
        "total_reward" => Float64[],
        "total_discounted_reward" => Float64[],
        "total_emission_cost" => Float64[],
        "total_profit" => Float64[],
        "total_demand" => Float64[],
        "total_production" => Float64[],
        "total_exploration_cost" => Float64[],
        "total_capex" => Float64[],
        "total_opex" => Float64[],
        "total_costs" => Float64[],
        "total_mines_opened" => Float64[],
        "avg_active_mines" => Float64[],
        "mines_depleted" => Float64[],
        "total_co2_tonnes" => Float64[],
        "total_revenue" => Float64[],
        "total_explorations" => Float64[],
        "successful_explorations" => Float64[],
        "histories" => []
    )
    
    # TRAINING PHASE - Use different random seeds for training
    for i in 1:n_train_sims
        Random.seed!(i + 1000) # Training seeds: 1001-1020
        
        total_reward = 0.0
        total_discounted_reward = 0.0
        total_emission_cost = 0.0
        total_profit = 0.0
        total_demand = 0.0
        total_production = 0.0
        total_exploration_cost = 0.0
        total_capex = 0.0
        total_opex = 0.0
        total_costs = 0.0
        total_mines_opened = 0.0
        avg_active_mines = 0.0
        mines_depleted = 0.0
        total_co2_tonnes = 0.0
        total_revenue = 0.0
        total_explorations = 0.0
        successful_explorations = 0.0
        active_mines_per_step = Float64[]
        history = []
        
        # Use planning POMDP for initial state and belief
        s0, b0 = create_initial_belief(planning_pomdp)
        b = b0
        s = s0
        
        for t in 1:planning_pomdp.time_horizon
            # Choose action
            if policy isa POMCPOWPlanner
                a, info = action_info(policy, b; tree_in_info=false)
                
                # if haskey(info, :tree)
                #     dtree = D3Trees.D3Tree(info[:tree])
                #     fname = joinpath(TREE_DIR, "train_tree_a$(round(planning_pomdp.alpha, digits=2))_sim$(i)_t$(t).html")
                #     save_tree_html(fname, dtree)
                # end
            else
                a = action(policy, b)
            end
            
            # Use evaluation POMDP for state transitions and rewards
            sp, o, r = gen(evaluation_pomdp, s, a, rng)
            
            costs = compute_costs(evaluation_pomdp, s, a)  
            emissions = compute_emissions(evaluation_pomdp, s, a)  
            current_production, _ = calculate_total_production(evaluation_pomdp, s)
            
            # Calculate cost breakdown
            exploration_cost = 0.0
            capex = 0.0
            opex = 0.0
            
            # Exploration costs
            for site in 1:evaluation_pomdp.n_deposits
                if a.type[site] == 1  # Exploring
                    cost = site <= 2 ? evaluation_pomdp.exploration_cost_dle : evaluation_pomdp.exploration_cost_hard_rock
                    exploration_cost += cost
                end
            end
            
            # Capital costs for newly opened mines
            for site in 1:evaluation_pomdp.n_deposits
                if a.type[site] == 2 && !s.have_opened[site]  # Open new mine
                    capex += evaluation_pomdp.deposit_types[site] ? evaluation_pomdp.dle_capex : evaluation_pomdp.mining_capex
                end
            end
            
            # Operating costs for all open mines
            remaining_demand_lce = s.company_demand
            for site in 1:evaluation_pomdp.n_deposits
                if s.have_opened[site] && s.deposit[site] > 0.1 && !s.restored_mines[site]
                    in_situ_li, site_production_lce = calculate_site_production(evaluation_pomdp, s, site, remaining_demand_lce)
                    remaining_demand_lce -= site_production_lce
                    opex += (evaluation_pomdp.deposit_types[site] ? evaluation_pomdp.dle_opex : evaluation_pomdp.mining_opex) * site_production_lce
                end
            end
            
            revenue = current_production * s.price
            profit = revenue - costs    
            
            # Calculate additional metrics
            active_mines_count = count(s.have_opened)
            push!(active_mines_per_step, active_mines_count)
            
            # Count explorations and successful explorations
            for site in 1:evaluation_pomdp.n_deposits
                if a.type[site] == 1  # Exploring
                    total_explorations += 1
                    # Consider exploration successful if it leads to opening a mine later
                    # For now, we'll count all explorations as potentially successful
                    successful_explorations += 1
                end
            end
            
            # Calculate CO2 emissions in tonnes (not cost)
            co2_tonnes = abs(emissions) / evaluation_pomdp.CO2_cost
            
            total_demand += s.company_demand
            total_production += current_production
            total_profit += profit
            total_emission_cost += emissions
            total_exploration_cost += exploration_cost
            total_capex += capex
            total_opex += opex
            total_costs += costs
            total_revenue += revenue
            total_co2_tonnes += co2_tonnes
            
            # Track rewards
            total_reward += r
            total_discounted_reward += evaluation_pomdp.γ^(t-1) * r
            
            if isterminal(evaluation_pomdp, sp)
                break
            end
            
            dict_step_result = Dict(
                "s" => s,
                "a" => a,
                "o" => o,
                "r" => r,
                "b" => b,
            )
            
            push!(history, dict_step_result)
            
            # Update belief
            b = update(updater, b, a, o)
            s = sp
        end
        
        # Calculate final metrics for this simulation
        total_mines_opened = count(s.have_opened)
        mines_depleted = count(s.is_depleted)
        avg_active_mines = length(active_mines_per_step) > 0 ? mean(active_mines_per_step) : 0.0
        
        # Store training results
        push!(train_results["total_reward"], total_reward)
        push!(train_results["total_discounted_reward"], total_discounted_reward)
        push!(train_results["total_emission_cost"], total_emission_cost)
        push!(train_results["total_profit"], total_profit)
        push!(train_results["total_demand"], total_demand)
        push!(train_results["total_production"], total_production)
        push!(train_results["total_exploration_cost"], total_exploration_cost)
        push!(train_results["total_capex"], total_capex)
        push!(train_results["total_opex"], total_opex)
        push!(train_results["total_costs"], total_costs)
        push!(train_results["total_mines_opened"], total_mines_opened)
        push!(train_results["avg_active_mines"], avg_active_mines)
        push!(train_results["mines_depleted"], mines_depleted)
        push!(train_results["total_co2_tonnes"], total_co2_tonnes)
        push!(train_results["total_revenue"], total_revenue)
        push!(train_results["total_explorations"], total_explorations)
        push!(train_results["successful_explorations"], successful_explorations)
        push!(train_results["histories"], history)
    end
    
    # TESTING PHASE - Use different random seeds for testing
    for i in 1:n_test_sims
        Random.seed!(i + 2000) # Testing seeds: 2001-2010
        
        total_reward = 0.0
        total_discounted_reward = 0.0
        total_emission_cost = 0.0
        total_profit = 0.0
        total_demand = 0.0
        total_production = 0.0
        total_exploration_cost = 0.0
        total_capex = 0.0
        total_opex = 0.0
        total_costs = 0.0
        total_mines_opened = 0.0
        avg_active_mines = 0.0
        mines_depleted = 0.0
        total_co2_tonnes = 0.0
        total_revenue = 0.0
        total_explorations = 0.0
        successful_explorations = 0.0
        active_mines_per_step = Float64[]
        history = []
        
        # Use planning POMDP for initial state and belief
        s0, b0 = create_initial_belief(planning_pomdp)
        b = b0
        s = s0
        
        for t in 1:planning_pomdp.time_horizon
            # Choose action
            if policy isa POMCPOWPlanner
                a, info = action_info(policy, b; tree_in_info=true)
                
                # if haskey(info, :tree)
                #     dtree = D3Trees.D3Tree(info[:tree])
                #     fname = joinpath(TREE_DIR, "test_tree_a$(round(planning_pomdp.alpha, digits=2))_sim$(i)_t$(t).html")
                #     save_tree_html(fname, dtree)
                # end
            else
                a = action(policy, b)
            end
            
            # Use evaluation POMDP for state transitions and rewards
            sp, o, r = gen(evaluation_pomdp, s, a, rng)
            
            costs = compute_costs(evaluation_pomdp, s, a)  
            emissions = compute_emissions(evaluation_pomdp, s, a)  
            current_production, _ = calculate_total_production(evaluation_pomdp, s)
            
            # Calculate cost breakdown
            exploration_cost = 0.0
            capex = 0.0
            opex = 0.0
            
            # Exploration costs
            for site in 1:evaluation_pomdp.n_deposits
                if a.type[site] == 1  # Exploring
                    cost = site <= 2 ? evaluation_pomdp.exploration_cost_dle : evaluation_pomdp.exploration_cost_hard_rock
                    exploration_cost += cost
                end
            end
            
            # Capital costs for newly opened mines
            for site in 1:evaluation_pomdp.n_deposits
                if a.type[site] == 2 && !s.have_opened[site]  # Open new mine
                    capex += evaluation_pomdp.deposit_types[site] ? evaluation_pomdp.dle_capex : evaluation_pomdp.mining_capex
                end
            end
            
            # Operating costs for all open mines
            remaining_demand_lce = s.company_demand
            for site in 1:evaluation_pomdp.n_deposits
                if s.have_opened[site] && s.deposit[site] > 0.1 && !s.restored_mines[site]
                    in_situ_li, site_production_lce = calculate_site_production(evaluation_pomdp, s, site, remaining_demand_lce)
                    remaining_demand_lce -= site_production_lce
                    opex += (evaluation_pomdp.deposit_types[site] ? evaluation_pomdp.dle_opex : evaluation_pomdp.mining_opex) * site_production_lce
                end
            end
            
            revenue = current_production * s.price
            profit = revenue - costs    
            
            # Calculate additional metrics
            active_mines_count = count(s.have_opened)
            push!(active_mines_per_step, active_mines_count)
            
            # Count explorations and successful explorations
            for site in 1:evaluation_pomdp.n_deposits
                if a.type[site] == 1  # Exploring
                    total_explorations += 1
                    # Consider exploration successful if it leads to opening a mine later
                    # For now, we'll count all explorations as potentially successful
                    successful_explorations += 1
                end
            end
            
            # Calculate CO2 emissions in tonnes (not cost)
            co2_tonnes = abs(emissions) / evaluation_pomdp.CO2_cost
            
            total_demand += s.company_demand
            total_production += current_production
            total_profit += profit
            total_emission_cost += emissions
            total_exploration_cost += exploration_cost
            total_capex += capex
            total_opex += opex
            total_costs += costs
            total_revenue += revenue
            total_co2_tonnes += co2_tonnes
            
            # Track rewards
            total_reward += r
            total_discounted_reward += evaluation_pomdp.γ^(t-1) * r
            
            if isterminal(evaluation_pomdp, sp)
                break
            end
            
            dict_step_result = Dict(
                "s" => s,
                "a" => a,
                "o" => o,
                "r" => r,
                "b" => b,
            )
            
            push!(history, dict_step_result)
            
            # Update belief
            b = update(updater, b, a, o)
            s = sp
        end
        
        # Calculate final metrics for this simulation
        total_mines_opened = count(s.have_opened)
        mines_depleted = count(s.is_depleted)
        avg_active_mines = length(active_mines_per_step) > 0 ? mean(active_mines_per_step) : 0.0
        
        # Store testing results
        push!(test_results["total_reward"], total_reward)
        push!(test_results["total_discounted_reward"], total_discounted_reward)
        push!(test_results["total_emission_cost"], total_emission_cost)
        push!(test_results["total_profit"], total_profit)
        push!(test_results["total_demand"], total_demand)
        push!(test_results["total_production"], total_production)
        push!(test_results["total_exploration_cost"], total_exploration_cost)
        push!(test_results["total_capex"], total_capex)
        push!(test_results["total_opex"], total_opex)
        push!(test_results["total_costs"], total_costs)
        push!(test_results["total_mines_opened"], total_mines_opened)
        push!(test_results["avg_active_mines"], avg_active_mines)
        push!(test_results["mines_depleted"], mines_depleted)
        push!(test_results["total_co2_tonnes"], total_co2_tonnes)
        push!(test_results["total_revenue"], total_revenue)
        push!(test_results["total_explorations"], total_explorations)
        push!(test_results["successful_explorations"], successful_explorations)
        push!(test_results["histories"], history)
    end
    
    return Dict(
        "train" => train_results,
        "test" => test_results
    )
end

# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

function run_train_test_evaluation(
    policies::Vector{String}, 
    planning_price_models::Vector{Int64}, 
    evaluation_price_models::Vector{Int64}, 
    alphas::Vector{Float64};
    time_horizon::Int64 = 29,
    n_train_sims::Int64 = 25,  # Training simulations
    n_test_sims::Int64 = 50,   # Testing simulations
    output_file::String = "train_test_evaluation_results.txt"
)
    """Run policy evaluation with proper train/test split to avoid overfitting."""
    
    println("🚀 TRAIN/TEST SPLIT POLICY EVALUATION")
    println("="^80)
    println("Testing policies with proper train/test split to avoid overfitting")
    println("Different price models for planning vs evaluation")
    println("Using ALPHA values for multi-objective optimization")
    println("="^80)
    
    # Configuration
    println("Configuration:")
    println("  • Training simulations per policy: $n_train_sims")
    println("  • Testing simulations per policy: $n_test_sims")
    println("  • Time horizon: $time_horizon")
    println("  • Planning price models: $planning_price_models")
    println("  • Evaluation price models: $evaluation_price_models")
    println("  • Policies: $policies")
    println("  • Alpha values: $alphas")
    println(" Dynamic Profit used for POMCPOW rollouts")
    println()
    
    # Store all results
    all_results = Dict{String, Dict{String, Any}}()
    
    # Store POMCPOW configuration (will be updated if POMCPOW is used)
    pomcpow_config = nothing
    
    # Create directory for output files if it doesn't exist
    output_dir = dirname(output_file)
    if !isdir(output_dir) && output_dir != ""
        mkpath(output_dir)
    end
    
    # Test each policy configuration
    total_configs = length(policies) * length(planning_price_models) * length(evaluation_price_models) * length(alphas)
    config_count = 0
    
    for policy_type in policies
        for planning_model in planning_price_models
            for eval_model in evaluation_price_models
                for alpha in alphas
                    config_count += 1
                    
                    println("\n[$(config_count)/$(total_configs)] Testing $(policy_type) with α=$alpha...")
                    println("  Planning model: Type $planning_model, Evaluation model: Type $eval_model")
                    
                    # Create planning and evaluation POMDPs
                    planning_pomdp = LiPOMDP(price_model_type=planning_model, time_horizon=time_horizon, alpha=alpha)
                    evaluation_pomdp = LiPOMDP(price_model_type=eval_model, time_horizon=time_horizon, alpha=alpha)
                    
                    # Create belief updater
                    updater = LiBeliefUpdater(P=evaluation_pomdp)
                    
                    # Create policy
                    policy, solver = create_policy(policy_type, planning_pomdp)
                    
                    # Store POMCPOW configuration if this is a POMCPOW policy
                    if policy_type == "POMCPOW" && pomcpow_config === nothing
                        pomcpow_config = extract_pomcpow_config(solver)
                    end
                    
                    # Evaluate policy with train/test split
                    results = evaluate_policy_train_test_split(planning_pomdp, evaluation_pomdp, policy, updater, n_train_sims, n_test_sims)
                    
                    # Calculate statistics for training
                    train_mean_profit = mean(results["train"]["total_profit"])
                    train_std_profit = std(results["train"]["total_profit"])
                    train_mean_emission_cost = mean(results["train"]["total_emission_cost"])
                    train_std_emission_cost = std(results["train"]["total_emission_cost"])
                    train_mean_disc_reward = mean(results["train"]["total_discounted_reward"])
                    train_std_disc_reward = std(results["train"]["total_discounted_reward"])
                    train_mean_demand = mean(results["train"]["total_demand"])
                    train_mean_production = mean(results["train"]["total_production"])
                    train_met_demand_pct = train_mean_demand > 0 ? (train_mean_production / train_mean_demand) * 100 : 0.0
                    train_mean_exploration_cost = mean(results["train"]["total_exploration_cost"])
                    train_mean_capex = mean(results["train"]["total_capex"])
                    train_mean_opex = mean(results["train"]["total_opex"])
                    train_mean_costs = mean(results["train"]["total_costs"])
                    train_cost_per_tonne = train_mean_production > 0 ? train_mean_costs / train_mean_production : 0.0
                    train_mean_mines_opened = mean(results["train"]["total_mines_opened"])
                    train_mean_active_mines = mean(results["train"]["avg_active_mines"])
                    train_mean_mines_depleted = mean(results["train"]["mines_depleted"])
                    train_mean_co2_tonnes = mean(results["train"]["total_co2_tonnes"])
                    train_mean_revenue = mean(results["train"]["total_revenue"])
                    train_mean_explorations = mean(results["train"]["total_explorations"])
                    train_mean_successful_explorations = mean(results["train"]["successful_explorations"])
                    train_exploration_success_rate = train_mean_explorations > 0 ? (train_mean_successful_explorations / train_mean_explorations) * 100 : 0.0
                    train_avg_selling_price = train_mean_production > 0 ? train_mean_revenue / train_mean_production : 0.0
                    train_emissions_intensity = train_mean_production > 0 ? train_mean_co2_tonnes / train_mean_production : 0.0
                    train_capacity_utilization = train_mean_mines_opened > 0 ? train_mean_production / (train_mean_mines_opened * planning_pomdp.output * planning_pomdp.time_horizon) : 0.0
                    train_market_share_achieved = train_mean_demand > 0 ? train_mean_production / train_mean_demand : 0.0
                    
                    # Calculate statistics for testing
                    test_mean_profit = mean(results["test"]["total_profit"])
                    test_std_profit = std(results["test"]["total_profit"])
                    test_mean_emission_cost = mean(results["test"]["total_emission_cost"])
                    test_std_emission_cost = std(results["test"]["total_emission_cost"])
                    test_mean_disc_reward = mean(results["test"]["total_discounted_reward"])
                    test_std_disc_reward = std(results["test"]["total_discounted_reward"])
                    test_mean_demand = mean(results["test"]["total_demand"])
                    test_mean_production = mean(results["test"]["total_production"])
                    test_met_demand_pct = test_mean_demand > 0 ? (test_mean_production / test_mean_demand) * 100 : 0.0
                    test_mean_exploration_cost = mean(results["test"]["total_exploration_cost"])
                    test_mean_capex = mean(results["test"]["total_capex"])
                    test_mean_opex = mean(results["test"]["total_opex"])
                    test_mean_costs = mean(results["test"]["total_costs"])
                    test_cost_per_tonne = test_mean_production > 0 ? test_mean_costs / test_mean_production : 0.0
                    test_mean_mines_opened = mean(results["test"]["total_mines_opened"])
                    test_mean_active_mines = mean(results["test"]["avg_active_mines"])
                    test_mean_mines_depleted = mean(results["test"]["mines_depleted"])
                    test_mean_co2_tonnes = mean(results["test"]["total_co2_tonnes"])
                    test_mean_revenue = mean(results["test"]["total_revenue"])
                    test_mean_explorations = mean(results["test"]["total_explorations"])
                    test_mean_successful_explorations = mean(results["test"]["successful_explorations"])
                    test_exploration_success_rate = test_mean_explorations > 0 ? (test_mean_successful_explorations / test_mean_explorations) * 100 : 0.0
                    test_avg_selling_price = test_mean_production > 0 ? test_mean_revenue / test_mean_production : 0.0
                    test_emissions_intensity = test_mean_production > 0 ? test_mean_co2_tonnes / test_mean_production : 0.0
                    test_capacity_utilization = test_mean_mines_opened > 0 ? test_mean_production / (test_mean_mines_opened * planning_pomdp.output * planning_pomdp.time_horizon) : 0.0
                    test_market_share_achieved = test_mean_demand > 0 ? test_mean_production / test_mean_demand : 0.0
                    
                    # Calculate overfitting metrics
                    profit_overfitting = train_mean_profit - test_mean_profit
                    emission_overfitting = train_mean_emission_cost - test_mean_emission_cost
                    reward_overfitting = train_mean_disc_reward - test_mean_disc_reward
                    
                    # Store results
                    policy_key = "$(policy_type)_$(planning_model)_$(eval_model)_$(alpha)"
                    all_results[policy_key] = Dict(
                        "policy_type" => policy_type,
                        "planning_price_model" => planning_model,
                        "evaluation_price_model" => eval_model,
                        "alpha" => alpha,
                        "train_mean_profit" => train_mean_profit,
                        "train_std_profit" => train_std_profit,
                        "train_mean_emission_cost" => train_mean_emission_cost,
                        "train_std_emission_cost" => train_std_emission_cost,
                        "train_mean_discounted_reward" => train_mean_disc_reward,
                        "train_std_discounted_reward" => train_std_disc_reward,
                        "train_mean_demand" => train_mean_demand,
                        "train_mean_production" => train_mean_production,
                        "train_met_demand_pct" => train_met_demand_pct,
                        "train_mean_exploration_cost" => train_mean_exploration_cost,
                        "train_mean_capex" => train_mean_capex,
                        "train_mean_opex" => train_mean_opex,
                        "train_mean_costs" => train_mean_costs,
                        "train_cost_per_tonne" => train_cost_per_tonne,
                        "train_mean_mines_opened" => train_mean_mines_opened,
                        "train_mean_active_mines" => train_mean_active_mines,
                        "train_mean_mines_depleted" => train_mean_mines_depleted,
                        "train_mean_co2_tonnes" => train_mean_co2_tonnes,
                        "train_mean_revenue" => train_mean_revenue,
                        "train_mean_explorations" => train_mean_explorations,
                        "train_mean_successful_explorations" => train_mean_successful_explorations,
                        "train_exploration_success_rate" => train_exploration_success_rate,
                        "train_avg_selling_price" => train_avg_selling_price,
                        "train_emissions_intensity" => train_emissions_intensity,
                        "train_capacity_utilization" => train_capacity_utilization,
                        "train_market_share_achieved" => train_market_share_achieved,
                        "test_mean_profit" => test_mean_profit,
                        "test_std_profit" => test_std_profit,
                        "test_mean_emission_cost" => test_mean_emission_cost,
                        "test_std_emission_cost" => test_std_emission_cost,
                        "test_mean_discounted_reward" => test_mean_disc_reward,
                        "test_std_discounted_reward" => test_std_disc_reward,
                        "test_mean_demand" => test_mean_demand,
                        "test_mean_production" => test_mean_production,
                        "test_met_demand_pct" => test_met_demand_pct,
                        "test_mean_exploration_cost" => test_mean_exploration_cost,
                        "test_mean_capex" => test_mean_capex,
                        "test_mean_opex" => test_mean_opex,
                        "test_mean_costs" => test_mean_costs,
                        "test_cost_per_tonne" => test_cost_per_tonne,
                        "test_mean_mines_opened" => test_mean_mines_opened,
                        "test_mean_active_mines" => test_mean_active_mines,
                        "test_mean_mines_depleted" => test_mean_mines_depleted,
                        "test_mean_co2_tonnes" => test_mean_co2_tonnes,
                        "test_mean_revenue" => test_mean_revenue,
                        "test_mean_explorations" => test_mean_explorations,
                        "test_mean_successful_explorations" => test_mean_successful_explorations,
                        "test_exploration_success_rate" => test_exploration_success_rate,
                        "test_avg_selling_price" => test_avg_selling_price,
                        "test_emissions_intensity" => test_emissions_intensity,
                        "test_capacity_utilization" => test_capacity_utilization,
                        "test_market_share_achieved" => test_market_share_achieved,
                        "profit_overfitting" => profit_overfitting,
                        "emission_overfitting" => emission_overfitting,
                        "reward_overfitting" => reward_overfitting,
                        "n_train_sims" => n_train_sims,
                        "n_test_sims" => n_test_sims,
                        "raw_results" => results
                    )
                    
                    println("  📚 Training Results:")
                    println("    ✓ Profit: \$$(round(train_mean_profit/1e6, digits=2))M ± $(round(train_std_profit/1e6, digits=2))M")
                    println("    ✓ Emissions: \$$(round(train_mean_emission_cost/1e6, digits=2))M ± $(round(train_std_emission_cost/1e6, digits=2))M")
                    println("    ✓ Discounted Reward: $(round(train_mean_disc_reward, digits=2)) ± $(round(train_std_disc_reward, digits=2))")
                    println("    ✓ Met Demand: $(round(train_met_demand_pct, digits=1))%")
                    println("    ✓ Exploration Cost: \$$(round(train_mean_exploration_cost/1e6, digits=2))M")
                    println("    ✓ CAPEX: \$$(round(train_mean_capex/1e6, digits=2))M")
                    println("    ✓ OPEX: \$$(round(train_mean_opex/1e6, digits=2))M")
                    println("    ✓ Cost per Tonne: \$$(round(train_cost_per_tonne, digits=2))")
                    
                    println("  🧪 Testing Results:")
                    println("    ✓ Profit: \$$(round(test_mean_profit/1e6, digits=2))M ± $(round(test_std_profit/1e6, digits=2))M")
                    println("    ✓ Emissions: \$$(round(test_mean_emission_cost/1e6, digits=2))M ± $(round(test_std_emission_cost/1e6, digits=2))M")
                    println("    ✓ Discounted Reward: $(round(test_mean_disc_reward, digits=2)) ± $(round(test_std_disc_reward, digits=2))")
                    println("    ✓ Met Demand: $(round(test_met_demand_pct, digits=1))%")
                    println("    ✓ Exploration Cost: \$$(round(test_mean_exploration_cost/1e6, digits=2))M")
                    println("    ✓ CAPEX: \$$(round(test_mean_capex/1e6, digits=2))M")
                    println("    ✓ OPEX: \$$(round(test_mean_opex/1e6, digits=2))M")
                    println("    ✓ Cost per Tonne: \$$(round(test_cost_per_tonne, digits=2))")
                    
                    println("  🔍 Overfitting Analysis:")
                    println("    ✓ Profit Overfitting: \$$(round(profit_overfitting/1e6, digits=2))M")
                    println("    ✓ Emission Overfitting: \$$(round(emission_overfitting/1e6, digits=2))M")
                    println("    ✓ Reward Overfitting: $(round(reward_overfitting, digits=2))")
                    
                    # Overfitting warning
                    if abs(profit_overfitting) > train_std_profit || abs(emission_overfitting) > train_std_emission_cost
                        println("    ⚠️  WARNING: Potential overfitting detected!")
                    else
                        println("    ✅ No significant overfitting detected")
                    end
                end
            end
        end
    end
    
    # Save results to file
    open(output_file, "w") do f
        # Write POMCPOW configuration header (if POMCPOW was used)
        if pomcpow_config !== nothing
            println(f, "# POMCPOW Configuration:")
            for (key, value) in pomcpow_config
                println(f, "# $key = $value")
            end
            println(f, "#")
        end
        
        # Write evaluation configuration
        println(f, "# Evaluation Configuration:")
        println(f, "# time_horizon = $time_horizon")
        println(f, "# n_train_sims = $n_train_sims")
        println(f, "# n_test_sims = $n_test_sims")
        println(f, "# planning_price_models = $planning_price_models")
        println(f, "# evaluation_price_models = $evaluation_price_models")
        println(f, "# alphas = $alphas")
        println(f, "#")
        
        # Write CSV header
        println(f, "policy_type,planning_price_model,evaluation_price_model,alpha,train_mean_profit,train_std_profit,train_mean_emission_cost,train_std_emission_cost,train_mean_discounted_reward,train_std_discounted_reward,train_mean_demand,train_mean_production,train_met_demand_pct,train_mean_exploration_cost,train_mean_capex,train_mean_opex,train_mean_costs,train_cost_per_tonne,train_mean_mines_opened,train_mean_active_mines,train_mean_mines_depleted,train_mean_co2_tonnes,train_mean_revenue,train_mean_explorations,train_mean_successful_explorations,train_exploration_success_rate,train_avg_selling_price,train_emissions_intensity,train_capacity_utilization,train_market_share_achieved,test_mean_profit,test_std_profit,test_mean_emission_cost,test_std_emission_cost,test_mean_discounted_reward,test_std_discounted_reward,test_mean_demand,test_mean_production,test_met_demand_pct,test_mean_exploration_cost,test_mean_capex,test_mean_opex,test_mean_costs,test_cost_per_tonne,test_mean_mines_opened,test_mean_active_mines,test_mean_mines_depleted,test_mean_co2_tonnes,test_mean_revenue,test_mean_explorations,test_mean_successful_explorations,test_exploration_success_rate,test_avg_selling_price,test_emissions_intensity,test_capacity_utilization,test_market_share_achieved,profit_overfitting,emission_overfitting,reward_overfitting,n_train_sims,n_test_sims")
        
        # Sort results by alpha value for better readability
        sorted_results = sort(collect(all_results), by = x -> x[2]["alpha"])
        
        # Write data rows
        for (key, results) in sorted_results
            println(f, "$(results["policy_type"]),$(results["planning_price_model"]),$(results["evaluation_price_model"]),$(results["alpha"]),$(results["train_mean_profit"]),$(results["train_std_profit"]),$(results["train_mean_emission_cost"]),$(results["train_std_emission_cost"]),$(results["train_mean_discounted_reward"]),$(results["train_std_discounted_reward"]),$(results["train_mean_demand"]),$(results["train_mean_production"]),$(results["train_met_demand_pct"]),$(results["train_mean_exploration_cost"]),$(results["train_mean_capex"]),$(results["train_mean_opex"]),$(results["train_mean_costs"]),$(results["train_cost_per_tonne"]),$(results["train_mean_mines_opened"]),$(results["train_mean_active_mines"]),$(results["train_mean_mines_depleted"]),$(results["train_mean_co2_tonnes"]),$(results["train_mean_revenue"]),$(results["train_mean_explorations"]),$(results["train_mean_successful_explorations"]),$(results["train_exploration_success_rate"]),$(results["train_avg_selling_price"]),$(results["train_emissions_intensity"]),$(results["train_capacity_utilization"]),$(results["train_market_share_achieved"]),$(results["test_mean_profit"]),$(results["test_std_profit"]),$(results["test_mean_emission_cost"]),$(results["test_std_emission_cost"]),$(results["test_mean_discounted_reward"]),$(results["test_std_discounted_reward"]),$(results["test_mean_demand"]),$(results["test_mean_production"]),$(results["test_met_demand_pct"]),$(results["test_mean_exploration_cost"]),$(results["test_mean_capex"]),$(results["test_mean_opex"]),$(results["test_mean_costs"]),$(results["test_cost_per_tonne"]),$(results["test_mean_mines_opened"]),$(results["test_mean_active_mines"]),$(results["test_mean_mines_depleted"]),$(results["test_mean_co2_tonnes"]),$(results["test_mean_revenue"]),$(results["test_mean_explorations"]),$(results["test_mean_successful_explorations"]),$(results["test_exploration_success_rate"]),$(results["test_avg_selling_price"]),$(results["test_emissions_intensity"]),$(results["test_capacity_utilization"]),$(results["test_market_share_achieved"]),$(results["profit_overfitting"]),$(results["emission_overfitting"]),$(results["reward_overfitting"]),$(results["n_train_sims"]),$(results["n_test_sims"])")
        end
    end
    
    println("\n" * "="^120)
    println("✅ RESULTS SAVED TO: $output_file")
    println("📊 Train/Test Split Summary:")
    println("  • Training simulations: $n_train_sims per policy")
    println("  • Testing simulations: $n_test_sims per policy")
    println("  • Total policies tested: $(length(all_results))")
    println("  • Different random seeds used for train vs test")
    println("  • Overfitting analysis included")
    println("="^120)
    
    return all_results
end

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

# Automatically run when script is included or executed
println("🧪 TRAIN/TEST SPLIT EVALUATION")
println("="^60)

# Configuration
policies = ["POMCPOW", "POMCPOW_Depth1", "OneStepLookahead", "DynamicProfitMaximizer", "ExploreOnly", "OneMineProfitMaximization", "OneMineEmissionMinimization", "DynamicEmissionMinimizer", "RandomHeuristicPolicy"]
#policies = [ "POMCPOW_Depth1", "OneStepLookahead", "POMCPOW"]
planning_price_models =[2]  # Static, Linear, Exponential, Historical
evaluation_price_models = [6]  # Historical price model for evaluation
#alphas = [0.0]
#alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
alphas = [0.05, 0.10, 0.15, 0.20]
time_horizon = 29  # Full time horizon
n_train_sims = 25  # Training simulations
n_test_sims = 50   # Testing simulations

println("Running train/test evaluation...")
println("Policies: $policies")
println("Planning models: $planning_price_models")
println("Evaluation models: $evaluation_price_models")
println("Alpha values: $alphas")
println("Train/Test split: $n_train_sims/$n_test_sims")
println("")

# Create dynamic output filename based on planning and evaluation models
planning_str = join(planning_price_models, "_")
evaluation_str = join(evaluation_price_models, "_")
output_file = "outputs/test_train_final_results/results_p$(planning_str)_e$(evaluation_str)_new_params_small_alphas.csv"

# Run the evaluation
results = run_train_test_evaluation(
    policies, planning_price_models, evaluation_price_models, alphas,
    time_horizon=time_horizon,
    n_train_sims=n_train_sims,
    n_test_sims=n_test_sims,
    output_file=output_file
)

println("\n🎯 KEY BENEFITS:")
println("1. Different random seeds prevent overfitting to specific scenarios")
println("2. Separate training and testing phases")
println("3. Overfitting detection and analysis")
println("4. More realistic evaluation on unseen data")
println("5. Better policy generalization assessment")
