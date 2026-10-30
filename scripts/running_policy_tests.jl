# Import without other commands doesn't do anything
using LiPOMDPs
using POMDPs
using POMDPTools
using POMCPOW
using Random
using Statistics
using Distributions
using JSON
using D3Trees
using POMCPOW: POMCPOWPlanner  # so `isa POMCPOWPlanner` works
using POMDPTools


# const TREE_DIR = "outputs/trees"
# isdir(TREE_DIR) || mkpath(TREE_DIR)

# function save_tree_html(path::AbstractString, dtree::D3Trees.D3Tree)
#     open(path, "w") do io
#         show(io, MIME"text/html"(), dtree)  # writes an interactive D3 page
#     end
# end

"""
Simple Policy Testing Script with different price models for planning vs evaluation.
"""

# Helper function to convert action to readable format
function action_to_string(action::Action)
    site_actions = []
    site_names = ["DLE-1", "DLE-2", "Mining-3", "Mining-4"]
    
    for (i, (site, type)) in enumerate(zip(action.site, action.type))
        # For each site, determine the action based on type
        # Use i (1-indexed position) to get site name instead of site value
        site_name = i <= 4 ? site_names[i] : "Site-$i"
        
        if type == 0
            push!(site_actions, "DoNothing")
        elseif type == 1
            push!(site_actions, "Explore $site_name")
        elseif type == 2
            push!(site_actions, "Open $site_name")
        elseif type == 3
            push!(site_actions, "Restore $site_name")
        else
            push!(site_actions, "Unknown($site,$type)")
        end
    end
    return join(site_actions, " | ")
end

# Global variable to store POMCPOW configuration for printing
global POMCPOW_CONFIG = nothing

function print_pomcpow_config()
    """Print current POMCPOW configuration dynamically."""
    if POMCPOW_CONFIG === nothing
        println("\n🔧 POMCPOW CONFIGURATION: Not available")
        return
    end
    
    println("\n🔧 POMCPOW CONFIGURATION:")
    println("  • Tree queries: $(POMCPOW_CONFIG.tree_queries)")
    println("  • Max depth: $(POMCPOW_CONFIG.max_depth)")
    println("  • UCB criterion: $(POMCPOW_CONFIG.criterion)")
    println("  • Enable action progressive widening: $(POMCPOW_CONFIG.enable_action_pw)")
    println("  • k_action: $(POMCPOW_CONFIG.k_action)")
    println("  • alpha_action: $(POMCPOW_CONFIG.alpha_action)")
    println("  • k_observation: $(POMCPOW_CONFIG.k_observation)")
    println("  • alpha_observation: $(POMCPOW_CONFIG.alpha_observation)")
    println("  • Estimate value: $(typeof(POMCPOW_CONFIG.estimate_value))")
    println("  • Tree in info: $(POMCPOW_CONFIG.tree_in_info)")
end

function create_policy(policy_type::String, pomdp::LiPOMDP; alpha::Float64=0.5)
    """Create a policy based on type."""
    
    if policy_type == "POMCPOW"
        solver = POMCPOWSolver(
            #tree_queries = 2500, #give it time to plan
            tree_queries = 2500,
            max_depth = 25, #max time step you can get, planning over POMDP in the future
            criterion = MaxUCB(32.0),#increased exploration constant for high uncertainty deposits
            # Progressive widening (actions) - ENABLED for large action space
            enable_action_pw = true,
            # k_action = 1.2,
            # alpha_action = 0.4,
            # k_observation = 0.8,
            # alpha_observation = 0.20,
           #SECOND using
            k_action = 5.0,  # increased for more action exploration
            alpha_action = 0.5,  # increased for more progressive widening
            # Progressive widening (observations)
            k_observation = 3.0,  # increased for more observation exploration
            alpha_observation = 0.3,  # increased for more observation progressive widening

            # k_action = 1.0,          # Less action exploration
            # alpha_action = 0.25,     # Less progressive widening
            # k_observation = 1.0,     # Less observation exploration
            # alpha_observation = 0.25, # Less observation progressive widening

            # #VSCode params
            # k_action = 0.5,          # Even less action exploration
            # alpha_action = 0.1,      # Much less progressive widening
            # k_observation = 0.5,     # Even less observation exploration
            # alpha_observation = 0.1, # Much less observation progressive wideningng
            #estimate_value = FORollout(OneStepLookaheadPolicy(pomdp, n_rollouts=1, debug=false)),
            estimate_value = FORollout(DynamicProfitMaximizerPolicy(pomdp)),
            #estimate_value = 0,
            tree_in_info = true,
           # rng = MersenneTwister(1234)
        )
        # Store the solver configuration for printing
        global POMCPOW_CONFIG = solver
        return solve(solver, pomdp)
        
    elseif policy_type == "POMCPOW_Depth1"
        # Shallow planning heuristic - only looks one step ahead
        solver = POMCPOWSolver(
            tree_queries = 2500,
            max_depth = 1,           # Only one step ahead
            criterion = MaxUCB(15.0),
            enable_action_pw = true,
            k_action = 5.0,
            alpha_action = 0.5,
            k_observation = 3.0,
            alpha_observation = 0.3,
            estimate_value = FORollout(DynamicProfitMaximizerPolicy(pomdp))
        )
        return solve(solver, pomdp)
        
    elseif policy_type == "ProfitMaximizer" || policy_type == "OneMineProfitMaximization"
        return ProfitMaximizerPolicy(pomdp, top_k=1, times=[1])
        
    elseif policy_type == "EmissionMinimizer" || policy_type == "OneMineEmissionMinimization"
        return EmissionMinimizerPolicy(pomdp, top_k=1, times=[1])
        
    elseif policy_type == "ExploreOnly"
        return ExploreOnlyPolicy(pomdp)
        
    elseif policy_type == "Random" || policy_type == "RandomHeuristicPolicy"
        return RandomHeuristicPolicy(pomdp, debug=false)
        
    elseif policy_type == "DynamicProfitMaximizer"
        return DynamicProfitMaximizerPolicy(pomdp, debug=false)
        
    elseif policy_type == "DynamicEmissionMinimizer"
        return DynamicEmissionMinimizerPolicy(pomdp, debug=false)
        
    elseif policy_type == "OneStepLookahead"
        return OneStepLookaheadPolicy(pomdp, n_rollouts=15, debug=false)
    else
        error("Unknown policy type: $policy_type")
    end
end

function evaluate_policy_simple(planning_pomdp::LiPOMDP, evaluation_pomdp::LiPOMDP, 
                              policy, updater::LiBeliefUpdater, n_sims::Int=50; 
                              rng::AbstractRNG=Random.GLOBAL_RNG)
    """Evaluate a policy with key metrics."""
    
    total_rewards = Float64[]
    total_discounted_rewards = Float64[]
    total_emission_costs = Float64[]
    total_profits = Float64[]
    total_demands = Float64[]
    total_productions = Float64[]
    all_histories = []
    
    # Belief evolution tracking
    belief_evolution_data = Dict()
    
    for i in 1:n_sims
        Random.seed!(i) #control randomness to be deterministic across runs for comparability
        #so initial belief and state are the same across runs -- be a deterministic belief that is still random
        total_reward = 0.0
        total_discounted_reward = 0.0
        total_emission_cost = 0.0
        total_profit = 0.0
        total_demand = 0.0
        total_production = 0.0
        history = []
        
        # Use planning POMDP for initial state and belief
        s0, b0 = create_initial_belief(planning_pomdp)
        b = b0
        s = s0
        

        for t in 1:planning_pomdp.time_horizon
             # --- choose action (+ optional tree dump for POMCPOW only) ---
            if policy isa POMCPOWPlanner
                a, info = action_info(policy, b; tree_in_info=true)

                # if haskey(info, :tree)
                #     dtree = D3Trees.D3Tree(info[:tree])
                #     fname = joinpath(TREE_DIR,
                #                         "gpt_p1_e1_tree_a$(round(planning_pomdp.alpha, digits=2))_t$(t).html")
                #     save_tree_html(fname, dtree)
                # end
            else
                # Heuristics don't support tree_in_info
                a = action(policy, b)
            end

            # Use evaluation POMDP for state transitions and rewards
            sp, o, r = gen(evaluation_pomdp, s, a, rng)

            costs = compute_costs(evaluation_pomdp, s, a)  
            emissions = compute_emissions(evaluation_pomdp, s, a)  
            current_production, _ = calculate_total_production(evaluation_pomdp, s)
            
            revenue = current_production * s.price
            #end
            
            profit = revenue - costs    
            
            total_demand += s.company_demand
            total_production += current_production
            total_profit += profit
            total_emission_cost += emissions
            
            # Track rewards
            total_reward += r  # Keep original reward for comparison
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
            
            # Update belief using planning POMDP's updater
            b = update(updater, b, a, o)
            
            # Track belief evolution for plotting
            if i == 1  # Only track first simulation to avoid data bloat
                if !haskey(belief_evolution_data, "deposits")
                    belief_evolution_data["deposits"] = []
                end
                if !haskey(belief_evolution_data, "price")
                    belief_evolution_data["price"] = []
                end
                if !haskey(belief_evolution_data, "demand")
                    belief_evolution_data["demand"] = []
                end
                
                # Store deposit beliefs
                deposit_beliefs = []
                for site in 1:length(b.deposits_distribution)
                    dist = b.deposits_distribution[site]
                    push!(deposit_beliefs, Dict("μ" => mean(dist), "σ" => std(dist)))
                end
                push!(belief_evolution_data["deposits"], deposit_beliefs)
                
                # Store price and demand beliefs
                push!(belief_evolution_data["price"], Dict("μ" => mean(b.price_distribution), "σ" => std(b.price_distribution)))
                push!(belief_evolution_data["demand"], Dict("μ" => mean(b.demand_distribution), "σ" => std(b.demand_distribution)))
            end
            
            s = sp
        end
        
        # Store results
        push!(total_rewards, total_reward)
        push!(total_discounted_rewards, total_discounted_reward)
        push!(total_emission_costs, total_emission_cost)
        push!(total_profits, total_profit)
        push!(total_demands, total_demand)
        push!(total_productions, total_production)
        push!(all_histories, history)
    end
    
    return Dict(
        "total_reward" => total_rewards,
        "total_discounted_reward" => total_discounted_rewards,
        "total_emission_cost" => total_emission_costs,
        "total_profit" => total_profits,
        "total_demand" => total_demands,
        "total_production" => total_productions,
        "history" => all_histories,
        "belief_evolution" => belief_evolution_data
    )
end

function print_results(all_results::Dict, alphas::Vector{Float64})
    """Print comprehensive results table grouped by alpha."""
    
    println("\n" * "="^120)
    println("POLICY COMPARISON ACROSS ALPHA VALUES")
    println("="^120)
    
    price_model_names = ["Static", "Linear", "Exponential", "State-Dependent", "Dynamic", "Historical"]
    
    for alpha in alphas
        println("\n" * "="^120)
        println("ALPHA = $alpha (Profit weight: $(round(alpha, digits=2)), Emission weight: $(round(1-alpha, digits=2)))")
        println("="^120)
        
        # Filter results for this alpha
        alpha_results = Dict(k => v for (k, v) in all_results if v["alpha"] == alpha)
        
        if isempty(alpha_results)
            println("No results for alpha = $alpha")
            continue
        end
        
        println("\n📊 PERFORMANCE METRICS")
        println("-"^120)
        println("Policy Type          | Planning | Eval | Total Profit | Emission Cost | Disc. Reward | Met Demand %")
        println("-"^120)
        
        # Sort by policy type for consistent ordering
        sorted_results = sort(collect(alpha_results), by = x -> x[2]["policy_type"])
        
        for (policy_name, results) in sorted_results
            policy_type = results["policy_type"]
            planning_model = results["planning_price_model"]
            eval_model = results["evaluation_price_model"]
            mean_profit = mean(results["total_profit"])
            mean_emission_cost = mean(results["total_emission_cost"])
            mean_disc_reward = mean(results["total_discounted_reward"])
            
            # Calculate demand meeting percentage
            mean_demand = mean(results["total_demand"])
            mean_production = mean(results["total_production"])
            met_demand_pct = mean_demand > 0 ? (mean_production / mean_demand) * 100 : 0.0
            
            planning_name = planning_model <= length(price_model_names) ? price_model_names[planning_model] : "Unknown"
            eval_name = eval_model <= length(price_model_names) ? price_model_names[eval_model] : "Unknown"
            
            println("$(rpad(policy_type, 20)) | $(lpad(planning_name[1], 8)) | $(lpad(eval_name[1], 4)) | " *
                    "$(lpad(round(mean_profit/1e6, digits=2), 11))M | " *
                    "$(lpad(round(mean_emission_cost/1e6, digits=2), 13))M | " *
                    "$(lpad(round(mean_disc_reward, digits=4), 19)) | " *
                    "$(lpad(round(met_demand_pct, digits=1), 10))%")
         end
        
        # Best performers for this alpha
        println("\n🏆 BEST PERFORMERS (α=$alpha)")
        println("-"^120)
        
        if !isempty(alpha_results)
            # Best by total profit
            best_profit_idx = argmax([mean(r["total_profit"]) for r in values(alpha_results)])
            best_profit_name = collect(keys(alpha_results))[best_profit_idx]
            best_profit_value = mean(alpha_results[best_profit_name]["total_profit"])
            println("Best Profit: $(alpha_results[best_profit_name]["policy_type"]) (\$$(round(best_profit_value/1e6, digits=2))M)")
            
            # Best by emission (lowest)
            best_emission_idx = argmin([mean(r["total_emission_cost"]) for r in values(alpha_results)])
            best_emission_name = collect(keys(alpha_results))[best_emission_idx]
            best_emission_value = mean(alpha_results[best_emission_name]["total_emission_cost"])
            println("Lowest Emissions: $(alpha_results[best_emission_name]["policy_type"]) (\$$(round(best_emission_value/1e6, digits=2))M)")
            
            # Best by combined objective (using alpha weighting)
            best_combined_idx = argmax([mean(r["total_discounted_reward"]) for r in values(alpha_results)])
            best_combined_name = collect(keys(alpha_results))[best_combined_idx]
            best_combined_value = mean(alpha_results[best_combined_name]["total_discounted_reward"])
            println("Best Combined (Discounted Reward): $(alpha_results[best_combined_name]["policy_type"]) (\$$(round(best_combined_value, digits=2))M)")
        end
    end
    
    println("\n" * "="^120)
end


function run_simple_policy_test(
    policies::Vector{String}, 
    planning_price_models::Vector{Int64},
    evaluation_price_models::Vector{Int64}, 
    alphas::Vector{Float64};
    time_horizon::Int64 = 29,
    n_sims::Int64 = 30, 
    output_file::String = "results.txt"
)
    """Run the simple policy testing."""
    
    println("🚀 SIMPLE POLICY TESTING")
    println("="^80)
    println("Emissions Scaled to 1, ExplorePolicy")
    println("Testing POMCPOW vs ProfitMaximizer vs EmissionMinimizer vs ExploreOnly vs Random vs DynamicProfitMaximizer vs DynamicEmissionMinimizer")
    println("Different price models for planning vs evaluation")
    println("Using ALPHA values for multi-objective optimization")
    println("="^80)
    
    # Configuration
    n_sims = n_sims
    time_horizon = time_horizon
    
    println("Configuration:")
    println("  • Simulations per policy: $n_sims")
    println("  • Time horizon: $time_horizon")
    println("  • Planning price models: $planning_price_models")
    println("  • Evaluation price models: $evaluation_price_models")
    println("  • Policies: $policies")
    println("  • Alpha values: $alphas")
    
    # Print POMCPOW configuration if POMCPOW is in the policies
    if "POMCPOW" in policies
        # Create a temporary POMDP and policy to initialize the configuration
        temp_pomdp = LiPOMDP(price_model_type=planning_price_models[1], time_horizon=time_horizon, alpha=alphas[1])
        temp_policy = create_policy("POMCPOW", temp_pomdp, alpha=alphas[1])
        print_pomcpow_config()  # Commented out to reduce output
    end
    
    # Store all results
    all_results = Dict{String, Dict{String, Any}}()
    
    # Create directory for output files if it doesn't exist
    output_dir = dirname(output_file)
    if !isdir(output_dir) && output_dir != ""
        mkpath(output_dir)
    end
    
    # Test each policy configuration
    total_configs = length(policies) * length(planning_price_models) * length(evaluation_price_models) * length(alphas)
    # Store Pareto data for plotting
    pareto_data = Dict{Float64, Vector{Dict{String, Any}}}()
    for alpha in alphas
        pareto_data[alpha] = []
    end
    
    config_count = 0
    
    # Test with alpha values
    for alpha in alphas
        println("\n" * "="^80)
        println("Testing with ALPHA = $alpha (All policies adapt to alpha)")
        println("="^80)
        
        for policy_type in policies
            for planning_model in planning_price_models
                for eval_model in evaluation_price_models
                    config_count += 1
                    policy_name = "$(policy_type)_P$(planning_model)_E$(eval_model)_A$(alpha)"
                    
                    println("\n[$(config_count)/$(total_configs)] Testing $(policy_type) with α=$alpha...")
                    println("  Planning model: Type $planning_model, Evaluation model: Type $eval_model")
                    
                    # Create planning and evaluation POMDPs
                    # All policies use the same alpha for both planning and evaluation
                    planning_pomdp = LiPOMDP(price_model_type=planning_model, time_horizon=time_horizon, alpha=alpha)
                    evaluation_pomdp = LiPOMDP(price_model_type=eval_model, time_horizon=time_horizon, alpha=alpha)
                    
                    # Create belief updater
                    updater = LiBeliefUpdater(P=evaluation_pomdp) #TODO is it supposed to be evaluation_pomdp? or planning_pomdp??
                    
                    # Create policy
                    policy = create_policy(policy_type, planning_pomdp, alpha=alpha)
                    
                    # Evaluate policy
                    results = evaluate_policy_simple(planning_pomdp, evaluation_pomdp, policy, updater, n_sims)
                    
                    # Calculate statistics
                    mean_profit = mean(results["total_profit"])
                    std_profit = std(results["total_profit"])
                    mean_emission_cost = mean(results["total_emission_cost"])
                    std_emission_cost = std(results["total_emission_cost"])
                    mean_disc_reward = mean(results["total_discounted_reward"])
                    
                    # Store results with metadata
                    all_results[policy_name] = merge(results, Dict(
                        "planning_price_model" => planning_model,
                        "evaluation_price_model" => eval_model,
                        "policy_type" => policy_type,
                        "alpha" => alpha,
                        "mean_profit" => mean_profit,
                        "std_profit" => std_profit,
                        "mean_discounted_reward" => mean_disc_reward,
                        "mean_emission_cost" => mean_emission_cost,
                        "std_emission_cost" => std_emission_cost,
                        "history" => results["history"]
                    ))
                    
                    # Add to Pareto data
                    push!(pareto_data[alpha], Dict(
                        "policy" => policy_type,
                        "profit" => mean_profit,
                        "profit_std" => std_profit,
                        "emissions" => mean_emission_cost,
                        "emissions_std" => std_emission_cost,
                        "discounted_reward" => mean_disc_reward,
                        "planning_model" => planning_model,
                        "eval_model" => eval_model
                    ))
                    
                    # Print quick summary
                    println("  ✓ Profit: \$$(round(mean_profit/1e6, digits=2))M ± $(round(std_profit/1e6, digits=2))M")
                    println("  ✓ Emissions: \$$(round(mean_emission_cost/1e6, digits=2))M ± $(round(std_emission_cost/1e6, digits=2))M")
                    println("  ✓ Disc. Reward: $(round(mean_disc_reward, digits=4))M")  
                    end
            end
        end
    end
    
    # Print comprehensive results
    print_results(all_results, alphas)
    
    # Save all results in main file
    println("Saving results to: $output_file")
    open(output_file, "w") do f
        write(f, JSON.json(all_results, 2))
    end
    println("Results saved successfully!")
    
    # Create summary CSV for easy analysis
    summary_file = replace(output_file, ".txt" => "_summary.csv")
    println("Saving summary to: $summary_file")
    open(summary_file, "w") do f
        println(f, "Alpha,Policy,PlanningModel,EvalModel,MeanProfit,StdProfit,MeanEmissions,StdEmissions,DiscountedReward")
        for (_, results) in all_results
            println(f, "$(results["alpha"]),$(results["policy_type"]),$(results["planning_price_model"]),$(results["evaluation_price_model"]),$(results["mean_profit"]),$(results["std_profit"]),$(results["mean_emission_cost"]),$(results["std_emission_cost"]),$(results["mean_discounted_reward"])")
        end
    end
    println("Summary saved successfully!")
    
    println("\n✅ Policy testing complete!")
    println("📊 Results saved to:")
    println("   - Main results: $(output_file)")
    println("   - Summary CSV: $(summary_file)")
    println("   - Individual alpha results in: $(dirname(output_file))")
    
    # Print POMCPOW configuration again at the end for reference
    if "POMCPOW" in policies
        print_pomcpow_config()  # Commented out to reduce output
    end
    
    return all_results, pareto_data
end

# Configuration
time_horizon = 29
n_sims = 10
planning_price_models = Int64[1]  # Exponential model for planning
evaluation_price_models = Int64[6]  # Static model for evaluation
policies = String[
    "POMCPOW",
    "OneStepLookahead",
]

# policies = String[
#     "POMCPOW",
#     "POMCPOW_Depth1",
#     "OneStepLookahead",
#     "DynamicProfitMaximizer",
#     "DynamicEmissionMinimizer",
#     "OneMineProfitMaximization",
#     "OneMineEmissionMinimization",
#     "ExploreOnly",
#     "RandomHeuristicPolicy"
# ]
alphas = Float64[0.25, 0.5, 0.75, 1.0]  # Range from pure profit (0) to pure emission minimization (1)
# Create dynamic output filename based on planning and evaluation models
planning_str = join(planning_price_models, "_")
evaluation_str = join(evaluation_price_models, "_")
output_file = "outputs/results v/policy_test_results_p$(planning_str)_e$(evaluation_str)_2Brewards_2.5k_32UCB_dynamic_rollout_depth25_10sims.txt"


results, pareto_data = run_simple_policy_test(
                policies, planning_price_models, evaluation_price_models, alphas,
                time_horizon=time_horizon, n_sims=n_sims, output_file=output_file)
 


# Function to show POMCPOW actions for different alpha values
function show_all_policies_actions_by_alpha(planning_models=[1], evaluation_models=[1], alpha_values=[0.5], policy_names=["POMCPOW"], time_horizon_val=29)
    println("\n🎯 ALL POLICIES ACTIONS BY ALPHA VALUES")
    println("=" ^ 80)
    
    # Set random seed for reproducibility
    Random.seed!(1)
    
    # Price model names for display
    price_model_names = ["Static", "Linear", "Exponential", "Dynamic", "State-Dependent", "Historical"]
    
    for alpha in alpha_values
        println("\n🔍 ALPHA = $alpha")
        println("=" ^ 60)
        
        for planning_model in planning_models
            for evaluation_model in evaluation_models
                println("\n📊 PLANNING MODEL: $(price_model_names[planning_model]) (Type $planning_model)")
                println("📊 EVALUATION MODEL: $(price_model_names[evaluation_model]) (Type $evaluation_model)")
                println("-" ^ 60)
                
                # Create planning POMDP
                planning_pomdp = LiPOMDP(
                    price_model_type = planning_model,
                    time_horizon = time_horizon_val,
                    alpha = alpha
                )
                
                # Create evaluation POMDP
                evaluation_pomdp = LiPOMDP(
                    price_model_type = evaluation_model,
                    time_horizon = time_horizon_val,
                    alpha = alpha
                )
                
                # Create updater for evaluation
                updater = LiBeliefUpdater(P=evaluation_pomdp)
                
                for policy_name in policy_names
                    println("\n📋 POLICY: $policy_name")
                    println("-" ^ 40)
                    
                    # Create policy using planning POMDP
                    policy = create_policy(policy_name, planning_pomdp, alpha=alpha)
                    
                    # Run one simulation using evaluation POMDP
                    s0 = rand(initialstate(evaluation_pomdp))
                    s0_init, b0 = create_initial_belief(evaluation_pomdp)
                    
                    state = s0
                    belief = b0
                    total_reward = 0.0
                    
                    println("Step | Action | Reward | Discounted Reward | True Deposits")
                    println("-" ^ 60)
                    
                    total_discounted_reward = 0.0
                    for step in 1:time_horizon_val  # Show all steps
                        # Get action from policy
                        action = POMDPs.action(policy, belief)
                        
                        # Step environment using evaluation POMDP
                        sp, o, r = gen(evaluation_pomdp, state, action, Random.GLOBAL_RNG)
                        
                        # Update belief
                        belief = POMDPs.update(updater, belief, action, o)
                        
                        # Convert action to string
                        action_str = action_to_string(action)
                        total_reward += r
                        total_discounted_reward += evaluation_pomdp.γ^(step-1) * r
                        
                        # Format deposits for display
                        deposits_str = "[$(round(state.deposit[1], digits=0)), $(round(state.deposit[2], digits=0)), $(round(state.deposit[3], digits=0)), $(round(state.deposit[4], digits=0))]"
                        
                        println("$step   | $action_str | $(round(r, digits=2)) | $(round(total_discounted_reward, digits=4)) | $deposits_str")
                        
                        state = sp
                    end
                    
                    # Show final results
                    println("\n📊 FINAL RESULTS $policy_name (α=$alpha):")
                    println("  Planning Model: Type $planning_model, Evaluation Model: Type $evaluation_model")
                    println("  Discounted Reward: $(round(total_discounted_reward, digits=4))")
                    println("  Final Price: $(round(state.price, digits=2))")
                    println("  Final Demand: $(round(state.company_demand, digits=2))")
                    println("  Sites Opened: $(sum(state.have_opened))")
                    println("  Total Mined: $(round(state.total_mined, digits=2))")
                end
            end
        end
    end
    
    println("\n" * "=" ^ 80)
end


# Run the ALL POLICIES actions analysis - using same models as main configuration
#show_all_policies_actions_by_alpha(planning_price_models, evaluation_price_models, alphas, policies, time_horizon)

println("="^80)


# planning_pomdp = LiPOMDP(price_model_type=1, time_horizon=25, alpha=0.5)

# actions(planning_pomdp)
# s0, b0 = create_initial_belief(planning_pomdp)
# actions(planning_pomdp, b0)
# a0 = rand(actions(planning_pomdp, b0))
# rng = MersenneTwister(1234)
# s1, o1, r1 = gen(planning_pomdp, s0, a0, rng)
# o1
# b1 = update(LiBeliefUpdater(planning_pomdp), b0, a0, o1)
# actions(planning_pomdp, b1)
# a1 = rand(actions(planning_pomdp, b1))




