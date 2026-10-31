module LiPOMDPs

using Parameters
using POMDPs
using POMDPTools
using Distributions
using Random
using Base: rand

include("historical_data.jl")

export State, Action, Observation, LiPOMDP, LiBelief, LiBeliefUpdater
include("model.jl")

export get_historical_price, 
         get_historical_production,
         get_historical_demand,
         get_historical_data
include("utils.jl")

export reward, transition, observation, update, gen, initialstate, isterminal, discount, actions, actionindex, 
initialobs, obs_weight, create_initial_belief, compute_costs, compute_emissions, calculate_total_production, expected_profit_site, expected_emission_site,
mean_transition, expected_price, expected_demand, mean_path_value
include("pomdp.jl")

export ProfitMaximizerPolicy, EmissionMinimizerPolicy, ExploreOnlyPolicy, RandomHeuristicPolicy, DynamicProfitMaximizerPolicy, DynamicEmissionMinimizerPolicy, OneStepLookaheadPolicy, AlphaAwareRolloutPolicy, TimingAwarePolicy
include("policies.jl")

end
