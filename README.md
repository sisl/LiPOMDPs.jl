# LiPOMDPs.jl

[![CI](https://github.com/sisl/LiPOMDPs.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/sisl/LiPOMDPs.jl/actions/workflows/CI.yml)

A Julia package for modeling lithium production decisions as a Partially Observable Markov Decision Process (POMDP). A firm decides, year by year from 1995 to 2025, which of four stylized deposits to explore, open or restore, two using direct lithium extraction (DLE) and two mineral ore, while deposit size, recovery, price and demand are uncertain. Emissions are priced at the social cost of carbon, and a single parameter α sets the weight on profit versus emissions.

This repository accompanies the paper *Optimizing Lithium Production Decisions under Geological, Demand, and Pricing Uncertainties: A POMDP Framework for Multi-Objective Decision Making*.

## Features

- **Multi-site decisions**: four deposits (sites 1–2 DLE, sites 3–4 mineral ore) acted on jointly each year, under a shared capital budget
- **Automatic production**: an open mine produces up to its capacity and the firm's remaining demand, one year after opening
- **Hidden geology**: deposit size and recovery are unobserved; exploration and production give noisy observations of them
- **Price models**: static, linear, exponential, geometric Brownian motion and historical (1995–2025), fitted to the historical data
- **Emissions**: production emissions per tonne LCE for each technology, monetized at $190/t CO₂
- **Planner and baselines**: POMCPOW compared with rule-based heuristics, a one-step lookahead and a timing-aware optimal-stopping policy

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/sisl/LiPOMDPs.jl")
```

For development:
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Quick Start

```julia
using LiPOMDPs
using POMDPs
using POMCPOW

# Create POMDP instance (defaults: static prices, 31-year horizon 1995–2025, alpha = 0.5)
pomdp = LiPOMDP(
    price_model_type=4,   # 1 static, 2 linear, 3 exponential, 4 GBM, 6 historical
    alpha=0.5             # weight on profit; 1 - alpha weights monetized emissions
)

# Initial state and belief
s0, b0 = create_initial_belief(pomdp)
updater = LiBeliefUpdater(P=pomdp)

# POMCPOW in the configuration used in the paper
solver = POMCPOWSolver(
    tree_queries=10_000,
    max_depth=25,
    criterion=MaxUCB(4.0),
    enable_action_pw=true, k_action=5.0, alpha_action=0.5,
    k_observation=3.0, alpha_observation=0.3,
    estimate_value=FORollout(TimingAwarePolicy(pomdp)),
)
policy = solve(solver, pomdp)
a = action(policy, b0)
```

## State and Action Structure

### State
```julia
@with_kw mutable struct State
    deposit::Vector{Float64}      # in-situ Li at each site (t Li)
    time::Float64                 # current time step (year, 1 = start_year)
    price::Float64                # market price ($/t LCE)
    company_demand::Float64       # company demand (t LCE / year)
    total_mined::Float64          # cumulative sold LCE (t)
    have_opened::Vector{Bool}     # which sites have been opened
    is_depleted::Vector{Bool}     # which mines are depleted
    restored_mines::Vector{Bool}  # which depleted mines have been restored
    recovery::Vector{Float64}     # true recovery per site (hidden)
end
```

### Action
```julia
@with_kw mutable struct Action
    site::Vector{Int64}   # site identifiers
    type::Vector{Int64}   # per site: 0 = do nothing, 1 = explore, 2 = open, 3 = restore
end
```

Open mines produce automatically; there is no operate action. A depleted mine can be restored (type 3).

## Policies

- **ProfitMaximizerPolicy** / **EmissionMinimizerPolicy**: open the best site by lifetime profit / emissions at the first time step only
- **DynamicProfitMaximizerPolicy** / **DynamicEmissionMinimizerPolicy**: re-evaluate every step
- **ExploreOnlyPolicy**: explores every unopened site, never opens
- **RandomHeuristicPolicy**: random baseline
- **TimingAwarePolicy**: optimal stopping on the planning model's expected price and demand path; POMCPOW's rollout policy
- **OneStepLookaheadPolicy**: evaluates every feasible action with rollouts of a base policy (TimingAware or AlphaAwareRolloutPolicy)
- **AlphaAwareRolloutPolicy**: DynamicProfitMaximizer ranked by the α-weighted reward

## Reproducing the paper

All scripts are run from the repository root with `julia --project=.`; the evaluation scripts are multithreaded (`-t <threads>`).

| Script | Produces |
|---|---|
| `scripts/final_results_best_pomcpow.jl <planning_model> [alphas]` | Policy comparison tables (train 25 / test 50 scenarios), planned under one price model and evaluated on historical prices → `outputs/test_train_final_results/` |
| `scripts/sensitivity_analysis.jl <n_seeds> <alpha> <planning_model> <queries> [core\|appendix] [uncert]` | One-at-a-time sensitivity tables → `outputs/sensitivity/` |
| `scripts/plot_belief_uncertainty.jl` | Belief-uncertainty-over-time figure |
| `scripts/policy_belief_action_timeline.jl [planning_models]` | Belief and action timeline figure → `outputs/belief_timeline/` |
| `scripts/fit_gbm.jl` | GBM drift and volatility fitted to the historical prices |

`scripts/pomcpow_tuning_harness.jl`, `pomcpow_components.jl` and `pomcpow_exploration_controls.jl` are shared helpers (paired-seed evaluation and POMCPOW construction) included by the scripts above.

`outputs/` holds the result files behind the paper's tables and figures: the final policy-comparison CSVs, the sensitivity CSVs and their LaTeX tables, the Pareto-curve TikZ source, and the belief/action timeline figures.

## Pricing Models

- **1 static**: constant price with 10% noise
- **2 linear**: linear trend with price-proportional noise
- **3 exponential**: exponential growth with price-proportional noise
- **4 GBM**: geometric Brownian motion, drift and volatility fitted to 1995–2025 log-returns
- **6 historical**: the observed 1995–2025 price series
