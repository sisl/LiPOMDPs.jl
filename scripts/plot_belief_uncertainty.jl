#!/usr/bin/env julia

"""
Average deposit-belief uncertainty over time for POMCPOW, the one-mine profit maximizer and
ExploreOnly (paper Figure 2).

Same setup as scripts/final_results_best_pomcpow.jl: policies plan under one price model and are
evaluated on historical prices (model 6) with the test seeds (2001, 2002, ...) and common random
numbers, so every policy faces the same scenarios. Uncertainty is the mean, over the four sites, of
the standard deviation of the deposit belief (t Li); the plot shows mean ± std over the scenarios.

Usage: julia --project=. scripts/plot_belief_uncertainty.jl [planning_model] [n_sims]
  planning_model  1 static, 2 linear, 3 exponential, 4 GBM, 6 historical (default 2)
  n_sims          number of evaluation scenarios (default 30)
Writes outputs/belief_uncertainty/belief_uncertainty_comparison_p<planning>_e6.tex (and .pdf when
pdflatex is available).
"""

using LiPOMDPs
using Random
using Statistics
using POMDPs
using POMDPTools
using POMCPOW
using POMCPOW: POMCPOWPlanner
using PGFPlots

const PLANNING_PRICE_MODEL = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2
const N_SIMS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 30
const EVALUATION_PRICE_MODEL = 6
const ALPHA = 0.5
const TIME_HORIZON = LiPOMDP().time_horizon   # model default (1995–2025)
const OUTPUT_DIR = "outputs/belief_uncertainty"
const POLICIES = ["POMCPOW", "OneMineProfitMaximization", "ExploreOnly"]

# =============================================================================
# POLICIES
# =============================================================================

# Same POMCPOW configuration as scripts/final_results_best_pomcpow.jl
function create_policy(policy_type::String, pomdp::LiPOMDP)
    if policy_type == "POMCPOW"
        solver = POMCPOWSolver(
            tree_queries = 10_000,
            max_depth = 25,
            criterion = MaxUCB(4.0),
            enable_action_pw = true,
            k_action = 5.0,
            alpha_action = 0.5,
            k_observation = 3.0,
            alpha_observation = 0.3,
            estimate_value = FORollout(TimingAwarePolicy(pomdp)),
        )
        return solve(solver, pomdp)
    elseif policy_type == "OneMineProfitMaximization"
        return ProfitMaximizerPolicy(pomdp, top_k=1, times=[1])
    elseif policy_type == "ExploreOnly"
        return ExploreOnlyPolicy(pomdp)
    else
        error("Unknown policy type: $policy_type")
    end
end

# =============================================================================
# BELIEF UNCERTAINTY
# =============================================================================

# Mean over sites of the std of the deposit belief (t Li)
belief_uncertainty(b::LiBelief) = mean(std(d) for d in b.deposits_distribution)

"""
Run `n_sims` evaluation scenarios and record the belief uncertainty at every time step.
Returns a (time_horizon × n_sims) matrix.
"""
function collect_belief_uncertainty(planning_pomdp::LiPOMDP, evaluation_pomdp::LiPOMDP,
                                    policy, updater::LiBeliefUpdater, n_sims::Int)
    T = planning_pomdp.time_horizon
    uncertainty = zeros(T, n_sims)

    for sim in 1:n_sims
        seed = sim + 2000   # test seeds, as in final_results_best_pomcpow.jl
        Random.seed!(seed)
        s0, b0 = create_initial_belief(planning_pomdp)
        b, s = b0, s0

        for t in 1:T
            uncertainty[t, sim] = belief_uncertainty(b)

            a = policy isa POMCPOWPlanner ? first(action_info(policy, b; tree_in_info=false)) :
                                            action(policy, b)

            # Common random numbers: environment noise depends only on (seed, t)
            Random.seed!(hash((seed, t)))
            env_rng = MersenneTwister(hash((seed, t, :env)))
            sp, o, _ = gen(evaluation_pomdp, s, a, env_rng)

            if isterminal(evaluation_pomdp, sp)
                uncertainty[(t+1):T, sim] .= uncertainty[t, sim]
                break
            end

            b = update(updater, b, a, o)
            s = sp
        end
    end

    return uncertainty
end

function analyze_belief_uncertainty(policies::Vector{String}; planning_model::Int, n_sims::Int)
    println("BELIEF UNCERTAINTY ANALYSIS")
    println("="^60)
    println("Planning price model: $planning_model   Evaluation price model: $EVALUATION_PRICE_MODEL")
    println("Alpha: $ALPHA   Time horizon: $TIME_HORIZON   Scenarios per policy: $n_sims")
    println()

    planning_pomdp = LiPOMDP(price_model_type=planning_model, time_horizon=TIME_HORIZON, alpha=ALPHA)
    evaluation_pomdp = LiPOMDP(price_model_type=EVALUATION_PRICE_MODEL, time_horizon=TIME_HORIZON, alpha=ALPHA)
    updater = LiBeliefUpdater(P=planning_pomdp)

    results = Dict{String, Dict}()
    for policy_type in policies
        println("Analyzing policy: $policy_type")
        policy = create_policy(policy_type, planning_pomdp)
        uncertainty = collect_belief_uncertainty(planning_pomdp, evaluation_pomdp, policy, updater, n_sims)
        results[policy_type] = Dict(
            "mean" => vec(mean(uncertainty, dims=2)),
            "std" => vec(std(uncertainty, dims=2)),
            "raw" => uncertainty,
        )
        println("  final uncertainty: $(round(results[policy_type]["mean"][end], digits=2)) ± $(round(results[policy_type]["std"][end], digits=2)) t Li")
    end
    return results
end

# =============================================================================
# PLOT
# =============================================================================

function write_standalone_tex(tikz_code::String, tex_file::String)
    open(tex_file, "w") do f
        write(f, """
\\documentclass[tikz]{standalone}
\\usepackage{pgfplots}
\\usepackage{amsmath}
\\pgfplotsset{compat=1.16}

\\begin{document}
$tikz_code
\\end{document}
""")
    end
end

function plot_belief_uncertainty(results::Dict, output_file::String)
    policy_styles = Dict(
        "POMCPOW" => ("blue", "solid", "*", "POMCPOW"),
        "OneMineProfitMaximization" => ("red", "dashed", "square*", "One Mine Profit Max"),
        "ExploreOnly" => ("green!70!black", "dotted", "triangle*", "Explore Only"),
    )

    ax = Axis(
        xlabel = "Time Step",
        ylabel = "Avg. Deposit Belief Uncertainty (tonnes Li)",
        title = "Belief Uncertainty Evolution by Policy",
        legendPos = "north east",
        width = "14cm",
        height = "9cm",
        style = "grid=major, grid style={line width=.2pt, draw=gray!20}, legend style={font=\\small}",
    )

    for policy_name in POLICIES
        haskey(results, policy_name) || continue
        mean_vals, std_vals = results[policy_name]["mean"], results[policy_name]["std"]
        color, line_style, marker, display_name = policy_styles[policy_name]
        push!(ax, Plots.Linear(
            collect(1:length(mean_vals)), mean_vals,
            errorBars = ErrorBars(; y = std_vals),
            style = "$(color), thick, $(line_style), mark=$(marker), mark size=2.5pt, error bars/.cd, y dir=both, y explicit",
            legendentry = display_name,
        ))
    end

    mkpath(dirname(output_file))
    tex_file = replace(output_file, ".pdf" => ".tex")
    write_standalone_tex(string(ax), tex_file)
    println("TEX saved to: $tex_file")

    if Sys.which("pdflatex") === nothing
        println("pdflatex not found; compile $tex_file manually to get the PDF")
        return ax
    end
    cd(dirname(abspath(output_file))) do
        run(pipeline(`pdflatex -interaction=nonstopmode $(basename(tex_file))`, devnull))
        for ext in (".aux", ".log", ".out")
            aux = replace(basename(output_file), ".pdf" => ext)
            isfile(aux) && rm(aux)
        end
    end
    println("PDF saved to: $output_file")
    return ax
end

# =============================================================================
# MAIN
# =============================================================================

results = analyze_belief_uncertainty(POLICIES; planning_model=PLANNING_PRICE_MODEL, n_sims=N_SIMS)

output_plot = joinpath(OUTPUT_DIR, "belief_uncertainty_comparison_p$(PLANNING_PRICE_MODEL)_e$(EVALUATION_PRICE_MODEL).pdf")
plot_belief_uncertainty(results, output_plot)

println("\n" * "="^60)
for policy in POLICIES
    data = results[policy]
    println("$policy: initial $(round(data["mean"][1], digits=2)) ± $(round(data["std"][1], digits=2)), ",
            "final $(round(data["mean"][end], digits=2)) ± $(round(data["std"][end], digits=2)) t Li")
end
