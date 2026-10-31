#!/usr/bin/env julia

"""
Deposit-belief uncertainty over time, per site, for POMCPOW, the one-mine profit maximizer and
ExploreOnly (paper Figure 2: a 2×2 panel, one per site).

Same setup as scripts/final_results_best_pomcpow.jl: policies plan under one price model and are
evaluated on historical prices (model 6) with the test seeds (2001, 2002, ...) and common random
numbers, so every policy faces the same scenarios. Uncertainty is the standard deviation of each
site's deposit belief (t Li); the plot shows mean ± std over the scenarios.

Usage: julia --project=. -t <threads> scripts/plot_belief_uncertainty.jl [planning_model] [n_sims]
  planning_model  1 static, 2 linear, 3 exponential, 4 GBM, 6 historical (default 2)
  n_sims          number of evaluation scenarios (default 30)
Writes outputs/belief_uncertainty/belief_uncertainty_p<planning>_e6.tex (the paper's figure
environment, a 2×2 pgfplots groupplot with the data as coordinates), .png (preview rendered with
Plots/GR) and .csv (mean/std per policy, site and step).
"""

using LiPOMDPs
using Random
using Statistics
using POMDPs
using POMDPTools
using POMCPOW
using POMCPOW: POMCPOWPlanner
import Plots

const PLANNING_PRICE_MODEL = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 2
const N_SIMS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 30
const EVALUATION_PRICE_MODEL = 6
const ALPHA = 0.5
const TIME_HORIZON = LiPOMDP().time_horizon   # model default (1995–2025)
const OUTPUT_DIR = "outputs/belief_uncertainty"
const POLICIES = ["POMCPOW", "OneMineProfitMaximization", "ExploreOnly"]
const SITE_NAMES = ["Site 1 (DLE)", "Site 2 (DLE)", "Site 3 (Mineral Ore)", "Site 4 (Mineral Ore)"]

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

# Std of each site's deposit belief (t Li)
belief_uncertainty(b::LiBelief) = [std(d) for d in b.deposits_distribution]

"""
Run `n_sims` evaluation scenarios and record each site's belief uncertainty at every time step.
Scenarios run in parallel over threads (`julia -t`); each builds its own policy, since a POMCPOW
planner holds its search tree. Returns a (time_horizon × n_sims × n_sites) array.
"""
function collect_belief_uncertainty(planning_pomdp::LiPOMDP, evaluation_pomdp::LiPOMDP,
                                    policy_type::String, updater::LiBeliefUpdater, n_sims::Int)
    T = planning_pomdp.time_horizon
    uncertainty = zeros(T, n_sims, planning_pomdp.n_deposits)

    Threads.@threads :greedy for sim in 1:n_sims
        policy = create_policy(policy_type, planning_pomdp)
        seed = sim + 2000   # test seeds, as in final_results_best_pomcpow.jl
        Random.seed!(seed)
        s0, b0 = create_initial_belief(planning_pomdp)
        b, s = b0, s0

        for t in 1:T
            uncertainty[t, sim, :] = belief_uncertainty(b)

            a = policy isa POMCPOWPlanner ? first(action_info(policy, b; tree_in_info=false)) :
                                            action(policy, b)

            # Common random numbers: environment noise depends only on (seed, t)
            Random.seed!(hash((seed, t)))
            env_rng = MersenneTwister(hash((seed, t, :env)))
            sp, o, _ = gen(evaluation_pomdp, s, a, env_rng)

            if isterminal(evaluation_pomdp, sp)
                uncertainty[(t+1):T, sim, :] .= uncertainty[t:t, sim, :]
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
        uncertainty = collect_belief_uncertainty(planning_pomdp, evaluation_pomdp, policy_type, updater, n_sims)
        results[policy_type] = Dict(
            "mean" => dropdims(mean(uncertainty, dims=2), dims=2),   # T × n_sites
            "std" => dropdims(std(uncertainty, dims=2), dims=2),
            "raw" => uncertainty,
        )
        final = round.(results[policy_type]["mean"][end, :], digits=0)
        println("  final uncertainty per site (t Li): $final")
    end
    return results
end

# =============================================================================
# OUTPUT
# =============================================================================

const POLICY_STYLES = Dict(
    "POMCPOW" => (tikz="pomstyle", label="POMCPOW", gr=(:royalblue, :solid, :circle)),
    "OneMineProfitMaximization" => (tikz="profstyle", label="Profit Maximizer", gr=(:indianred, :dash, :rect)),
    "ExploreOnly" => (tikz="explstyle", label="Explore Only", gr=(:seagreen, :dot, :utriangle)),
)

coords(m, sd) = join(("($(t-1), $(round(m[t], digits=2))) +- (0, $(round(sd[t], digits=2)))" for t in eachindex(m)), " ")

"""
LaTeX figure for the paper: 2×2 groupplot, one panel per site, with the mean ± std written as
coordinates. The x axis is the number of steps taken (0 = initial belief). Requires in the preamble
\\usepackage{pgfplots}, \\usepgfplotslibrary{groupplots} and the colors lightblue, lightred, lightgreen.
"""
function write_figure_tex(results::Dict, tex_file::String)
    T = size(results[POLICIES[1]]["mean"], 1)
    io = IOBuffer()
    print(io, """
% Generated by scripts/plot_belief_uncertainty.jl
% Preamble requirements:
% \\usepackage{pgfplots}
% \\pgfplotsset{compat=1.18}
% \\usepgfplotslibrary{groupplots}
% \\definecolor{lightblue}{RGB}{100,149,237} \\definecolor{lightred}{RGB}{240,128,128} \\definecolor{lightgreen}{RGB}{119,190,119}

\\begin{figure}[H]
\\centering

\\pgfplotsset{
  every axis/.append style={
    tick label style={font=\\scriptsize},
    label style={font=\\small},
    title style={font=\\small, yshift=3pt},
    xlabel near ticks,
    ylabel near ticks,
    xlabel={Time Step},
    ylabel={Belief Uncertainty (tonnes Li)},
    xtick align=outside,
    ytick align=outside,
    grid=major,
    grid style={line width=.2pt, draw=gray!20},
    enlarge y limits=0.03,
    scaled y ticks=false,
    yticklabel style={/pgf/number format/fixed},
  },
  pomstyle/.style={lightblue,  thick,  mark=*,        mark size=1.2pt, error bars/.cd, y dir=both, y explicit},
  profstyle/.style={lightred,   thick,  dashed,        mark=square*,   mark size=1.2pt, error bars/.cd, y dir=both, y explicit},
  explstyle/.style={lightgreen, thick,  dotted,        mark=triangle*, mark size=1.4pt, error bars/.cd, y dir=both, y explicit}
}

\\begin{tikzpicture}
\\begin{groupplot}[
  group style={
    group size=2 by 2,
    horizontal sep=14mm,
    vertical sep=12mm,
    x descriptions at=edge bottom,
    y descriptions at=edge left
  },
  width=0.46\\textwidth,
  height=5.2cm,
  xmin=0, xmax=$(T-1)
]
""")
    for site in 1:4
        if site == 1
            println(io, "\n% ===== Site 1 =====")
            println(io, "\\nextgroupplot[title={$(SITE_NAMES[site])}, legend to name=depositLegend, legend columns=3, legend style={/tikz/every even column/.append style={column sep=7pt}, draw=none, fill=none, font=\\scriptsize}]")
        else
            println(io, "\n% ===== Site $site =====")
            println(io, "\\nextgroupplot[title={$(SITE_NAMES[site])}, legend=false]")
        end
        for policy_name in POLICIES
            st = POLICY_STYLES[policy_name]
            println(io, "% $(st.label)")
            println(io, "\\addplot[$(st.tikz)] coordinates {\n    ",
                    coords(results[policy_name]["mean"][:, site], results[policy_name]["std"][:, site]), "\n};")
        end
        site == 1 && println(io, "\\legend{", join((POLICY_STYLES[p].label for p in POLICIES), ", "), "}")
    end
    print(io, """

\\end{groupplot}

% Shared legend below the grid
\\node at ([yshift=-0.7em]current bounding box.south) {\\ref{depositLegend}};

\\end{tikzpicture}

\\caption{Average over $(N_SIMS) simulations of deposit belief uncertainty evolution for sites 1-4 under three policies (POMCPOW, Profit Maximizer, Explore Only).}
\\label{fig:deposit_uncertainty_grid}
\\end{figure}
""")
    write(tex_file, String(take!(io)))
end

# PNG preview of the same panels with Plots/GR (no LaTeX needed)
function write_png(results::Dict, png_file::String)
    panels = []
    for site in 1:4
        p = Plots.plot(title = SITE_NAMES[site], xlabel = site >= 3 ? "Time Step" : "",
                       ylabel = isodd(site) ? "Belief Uncertainty (t Li)" : "", legend = site == 2 ? :topright : false,
                       titlefontsize = 10, guidefontsize = 9, tickfontsize = 8, legendfontsize = 8, ylims = (0, Inf))
        for policy_name in POLICIES
            m = results[policy_name]["mean"][:, site]
            sd = results[policy_name]["std"][:, site]
            c, ls, mk = POLICY_STYLES[policy_name].gr
            Plots.plot!(p, 0:length(m)-1, m, ribbon = sd, fillalpha = 0.12, color = c, linestyle = ls,
                        marker = mk, markersize = 2.5, markerstrokewidth = 0, label = POLICY_STYLES[policy_name].label)
        end
        push!(panels, p)
    end
    Plots.savefig(Plots.plot(panels..., layout = (2, 2), size = (900, 650), left_margin = 4Plots.mm, bottom_margin = 3Plots.mm), png_file)
end

function write_csv(results::Dict, csv_file::String)
    open(csv_file, "w") do f
        println(f, "policy,site,steps_taken,mean_uncertainty,std_uncertainty")
        for policy_name in POLICIES, site in 1:4, t in 1:size(results[policy_name]["mean"], 1)
            println(f, "$policy_name,$site,$(t-1),$(results[policy_name]["mean"][t, site]),$(results[policy_name]["std"][t, site])")
        end
    end
end

# =============================================================================
# MAIN
# =============================================================================

results = analyze_belief_uncertainty(POLICIES; planning_model=PLANNING_PRICE_MODEL, n_sims=N_SIMS)

mkpath(OUTPUT_DIR)
base = joinpath(OUTPUT_DIR, "belief_uncertainty_p$(PLANNING_PRICE_MODEL)_e$(EVALUATION_PRICE_MODEL)")
write_figure_tex(results, base * ".tex"); println("LaTeX figure saved to: $base.tex")
write_png(results, base * ".png");        println("PNG preview saved to: $base.png")
write_csv(results, base * ".csv");        println("CSV saved to: $base.csv")

println("\n" * "="^60)
for policy in POLICIES, site in 1:4
    m = results[policy]["mean"][:, site]
    println("$policy, $(SITE_NAMES[site]): initial $(round(m[1], digits=0)) → final $(round(m[end], digits=0)) t Li")
end
