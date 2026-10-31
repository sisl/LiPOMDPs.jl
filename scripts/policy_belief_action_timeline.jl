using LiPOMDPs
using POMDPs
using POMCPOW
using Random
using Distributions
using Plots
using PGFPlotsX
using Statistics

import LiPOMDPs: calculate_total_production, compute_costs, compute_emissions

const PGFPLOTSX_PRIMED = Ref(false)

const PLANNING_PRICE_MODELS = length(ARGS) >= 1 ? parse.(Int, split(ARGS[1], ",")) : [1, 2, 3, 4, 6]
const EVALUATION_PRICE_MODEL = 6
const ALPHA = 0.5
const TIME_HORIZON = LiPOMDP().time_horizon   # model default (1995–2025)
const OUTPUT_DIR = "outputs/belief_timeline"
const N_STEPS = TIME_HORIZON
const HEURISTIC = "DynamicProfitMaximizer" # e.g. "ExploreOnly", "RandomHeuristicPolicy", "OneStepLookahead"

# Match final_results_best_pomcpow.jl seed scheme:
# train seeds = 1001.., test seeds = 2001..
const SEED_MODE = "test" # "train", "test", or "custom"
const SEED_INDEX = 1     # i in (i + 1000) or (i + 2000)
const CUSTOM_SEED = 42

# Same POMCPOW setup as scripts/final_results_best_pomcpow.jl
const POMCPOW_CONFIG = Dict(
    "tree_queries" => 10_000,
    "max_depth" => 25,
    "ucb_c" => 4.0,
    "enable_action_pw" => true,
    "k_action" => 5.0,
    "alpha_action" => 0.5,
    "k_observation" => 3.0,
    "alpha_observation" => 0.3,
)

const ACTION_COLORS = Dict(
    :explore => "#B8E0D2",     # baby green
    :open => "#F7C59F",        # orange
    :active => "#F3E8A7",      # yellow
    :inactive => "#E6E6E6",    # light gray
    :restore => "#E57373",     # red
    :true_reserves => "#2F2F2F" # charcoal
)

const ACTION_MARKERS = Dict(
    :explore => :diamond,
    :open => :square,
    :active => :utriangle,
    :inactive => :circle,
    :restore => :star5,
    :true_reserves => :circle
)

const DEPOSIT_LABELS = [
    "DLE #1",
    "DLE #2",
    "Mineral ore #1",
    "Mineral ore #2"
]

const DEPOSIT_COLORS = [
    "#A7C7E7", # baby blue
    "#A7C7E7", # baby blue
    "#A7C7E7", # baby blue
    "#A7C7E7"  # baby blue
]

function make_heuristic(pomdp::LiPOMDP, name::String)
    if name == "ExploreOnly"
        return ExploreOnlyPolicy(pomdp), "ExploreOnly"
    elseif name == "RandomHeuristic" || name == "RandomHeuristicPolicy"
        return RandomHeuristicPolicy(pomdp, rng=Random.GLOBAL_RNG, debug=false), "RandomHeuristic"
    elseif name == "ProfitMaximizer"
        return ProfitMaximizerPolicy(pomdp, debug=false, top_k=1, times=[1]), "ProfitMaximizer"
    elseif name == "EmissionMinimizer"
        return EmissionMinimizerPolicy(pomdp, debug=false, top_k=1, times=[1]), "EmissionMinimizer"
    elseif name == "DynamicProfitMaximizer"
        return DynamicProfitMaximizerPolicy(pomdp, debug=false), "DynamicProfitMaximizer"
    elseif name == "DynamicEmissionMinimizer"
        return DynamicEmissionMinimizerPolicy(pomdp, debug=false), "DynamicEmissionMinimizer"
    elseif name == "OneMineProfitMaximization"
        return ProfitMaximizerPolicy(pomdp, debug=false, top_k=1, times=[1]), "OneMineProfitMaximization"
    elseif name == "OneMineEmissionMinimization"
        return EmissionMinimizerPolicy(pomdp, debug=false, top_k=1, times=[1]), "OneMineEmissionMinimization"
    elseif name == "OneStepLookahead"
        return OneStepLookaheadPolicy(pomdp, n_rollouts=30, debug=false), "OneStepLookahead"
    else
        error("Unsupported heuristic: $name")
    end
end

function resolve_seed()
    if SEED_MODE == "train"
        return SEED_INDEX + 1000
    elseif SEED_MODE == "test"
        return SEED_INDEX + 2000
    elseif SEED_MODE == "custom"
        return CUSTOM_SEED
    else
        error("Unsupported SEED_MODE: $SEED_MODE")
    end
end

# The agent filters with its own (planning) model, as in final_results_best_pomcpow.jl. Each step's
# environment noise comes from (seed, t), so every policy faces the same price/demand draws.
function simulate_policy(planning_pomdp::LiPOMDP, evaluation_pomdp::LiPOMDP, policy, n_steps::Int, seed::Int;
                         initial_state, initial_belief)
    s, b = initial_state, initial_belief
    updater = LiBeliefUpdater(P=planning_pomdp)

    prev_states = State[]   # state the action was taken in (used for costs and rewards)
    states = State[]        # state after the step (plotted)
    beliefs = LiBelief[]
    actions = Action[]      # executed actions: opens refused by the capital budget are removed
    rewards = Float64[]

    for t in 1:n_steps
        a = POMDPs.action(policy, b)
        Random.seed!(hash((seed, t)))
        sp, o, r = gen(evaluation_pomdp, s, a, MersenneTwister(hash((seed, t, :env))))
        b = update(updater, b, a, o)

        push!(prev_states, s)
        push!(actions, LiPOMDPs.effective_action(evaluation_pomdp, s.have_opened, a))
        push!(states, sp)
        push!(beliefs, b)
        push!(rewards, r)

        s = sp
        isterminal(evaluation_pomdp, sp) && break
    end

    return (prev_states=prev_states, states=states, beliefs=beliefs, actions=actions, rewards=rewards)
end

function action_status(a::Action, s::State, site::Int)
    if a.type[site] == 1
        return :explore
    elseif a.type[site] == 2
        return :open
    elseif a.type[site] == 3
        return :restore
    elseif s.have_opened[site] && !s.is_depleted[site] && !s.restored_mines[site]
        return :active
    else
        return :inactive
    end
end

function belief_plot_range(results_list, site::Int)
    max_true = maximum([s.deposit[site] for res in results_list for s in res.states])
    max_belief = maximum([mean(b.deposits_distribution[site]) + 3 * std(b.deposits_distribution[site])
        for res in results_list for b in res.beliefs])
    return max(1.0, max(max_true, max_belief))
end

# Convert exponent to superscript Unicode characters
function exp_to_superscript(exp::Int)
    superscript_digits = Dict(0 => '⁰', 1 => '¹', 2 => '²', 3 => '³', 4 => '⁴', 5 => '⁵', 6 => '⁶', 7 => '⁷', 8 => '⁸', 9 => '⁹')
    minus = '⁻'
    if exp < 0
        return string(minus) * join([superscript_digits[d] for d in reverse(digits(abs(exp)))])
    else
        return join([superscript_digits[d] for d in reverse(digits(exp))])
    end
end

function add_belief_violin!(p, dist::UnivariateDistribution, t::Int, y_max::Float64, color; width=0.35)
    ys = range(0.0, y_max, length=200)
    pdf_vals = pdf.(dist, ys)
    max_pdf = maximum(pdf_vals)
    if max_pdf <= 0
        return
    end
    scaled = width .* (pdf_vals ./ max_pdf)
    x_left = t .- scaled
    x_right = t .+ scaled
    x = vcat(x_left, reverse(x_right))
    y = vcat(ys, reverse(ys))
    plot!(p, x, y, seriestype=:shape, color=color, alpha=0.55, linecolor=color, linewidth=0.7, label="")
end

function add_action_marker!(p, t::Int, y::Float64, status::Symbol)
    marker_shape = ACTION_MARKERS[status]
    x = t
    # Make RESTORE star bigger
    marker_size = status == :restore ? 12 : 8
    if status == :inactive
        scatter!(p, [x], [y], marker=marker_shape, markersize=marker_size,
            markercolor=:white, markerstrokecolor=ACTION_COLORS[status],
            markerstrokewidth=0.5, label="")
    else
        scatter!(p, [x], [y], marker=marker_shape, markersize=marker_size,
            markercolor=ACTION_COLORS[status], markerstrokecolor=:black,
            markerstrokewidth=0.3, label="")
    end
end

function add_legend!(p)    scatter!(p, [NaN], [NaN], marker=ACTION_MARKERS[:explore], markersize=2.5,
        markercolor=ACTION_COLORS[:explore], markerstrokecolor=:black, markerstrokewidth=0.08, label="EXPLORE")
    scatter!(p, [NaN], [NaN], marker=ACTION_MARKERS[:open], markersize=2.5,
        markercolor=ACTION_COLORS[:open], markerstrokecolor=:black, markerstrokewidth=0.08, label="BUILD")
    scatter!(p, [NaN], [NaN], marker=ACTION_MARKERS[:active], markersize=2.5,
        markercolor=ACTION_COLORS[:active], markerstrokecolor=:black, markerstrokewidth=0.08, label="MINE ACTIVE")
    scatter!(p, [NaN], [NaN], marker=ACTION_MARKERS[:inactive], markersize=2.5,
        markercolor=:white, markerstrokecolor=ACTION_COLORS[:inactive],
        markerstrokewidth=0.1, label="MINE INACTIVE")
    scatter!(p, [NaN], [NaN], marker=ACTION_MARKERS[:restore], markersize=2.5,
        markercolor=ACTION_COLORS[:restore], markerstrokecolor=:black, markerstrokewidth=0.08, label="RESTORE")
    scatter!(p, [NaN], [NaN], marker=ACTION_MARKERS[:true_reserves], markersize=2.5,
        markercolor=ACTION_COLORS[:true_reserves], markerstrokecolor=:black, markerstrokewidth=0.08, label="TRUE RESERVES")
    scatter!(p, [NaN], [NaN], marker=:rect, markersize=4,
        markercolor=DEPOSIT_COLORS[1], markeralpha=0.55, markerstrokecolor=DEPOSIT_COLORS[1],
        markerstrokewidth=0.7, label="BELIEF")
end

function summarize_metrics(pomdp::LiPOMDP, result)
    total_reward = 0.0
    total_discounted_reward = 0.0
    total_profit = 0.0
    total_emission_cost = 0.0
    total_demand = 0.0
    total_production = 0.0

    for (t, (s, a, r)) in enumerate(zip(result.prev_states, result.actions, result.rewards))
        costs = compute_costs(pomdp, s, a)
        emissions = compute_emissions(pomdp, s, a)
        current_production, _ = calculate_total_production(pomdp, s)
        revenue = current_production * s.price
        profit = revenue - costs

        total_reward += r
        total_discounted_reward += pomdp.γ^(t - 1) * r
        total_profit += profit
        total_emission_cost += emissions
        total_demand += s.company_demand
        total_production += current_production
    end

    met_demand_pct = total_demand > 0 ? (total_production / total_demand) * 100 : 0.0

    return Dict(
        "discounted_reward" => total_discounted_reward,
        "total_profit" => total_profit,
        "total_emission_cost" => total_emission_cost,
        "met_demand_pct" => met_demand_pct
    )
end

function format_metrics(metrics::Dict)
    disc_reward = round(metrics["discounted_reward"], digits=2)
    profit_m = round(metrics["total_profit"] / 1e6, digits=0)
    emission_m = round(metrics["total_emission_cost"] / 1e6, digits=1)
    met_pct = round(metrics["met_demand_pct"], digits=1)
    return "Time Period\n\nDisc. Reward: $(disc_reward)\nProfit: $(profit_m)M USD\nEmissions: $(emission_m)M USD\nMet demand: $(met_pct)%"
end

mkpath(OUTPUT_DIR)
seed = resolve_seed()
Random.seed!(seed)
println("Using seed: $seed ($(SEED_MODE) sim index $(SEED_INDEX))")

# Prime PGFPlotsX with a dummy plot to avoid empty exports for the first model
pgfplotsx()
dummy_plot = plot([0, 1], [0, 1], legend=false, framestyle=:none)
savefig(dummy_plot, "pgfplotsx_dummy.tex")
rm("pgfplotsx_dummy.tex"; force=true)
gr()  # Switch back to GR for PNG generation

for planning_price_model in PLANNING_PRICE_MODELS
    println("Generating belief/action timeline comparison for planning price model $(planning_price_model)…")

    planning_pomdp = LiPOMDP(price_model_type=planning_price_model, time_horizon=TIME_HORIZON, alpha=ALPHA)
    evaluation_pomdp = LiPOMDP(price_model_type=EVALUATION_PRICE_MODEL, time_horizon=TIME_HORIZON, alpha=ALPHA)

    # Create initial state/belief ONCE so all policies start from the same true deposits
    initial_state, initial_belief = create_initial_belief(planning_pomdp)

    pomcpow_solver = POMCPOWSolver(
        max_depth = POMCPOW_CONFIG["max_depth"],
        tree_queries = POMCPOW_CONFIG["tree_queries"],
        criterion = MaxUCB(POMCPOW_CONFIG["ucb_c"]),
        enable_action_pw = POMCPOW_CONFIG["enable_action_pw"],
        k_action = POMCPOW_CONFIG["k_action"],
        alpha_action = POMCPOW_CONFIG["alpha_action"],
        k_observation = POMCPOW_CONFIG["k_observation"],
        alpha_observation = POMCPOW_CONFIG["alpha_observation"],
        estimate_value = FORollout(TimingAwarePolicy(planning_pomdp)),
        final_criterion = POMCPOW.MaxQ()
    )
    pomcpow_policy = solve(pomcpow_solver, planning_pomdp)

    # Deep copy initial state/belief for each simulation since State is mutable
    pomcpow_results = simulate_policy(planning_pomdp, evaluation_pomdp, pomcpow_policy, N_STEPS, seed; initial_state=deepcopy(initial_state), initial_belief=deepcopy(initial_belief))
    explore_results = simulate_policy(planning_pomdp, evaluation_pomdp, ExploreOnlyPolicy(planning_pomdp), N_STEPS, seed; initial_state=deepcopy(initial_state), initial_belief=deepcopy(initial_belief))
    onemine_results = simulate_policy(
        planning_pomdp,
        evaluation_pomdp,
        ProfitMaximizerPolicy(planning_pomdp, debug=false, top_k=1, times=[1]),
        N_STEPS, seed;
        initial_state=deepcopy(initial_state),
        initial_belief=deepcopy(initial_belief)
    )
    dynprofit_results = simulate_policy(
        planning_pomdp,
        evaluation_pomdp,
        DynamicProfitMaximizerPolicy(planning_pomdp, debug=false),
        N_STEPS, seed;
        initial_state=deepcopy(initial_state),
        initial_belief=deepcopy(initial_belief)
    )

    policy_entries = [
        ("POMCPOW", pomcpow_results),
        ("ExploreOnly", explore_results),
        ("OneMineProfitMaximization", onemine_results),
        ("DynamicProfitMaximizer", dynprofit_results)
    ]

    policy_metrics = Dict(
        name => summarize_metrics(evaluation_pomdp, result) for (name, result) in policy_entries
    )

function build_plot(policy_entries, n_sites, policy_metrics)
    layout_spec = @layout([grid(4, 4); legend_row{0.003h}])  # Small legend row height with space for bottom text
    layout_plot = plot(
        layout=layout_spec,
        size=(2800, 1600),
        dpi=300,
        margin=0Plots.mm,  # Minimal spacing between plots
        left_margin=14Plots.mm,  # Keep left margin for y-axis labels
        right_margin=85Plots.mm,  # Increased right margin to ensure legend fits completely
        top_margin=2Plots.mm,  # Reduced top margin for less space between graphs
        bottom_margin=0Plots.mm,  # No bottom margin - legend at very bottom
        plot_titlefont=font(20)  # Increased title font
    )

    for site in 1:n_sites
        y_max = belief_plot_range([pomcpow_results, explore_results, onemine_results, dynprofit_results], site)
        y_action = y_max * 1.04
        y_top = y_max * 1.10

        for (col, (policy_name, result)) in enumerate(policy_entries)
            idx = (site - 1) * 4 + col
            p = layout_plot[idx]

            for t in eachindex(result.states)
                dist = result.beliefs[t].deposits_distribution[site]
                add_belief_violin!(p, dist, t, y_max, DEPOSIT_COLORS[site])

                true_val = result.states[t].deposit[site]
                scatter!(p, [t], [true_val], marker=:circle, markersize=3,
                    markercolor=ACTION_COLORS[:true_reserves], markerstrokecolor=:black, markerstrokewidth=0.3, label="")

                status = action_status(result.actions[t], result.states[t], site)
                add_action_marker!(p, t, y_action, status)
            end

            # Format y-axis ticks to scientific notation with one decimal place and superscript exponents
            y_tick_values = range(0.0, y_top, length=6)
            y_tick_labels = [if v == 0.0
                    "0.0"
                else
                    exp = floor(Int, log10(abs(v)))
                    mantissa = round(v / 10.0^exp, digits=1)
                    if exp == 0
                        string(mantissa)
                    else
                        # Use Unicode superscripts in regular font
                        "$(mantissa)×10$(exp_to_superscript(exp))"
                    end
                end for v in y_tick_values]
            
            # X-axis ticks: 0, 5, 10, 15, 20, 25
            x_tick_values = collect(0:5:N_STEPS)
            x_tick_labels = string.(x_tick_values)
            
            # Larger bottom margin for bottom row subplots to accommodate metrics text
            # More space between rows, more space between columns
            subplot_bottom_margin = site == n_sites ? 50Plots.mm : 4Plots.mm  # More space for met demand text
            subplot_left_margin = col == 1 ? 0Plots.mm : 5Plots.mm  # More space between columns, col 1 uses overall left margin
            subplot_right_margin = col == 4 ? 0Plots.mm : 5Plots.mm  # More space between columns
            
            # For column 1, don't set left_margin to preserve y-axis labels
            if col == 1
                plot!(p,
                    xlim=(0, N_STEPS + 0.5),
                    ylim=(0.0, y_top),
                    xticks=(x_tick_values, x_tick_labels),
                    yticks=(y_tick_values, y_tick_labels),
                    grid=:y,
                    gridalpha=0.2,
                    legend=false,
                    xaxis=true,
                    yaxis=true,
                    tickfont=font(14),
                    xguidefont=font(24, :normal),  # Set x-axis font consistently for all columns
                    bottom_margin=subplot_bottom_margin,
                    right_margin=subplot_right_margin
                )
            else
                plot!(p,
                    xlim=(0, N_STEPS + 0.5),
                    ylim=(0.0, y_top),
                    xticks=(x_tick_values, x_tick_labels),
                    yticks=(y_tick_values, y_tick_labels),
                    grid=:y,
                    gridalpha=0.2,
                    legend=false,
                    xaxis=true,
                    yaxis=true,
                    tickfont=font(14),
                    xguidefont=font(24, :normal),  # Set x-axis font consistently for all columns
                    bottom_margin=subplot_bottom_margin,
                    left_margin=subplot_left_margin,
                    right_margin=subplot_right_margin
                )
            end

            if site == 1
                title!(p, policy_name == "POMCPOW" ? "POMDP Policies (POMCPOW)" : policy_name, titlefont=font(20))
            end

            if site == n_sites
                # Explicitly set x-axis label with consistent font size for all policies
                xlabel!(p, format_metrics(policy_metrics[policy_name]), guidefont=font(24, :normal))  # Same font for all x-axis labels
            end

            if col == 1
                # Set ylabel after xlabel to ensure it's not affected, with explicit font settings
                ylabel!(p, "Deposit belief\n$(DEPOSIT_LABELS[site])", guidefont=font(24, :bold))  # Same font for all rows
            end
        end
    end

    legend_plot = layout_plot[end]
    plot!(legend_plot,
        legend=:bottom,
        legendtitle="",  # No legend title
        legend_columns=7,
        legendfont=font(14),
        legend_title_font=font(0.1),  # Minimal title font to remove title area padding
        legend_title_position=:top,  # Position title at top (even though empty)
        legend_column_gap=0.0001,  # Minimal spacing between columns
        legend_row_gap=0.0005,  # Minimal spacing between rows
        legend_marker_gap=0.03,  # Minimal gap between marker and text
        left_margin=-10Plots.mm,  # Negative margin to reduce spacing
        right_margin=-10Plots.mm,  # Negative margin to reduce spacing
        top_margin=-5Plots.mm,  # Negative margin to remove title area padding
        bottom_margin=0Plots.mm,  # No bottom margin for legend
        framestyle=:none,
        grid=false,
        xticks=false,
        yticks=false,
        xaxis=false,
        yaxis=false
    )
    add_legend!(legend_plot)

    return layout_plot
end

function build_plot_tex(policy_entries, n_sites, policy_metrics)
    layout_plot = plot(
        layout=(4, 4),
        size=(2800, 1600),
        dpi=300,
        margin=0Plots.mm,  # Minimal spacing between plots
        left_margin=14Plots.mm,  # Keep left margin for y-axis labels
        right_margin=85Plots.mm,  # Increased right margin to ensure legend fits completely
        top_margin=2Plots.mm,  # Reduced top margin
        bottom_margin=0Plots.mm,  # No bottom margin - legend at very bottom
        plot_titlefont=font(20)  # Increased title font
    )

    for site in 1:n_sites
        y_max = belief_plot_range([pomcpow_results, explore_results, onemine_results, dynprofit_results], site)
        y_action = y_max * 1.10
        y_top = y_max * 1.18

        for (col, (policy_name, result)) in enumerate(policy_entries)
            idx = (site - 1) * 4 + col
            p = layout_plot[idx]

            for t in eachindex(result.states)
                dist = result.beliefs[t].deposits_distribution[site]
                add_belief_violin!(p, dist, t, y_max, DEPOSIT_COLORS[site])

                true_val = result.states[t].deposit[site]
                scatter!(p, [t], [true_val], marker=:circle, markersize=3,
                    markercolor=ACTION_COLORS[:true_reserves], markerstrokecolor=:black, markerstrokewidth=0.3, label="")

                status = action_status(result.actions[t], result.states[t], site)
                add_action_marker!(p, t, y_action, status)
            end

            show_legend = site == 1 && col == 1
            # Format y-axis ticks to scientific notation with one decimal place and superscript exponents
            y_tick_values = range(0.0, y_top, length=6)
            y_tick_labels = [if v == 0.0
                    "0.0"
                else
                    exp = floor(Int, log10(abs(v)))
                    mantissa = round(v / 10.0^exp, digits=1)
                    if exp == 0
                        string(mantissa)
                    else
                        # Use Unicode superscripts in regular font
                        "$(mantissa)×10$(exp_to_superscript(exp))"
                    end
                end for v in y_tick_values]
            
            # X-axis ticks: 0, 5, 10, 15, 20, 25
            x_tick_values = collect(0:5:N_STEPS)
            x_tick_labels = string.(x_tick_values)
            
            plot!(p,
                xlim=(0, N_STEPS + 0.5),
                ylim=(0.0, y_top),
                xticks=(x_tick_values, x_tick_labels),
                yticks=(y_tick_values, y_tick_labels),
                grid=:y,
                gridalpha=0.2,
                legend=show_legend ? :bottom : false,
                legendtitle="",  # No legend title
                legend_columns=show_legend ? 7 : 1,
                legendfont=show_legend ? font(16) : font(8),  # Bigger font for legend
                legend_title_font=show_legend ? font(1) : font(8),  # Minimal title font since no title
                legend_column_gap=show_legend ? 0.0001 : 0,  # Minimal spacing between columns
                legend_row_gap=show_legend ? 0.0005 : 0,  # Minimal spacing between rows
                legend_marker_gap=show_legend ? 0.03 : 0,  # Minimal gap between marker and text
                xaxis=true,  # Explicitly enable x-axis
                yaxis=true,   # Explicitly enable y-axis
                tickfont=font(14),  # Smaller font size for y-axis numbers
                xguidefont=font(22, :normal),  # Set x-axis font consistently for all columns
                bottom_margin=(site == n_sites ? 50Plots.mm : 4Plots.mm),  # More space for met demand text
                left_margin=(col == 1 ? 0Plots.mm : 5Plots.mm),  # More space between columns
                right_margin=(col == 4 ? 0Plots.mm : 5Plots.mm)  # More space between columns
            )

            if site == 1
                title!(p, policy_name == "POMCPOW" ? "POMDP Policies (POMCPOW)" : policy_name, titlefont=font(20))
                if show_legend
                    add_legend!(p)
                end
            end

            if site == n_sites
                # Explicitly set x-axis label with consistent font size for all policies
                xlabel!(p, format_metrics(policy_metrics[policy_name]), guidefont=font(24, :normal))  # Same font for all x-axis labels
            end

            if col == 1
                # Set ylabel after xlabel to ensure it's not affected, with explicit font settings
                ylabel!(p, "Deposit belief\n$(DEPOSIT_LABELS[site])", guidefont=font(24, :bold))  # Same font for all rows
            end
        end
    end

    return layout_plot
end

    # Save LaTeX (pgfplots) version first
    pgfplotsx()
    # Prime PGFPlotsX for each model to avoid empty exports
    init_plot = plot([0, 1], [0, 1])
    savefig(init_plot, "pgfplotsx_init.tex")
    rm("pgfplotsx_init.tex"; force=true)
    plot_layout_tex = build_plot_tex(policy_entries, planning_pomdp.n_deposits, policy_metrics)
    tex_path = joinpath(OUTPUT_DIR, "belief_action_timeline_comparison_p$(planning_price_model)_e$(EVALUATION_PRICE_MODEL).tex")
    try
        savefig(plot_layout_tex, tex_path)
        if filesize(tex_path) < 1000
            plot_layout_tex = build_plot_tex(policy_entries, planning_pomdp.n_deposits, policy_metrics)
            savefig(plot_layout_tex, tex_path)
        end
        println("Saved plot to $(tex_path)")
    catch err
        println("LaTeX export failed: $(err)")
    end

    # Save PDF (vector) after TeX export
    gr()
    plot_layout = build_plot(policy_entries, planning_pomdp.n_deposits, policy_metrics)
    pdf_path = joinpath(OUTPUT_DIR, "belief_action_timeline_comparison_p$(planning_price_model)_e$(EVALUATION_PRICE_MODEL).pdf")
    savefig(plot_layout, pdf_path)
    println("Saved plot to $(pdf_path)")
end
