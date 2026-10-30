"""
One-at-a-time sensitivity analysis (reviewer request): each parameter is set to a low and a high value
with everything else at its baseline (src/model.jl defaults), and the main policies are rerun on paired
seeds. Planning and evaluation use the same parameter values (the change describes the world).

Parameters: demand share, deposit uncertainty, CAPEX, OPEX, recovery rate, carbon cost (social cost of
carbon × emissions intensity, which enter the reward only as a product), observation noise, discount
factor, production capacity and deposit size.

  julia --project=. -t 80 scripts/sensitivity_analysis.jl [n_seeds] [alpha] [planning_model] [tree_queries]
"""

include(joinpath(@__DIR__, "pomcpow_tuning_harness.jl"))

n_seeds        = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
α              = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.5
planning_model = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 6
queries        = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 10_000
essential      = length(ARGS) >= 5 && ARGS[5] == "essential"   # 7 parameters instead of 10 (faster)
core           = length(ARGS) >= 5 && ARGS[5] == "core"        # 5 parameters: demand share, deposit uncertainty, recovery, capacity, CAPEX
appendix       = length(ARGS) >= 5 && ARGS[5] == "appendix"    # the remaining reviewer inputs: OPEX, carbon cost, observation noise, discount factor, deposit size
noisedecomp    = length(ARGS) >= 5 && ARGS[5] == "noisedecomp"  # which observation noise drives the ×2 effect: deposit / recovery / exploring / producing
recexp         = length(ARGS) >= 5 && ARGS[5] == "recexp"       # recovery-while-exploring noise sweep: 0.05 (baseline), 0.10, 0.15, 0.20
uncert         = length(ARGS) >= 6 && ARGS[6] == "uncert"      # + all uncertainty magnitudes jointly ×0.5 / ×1.5
eval_model     = 6
seeds          = collect(2001:(2000 + n_seeds))   # held out: tuning experiments used 4001+ and 6001+

B = LiPOMDP()   # baseline values
scale(x::Int, f) = round(Int, x * f)
scale(x, f) = x * f
noise(f) = (σ_obs_d_dle = B.σ_obs_d_dle * f, σ_obs_d_hard_rock = B.σ_obs_d_hard_rock * f,
            σ_obs_recovery_explore = B.σ_obs_recovery_explore * f,
            σ_obs_recovery_production = B.σ_obs_recovery_production * f)

# (parameter, level, keyword overrides)
settings = [
    ("Baseline", "", NamedTuple()),
    ("Demand share", "20%", (market_share_target = 0.20,)),
    ("Demand share", "40%", (market_share_target = 0.40,)),
    ("Deposit uncertainty", "×0.5", (dle_uncertainty_factor = B.dle_uncertainty_factor * 0.5,
                                     mining_uncertainty_factor = B.mining_uncertainty_factor * 0.5)),
    ("Deposit uncertainty", "×1.5", (dle_uncertainty_factor = B.dle_uncertainty_factor * 1.5,
                                     mining_uncertainty_factor = B.mining_uncertainty_factor * 1.5)),
    ("CAPEX", "×0.75", (dle_capex = scale(B.dle_capex, 0.75), mining_capex = scale(B.mining_capex, 0.75))),
    ("CAPEX", "×1.25", (dle_capex = scale(B.dle_capex, 1.25), mining_capex = scale(B.mining_capex, 1.25))),
    ("OPEX", "×0.75", (dle_opex = scale(B.dle_opex, 0.75), mining_opex = scale(B.mining_opex, 0.75))),
    ("OPEX", "×1.25", (dle_opex = scale(B.dle_opex, 1.25), mining_opex = scale(B.mining_opex, 1.25))),
    ("Recovery rate", "−0.10", (recovery = B.recovery .- 0.10,)),
    ("Recovery rate", "+0.10", (recovery = B.recovery .+ 0.10,)),
    ("Carbon cost", "×0.5", (CO2_cost = scale(B.CO2_cost, 0.5),)),
    ("Carbon cost", "×2", (CO2_cost = scale(B.CO2_cost, 2.0),)),
    ("Observation noise", "×0.5", noise(0.5)),
    ("Observation noise", "×2", noise(2.0)),
    ("Discount factor", "0.90", (γ = 0.90,)),
    ("Discount factor", "0.95", (γ = 0.95,)),
    ("Production capacity", "20k t/yr", (output = 20_000.0,)),
    ("Production capacity", "30k t/yr", (output = 30_000.0,)),
    ("Deposit size", "×0.75", (init_deposit_size_dle = B.init_deposit_size_dle * 0.75,
                               init_deposit_size_mining = B.init_deposit_size_mining * 0.75)),
    ("Deposit size", "×1.25", (init_deposit_size_dle = B.init_deposit_size_dle * 1.25,
                               init_deposit_size_mining = B.init_deposit_size_mining * 1.25)),
]

if essential
    keep = ("Baseline", "Demand share", "Deposit uncertainty", "CAPEX", "Recovery rate", "Carbon cost", "Production capacity", "Deposit size")
    settings = [st for st in settings if st[1] in keep]
elseif core
    keep = ("Baseline", "Demand share", "Deposit uncertainty", "Recovery rate", "Production capacity", "CAPEX")
    settings = [st for st in settings if st[1] in keep]
elseif appendix
    keep = ("Baseline", "OPEX", "Carbon cost", "Observation noise", "Discount factor", "Deposit size")
    settings = [st for st in settings if st[1] in keep]
elseif recexp
    settings = [("Baseline", "", NamedTuple())] |> x -> vcat(x,
        [("Recovery explore noise", "$(σ)", (σ_obs_recovery_explore = σ,)) for σ in (0.10, 0.15, 0.20)])
elseif noisedecomp
    settings = [
        ("Baseline", "", NamedTuple()),
        ("Obs noise ×2", "all four", noise(2.0)),
        ("Obs noise ×2", "deposit only", (σ_obs_d_dle = B.σ_obs_d_dle * 2, σ_obs_d_hard_rock = B.σ_obs_d_hard_rock * 2)),
        ("Obs noise ×2", "recovery only", (σ_obs_recovery_explore = B.σ_obs_recovery_explore * 2, σ_obs_recovery_production = B.σ_obs_recovery_production * 2)),
        ("Obs noise ×2", "recovery explore only", (σ_obs_recovery_explore = B.σ_obs_recovery_explore * 2,)),
        ("Obs noise ×2", "recovery produce only", (σ_obs_recovery_production = B.σ_obs_recovery_production * 2,)),
    ]
end
if uncert
    # deposit prior, demand forecast noise and observation noise scaled together; the recovery prior stays at
    # baseline (scaling it ×1.5 would push the Beta prior past the well-formed range)
    allunc(f) = merge(noise(f), (dle_uncertainty_factor = B.dle_uncertainty_factor * f, mining_uncertainty_factor = B.mining_uncertainty_factor * f,
                                 demand_uncertainty = B.demand_uncertainty * f))
    push!(settings, ("All uncertainty", "×0.5", allunc(0.5)))
    push!(settings, ("All uncertainty", "×1.5", allunc(1.5)))
end

policies = [
    "POMCPOW"                      => p -> make_pomcpow(p; rollout=:timing, ucb_c=4.0, max_depth=25, tree_queries=queries),   # final configuration
    "OneStepLookahead_TimingAware" => p -> OneStepLookaheadPolicy(p, n_rollouts=5, rollout_policy=TimingAwarePolicy(p)),
    "TimingAware"                  => p -> TimingAwarePolicy(p),
    "DynamicProfitMaximizer"       => p -> DynamicProfitMaximizerPolicy(p),
]
names = first.(policies)

# Slow POMCPOW jobs first so the tail of the run stays parallel
jobs = [(k, name, seed) for name in names for k in eachindex(settings) for seed in seeds]
results = Dict{Tuple{Int,String,Int}, NamedTuple}()
lk = ReentrantLock(); t0 = time()
println("SENSITIVITY  $(length(settings)) settings × $(length(names)) policies × $(length(seeds)) seeds = $(length(jobs)) episodes  " *
        "planning=$planning_model eval=$eval_model α=$α queries=$queries threads=$(Threads.nthreads())")
Threads.@threads :greedy for (k, name, seed) in jobs
    kw = settings[k][3]
    planning   = LiPOMDP(; price_model_type=planning_model, alpha=α, kw...)
    evaluation = LiPOMDP(; price_model_type=eval_model,     alpha=α, kw...)
    r = run_scenario(Dict(policies)[name](planning), planning, evaluation, LiBeliefUpdater(P=planning), seed)
    lock(lk) do
        results[(k, name, seed)] = r
        @printf("  [%d/%d] %.0f min  %s %s  %s seed=%d  disc=%.3f\n", length(results), length(jobs),
                (time() - t0) / 60, settings[k][1], settings[k][2], name, seed, r.disc)
        flush(stdout)
    end
end
@printf("%d episodes in %.0f min\n", length(jobs), (time() - t0) / 60)

mkpath("outputs/sensitivity")
csv_path = "outputs/sensitivity/sensitivity_p$(planning_model)_a$(α)_q$(queries)_n$(n_seeds)_e$(eval_model)$(essential ? "_essential" : core ? "_core" : appendix ? "_appendix" : noisedecomp ? "_noisedecomp" : recexp ? "_recexp" : "")$(uncert ? "_uncert" : "")_rec20-25_osl5_recexp15.csv"
open(csv_path, "w") do io
    println(io, "parameter,level,policy,seed,discounted_reward,profit,emission_cost,co2_tonnes,met_demand_pct,mines_opened,capex,explorations")
    for ((k, name, seed), r) in sort(collect(results), by=first)
        println(io, join(["\"$(settings[k][1])\"", "\"$(settings[k][2])\"", name, seed, r.disc, r.profit, r.emis,
                          r.co2, r.met, r.mines, r.capex, r.explores], ","))
    end
end

se(x) = std(x) / sqrt(length(x))
vals(k, n, f) = [getfield(results[(k, n, s)], f) for s in seeds]
base_pomcpow = mean(vals(1, "POMCPOW", :disc))
short = Dict("POMCPOW" => "POMCPOW", "OneStepLookahead_TimingAware" => "OSL-TA",
             "TimingAware" => "TA", "DynamicProfitMaximizer" => "DPM")
println("\nMean discounted reward per policy; POMCPOW change vs baseline; POMCPOW − OSL-TA paired over seeds; best policy")
@printf("%-22s %-9s %s %9s %8s %8s %16s  %s\n", "parameter", "level", join([lpad(short[n], 8) for n in names], ""),
        "POMCPOW%", "met%", "mines", "PC−OSL-TA ± SE", "best")
for (k, (param, level, _)) in enumerate(settings)
    means = [mean(vals(k, n, :disc)) for n in names]
    Δ = vals(k, "POMCPOW", :disc) .- vals(k, "OneStepLookahead_TimingAware", :disc)
    @printf("%-22s %-9s %s %+8.1f%% %7.0f%% %8.2f %+8.3f ± %.3f  %s\n", param, level, join([@sprintf("%8.3f", m) for m in means]),
            100 * (means[1] - base_pomcpow) / abs(base_pomcpow), mean(vals(k, "POMCPOW", :met)), mean(vals(k, "POMCPOW", :mines)),
            mean(Δ), se(Δ), short[names[argmax(means)]])
end
println("Note: γ and carbon cost change the objective itself, so compare rankings rather than absolute rewards for those rows.")
println("CSV: $csv_path")
