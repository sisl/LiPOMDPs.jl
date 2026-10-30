"""
Where does POMCPOW beat every heuristic? Observation-noise search with a search/confirm split.

Modes (run in this order; each writes its choice to outputs/noise_tuning/):

  pick     Choose POMCPOW's tree-query budget from the compute sweeps: the budget that beats the
           best heuristic in the most cells (ties: larger mean margin). Heuristics are every
           non-POMCPOW policy in the compute sweeps plus the γ = 0.926 grid.
  search   For that budget, run a grid of recovery-test noise × deposit-noise multipliers on
           SEARCH_CELLS with seeds 4001–4010, against every heuristic. Writes the full map and
           picks the setting with the most cells won (ties: larger mean margin).
  confirm  Run the chosen setting on all 25 cells with FRESH seeds 5001–5010. Selecting among
           12 settings on the same seeds overstates the winner's margin; report these numbers.

Every setting in the search grid is within a realistic range (recovery σ: pilot plant 0.03 to
bench test 0.12; deposit noise ×0.5–×2 of the base drilling precision), and the full map is saved
so the result can be reported as "POMCPOW's advantage vs information quality", not one setting.

  julia --project=. -t 72 scripts/tune_observation_noise.jl pick
  julia --project=. -t 72 scripts/tune_observation_noise.jl search  [n_seeds] [resume_log]
  julia --project=. -t 72 scripts/tune_observation_noise.jl confirm [n_seeds] [gamma]
"""

include(joinpath(@__DIR__, "pomcpow_tuning_harness.jl"))
using CSV, DataFrames

const OUT            = "outputs/noise_tuning"
const SWEEP_DIR      = "outputs/pomcpow_vs_heuristics"
const EVAL_MODEL     = 6
const ALL_CELLS      = [(pm, α) for pm in (1, 2, 3, 4, 6) for α in (0.0, 0.25, 0.5, 0.75, 1.0)]
const SEARCH_CELLS   = [(1, 0.75), (2, 1.0), (3, 0.5), (3, 1.0), (4, 1.0), (6, 1.0)]  # wins, ties and losses
const RECOVERY_SIGMA = [0.03, 0.05, 0.08, 0.12]
const DEPOSIT_MULT   = [0.5, 1.0, 2.0]
const GRID_HEURISTICS = ["OneStepLookahead", "DynamicProfitMaximizer", "DynamicEmissionMinimizer", "AlphaAware",
                         "ExploreOnly", "OneMineProfit", "OneMineEmission", "Random"]
const BASE = LiPOMDP()

mode = length(ARGS) >= 1 ? ARGS[1] : error("mode required: pick | search | confirm")
mkpath(OUT)

heuristic_configs() = [
    "OneStep timing n=10"      => p -> OneStepLookaheadPolicy(p, n_rollouts=10, rollout_policy=TimingAwarePolicy(p)),
    "OneStep timing n=20"      => p -> OneStepLookaheadPolicy(p, n_rollouts=20, rollout_policy=TimingAwarePolicy(p)),
    "OneStep timing n=30"      => p -> OneStepLookaheadPolicy(p, n_rollouts=30, rollout_policy=TimingAwarePolicy(p)),
    "TimingAware"              => p -> TimingAwarePolicy(p),
    "OneStepLookahead"         => make_onestep,                     # AlphaAware base
    "DynamicProfitMaximizer"   => p -> DynamicProfitMaximizerPolicy(p),
    "DynamicEmissionMinimizer" => p -> DynamicEmissionMinimizerPolicy(p),
    "AlphaAware"               => p -> AlphaAwareRolloutPolicy(p),
    "ExploreOnly"              => p -> ExploreOnlyPolicy(p),
    "OneMineProfit"            => p -> ProfitMaximizerPolicy(p, top_k=1, times=[1]),
    "OneMineEmission"          => p -> EmissionMinimizerPolicy(p, top_k=1, times=[1]),
    "Random"                   => p -> RandomHeuristicPolicy(p, debug=false),
]
pomcpow_name(q) = "POMCPOW timing q=$q"
noise_kwargs(rσ, dm) = (σ_obs_recovery_explore = rσ,
                        σ_obs_d_dle = dm * BASE.σ_obs_d_dle, σ_obs_d_hard_rock = dm * BASE.σ_obs_d_hard_rock)
se(x) = length(x) > 1 ? std(x) / sqrt(length(x)) : NaN

"""Paired verdict per cell: POMCPOW − best heuristic. `disc(cell, name)` returns per-seed rewards."""
function verdicts(cells, pname, hnames, disc)
    map(cells) do c
        means = Dict(h => mean(disc(c, h)) for h in hnames)
        bh = argmax(means)
        Δ = disc(c, pname) .- disc(c, bh)
        (cell = c, best = bh, Δ = mean(Δ), se = se(Δ))
    end
end

summary(vs) = (wins = count(v -> v.Δ > 0, vs), sig = count(v -> v.Δ > 2v.se, vs),
               losses_sig = count(v -> v.Δ < -2v.se, vs), mean_Δ = mean(v.Δ for v in vs))

function print_verdicts(vs)
    for v in vs
        tag = (v.Δ == 0 && v.se == 0) ? "tie" : v.Δ > 2v.se ? "WIN (>2SE)" : v.Δ > 0 ? "win (noise)" : v.Δ > -2v.se ? "loss (noise)" : "LOSS (>2SE)"
        @printf("  p%d α=%-5.2f  best heuristic %-26s  Δ %+7.3f ± %.3f  %s\n", v.cell[1], v.cell[2], v.best, v.Δ, v.se, tag)
    end
    s = summary(vs)
    @printf("  POMCPOW ahead in %d/%d cells (%d beyond 2 SE); clear losses %d; mean Δ %+.3f\n",
            s.wins, length(vs), s.sig, s.losses_sig, s.mean_Δ)
end

"""
Episodes finished by an earlier, interrupted run, read back from its log. Only the discounted
reward and exploration count are logged; the other metrics are NaN for these episodes.
"""
function read_done(log_path)
    done = Dict{Tuple{Tuple{Float64,Float64},Tuple{Int,Float64},String,Int}, NamedTuple}()
    pat = r"rσ=([\d.]+) dep×([\d.]+) p(\d) α=([\d.]+) (.+) seed=(\d+)\s+disc=(-?[\d.]+) expl=(\d+)"
    for line in eachline(log_path)
        m = match(pat, line)
        m === nothing && continue
        rσ, dm, pm, α, name, seed, d, e = m.captures
        done[((parse(Float64, rσ), parse(Float64, dm)), (parse(Int, pm), parse(Float64, α)), name, parse(Int, seed))] =
            (disc = parse(Float64, d), profit = NaN, emis = NaN, co2 = NaN, met = NaN, mines = NaN, capex = NaN,
             explores = parse(Int, e))
    end
    return done
end

"""Run every (setting, cell, policy, seed) episode in parallel, skipping any in `done`. Settings are (recovery σ, deposit ×)."""
function run_all(settings, cells, configs, seeds; γ=BASE.γ, label="", done=Dict())
    all_jobs = [(st, c, name, seed) for (name, _) in configs for st in settings for c in cells for seed in seeds]
    jobs = filter(j -> !haskey(done, j), all_jobs)
    cfg = Dict(configs)
    results = Dict{Tuple{Tuple{Float64,Float64},Tuple{Int,Float64},String,Int}, NamedTuple}()
    for j in all_jobs
        haskey(done, j) && (results[j] = done[j])
    end
    isempty(done) || println("resuming: $(length(all_jobs) - length(jobs)) episodes reused from the earlier log")
    lk = ReentrantLock(); t0 = time()
    println("$label: $(length(jobs)) episodes on $(Threads.nthreads()) threads, γ=$γ")
    Threads.@threads :greedy for (st, c, name, seed) in jobs
        kw = noise_kwargs(st...)
        planning   = LiPOMDP(; price_model_type=c[1],      alpha=c[2], γ=γ, kw...)
        evaluation = LiPOMDP(; price_model_type=EVAL_MODEL, alpha=c[2], γ=γ, kw...)
        r = run_scenario(cfg[name](planning), planning, evaluation, LiBeliefUpdater(P=planning), seed)
        lock(lk) do
            results[(st, c, name, seed)] = r
            @printf("  [%d/%d] %.0f min  rσ=%.2f dep×%.1f p%d α=%.2f %s seed=%d  disc=%.3f expl=%d\n",
                    length(results), length(all_jobs), (time() - t0) / 60, st..., c..., name, seed, r.disc, r.explores)
            flush(stdout)
        end
    end
    @printf("%d episodes in %.0f min\n", length(jobs), (time() - t0) / 60)
    return results
end

function write_csv(path, results)
    open(path, "w") do io
        println(io, "recovery_sigma,deposit_mult,planning_model,eval_model,alpha,policy,seed,discounted_reward,profit,emission_cost,co2_tonnes,met_demand_pct,mines_opened,capex,explorations")
        for ((st, c, name, seed), r) in sort(collect(results), by=first)
            println(io, join([st..., c[1], EVAL_MODEL, c[2], name, seed, r.disc, r.profit, r.emis, r.co2, r.met, r.mines, r.capex, r.explores], ","))
        end
    end
    println("CSV: $path")
end

if mode == "pick"
    # ---- choose POMCPOW's tree-query budget from the compute sweeps --------------------------
    sweeps = vcat([DataFrame(CSV.File(f)) for f in readdir(SWEEP_DIR; join=true)
                   if occursin(r"compute_sweep_.*_e6\.csv$", f)]...; cols=:union)
    grid = DataFrame(CSV.File(joinpath(SWEEP_DIR, "grid_e6_g0.926_recovery_budget.csv")))
    grid = grid[in.(grid.policy, Ref(GRID_HEURISTICS)), :]
    df = vcat(select(sweeps, :planning_model, :alpha, :policy, :seed, :discounted_reward),
              select(grid,   :planning_model, :alpha, :policy, :seed, :discounted_reward))
    seeds = 4001:4010
    lookup = Dict((r.planning_model, Float64(r.alpha), r.policy, r.seed) => r.discounted_reward for r in eachrow(df))
    disc(c, name) = [lookup[(c[1], c[2], name, s)] for s in seeds]
    pnames = sort(unique(filter(startswith("POMCPOW timing q="), df.policy)))
    hnames = unique(filter(!startswith("POMCPOW"), df.policy))
    println("Heuristics compared: ", hnames)
    best = nothing; best_key = (-1, -Inf)
    for p in pnames
        global best, best_key
        vs = verdicts(ALL_CELLS, p, hnames, disc)
        println("\n$p"); print_verdicts(vs)
        s = summary(vs)
        if (s.wins, s.mean_Δ) > best_key
            best, best_key = p, (s.wins, s.mean_Δ)
        end
    end
    q = parse(Int, split(best, "q=")[2])
    write(joinpath(OUT, "pomcpow_choice.txt"), string(q))
    println("\nCHOSEN: $best  (wins $(best_key[1])/25, mean Δ $(round(best_key[2], digits=3)))")

elseif mode == "search"
    # ---- observation-noise grid on the search cells -------------------------------------------
    n_seeds = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
    q = parse(Int, strip(read(joinpath(OUT, "pomcpow_choice.txt"), String)))
    seeds = collect(4001:(4000 + n_seeds))
    configs = vcat([pomcpow_name(q) => (p -> make_pomcpow(p; rollout=:timing, tree_queries=q))], heuristic_configs())
    settings = [(rσ, dm) for rσ in RECOVERY_SIGMA for dm in DEPOSIT_MULT]
    done = length(ARGS) >= 3 ? read_done(ARGS[3]) : Dict()
    results = run_all(settings, SEARCH_CELLS, configs, seeds; label="NOISE SEARCH q=$q", done=done)
    write_csv(joinpath(OUT, "noise_search_q$(q).csv"), results)
    hnames = first.(heuristic_configs())
    println("\nMap of POMCPOW − best heuristic over the search cells")
    @printf("  %-8s %-9s %6s %6s %8s %8s\n", "rec σ", "dep ×", "wins", ">2SE", "clear L", "mean Δ")
    best = nothing; best_key = (-1, -Inf)
    for st in settings
        global best, best_key
        vs = verdicts(SEARCH_CELLS, pomcpow_name(q), hnames, (c, n) -> [results[(st, c, n, s)].disc for s in seeds])
        s = summary(vs)
        @printf("  %-8.2f %-9.1f %3d/%d %6d %8d %+8.3f\n", st..., s.wins, length(SEARCH_CELLS), s.sig, s.losses_sig, s.mean_Δ)
        if (s.wins, s.mean_Δ) > best_key
            best, best_key = st, (s.wins, s.mean_Δ)
        end
    end
    write(joinpath(OUT, "noise_choice.txt"), "$(best[1]) $(best[2])")
    println("\nCHOSEN: recovery σ = $(best[1]), deposit noise ×$(best[2])")

elseif mode == "confirm"
    # ---- chosen setting on all 25 cells with fresh seeds --------------------------------------
    n_seeds = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10
    γ       = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : BASE.γ
    q = parse(Int, strip(read(joinpath(OUT, "pomcpow_choice.txt"), String)))
    rσ, dm = parse.(Float64, split(strip(read(joinpath(OUT, "noise_choice.txt"), String))))
    seeds = collect(5001:(5000 + n_seeds))
    configs = vcat([pomcpow_name(q) => (p -> make_pomcpow(p; rollout=:timing, tree_queries=q))], heuristic_configs())
    st = (rσ, dm)
    results = run_all([st], ALL_CELLS, configs, seeds; γ=γ, label="CONFIRM q=$q rσ=$rσ dep×$dm (fresh seeds)")
    write_csv(joinpath(OUT, "confirm_q$(q)_r$(rσ)_d$(dm)_g$(γ).csv"), results)
    println("\nPOMCPOW − best heuristic, all 25 cells, fresh seeds $(first(seeds))–$(last(seeds)), γ=$γ")
    print_verdicts(verdicts(ALL_CELLS, pomcpow_name(q), first.(heuristic_configs()),
                            (c, n) -> [results[(st, c, n, s)].disc for s in seeds]))
else
    error("unknown mode $mode")
end
