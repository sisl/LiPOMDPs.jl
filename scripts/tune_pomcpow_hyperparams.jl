"""
POMCPOW hyperparameter tuning with a train/test split.

Uses the query budget and observation-noise setting chosen by tune_observation_noise.jl
(outputs/noise_tuning/pomcpow_choice.txt and noise_choice.txt).

TRAIN  (cells SEARCH_CELLS, seeds 4001–4010) — tune one group at a time, keeping the best so far:
  stage 1  UCB constant scaled to the reward size: c = c_rel · R(α), R(α) = max(1.8α, 0.1)
           (returns grow roughly in proportion to α; c_rel = 1/0.9 ≈ the current c = 1 at α = 0.5)
  stage 2  observation progressive widening (k_observation, alpha_observation)
  stage 3  final root action: highest Q (POMCPOW default) vs most visits
  Selection metric: mean paired margin over the best heuristic across the train cells
  (ties: cells won). Heuristic results come from the noise search (same cells, seeds, noise).
  The untuned POMCPOW from the noise search is the stage-0 baseline.

TEST   (all 25 cells, FRESH seeds 6001–6010) — tuned and untuned POMCPOW against every heuristic.
  Report the test numbers; the train numbers are optimistic by construction.

  julia --project=. -t 72 scripts/tune_pomcpow_hyperparams.jl [n_seeds]
"""

include(joinpath(@__DIR__, "pomcpow_tuning_harness.jl"))
using CSV, DataFrames

const OUT          = "outputs/noise_tuning"
const EVAL_MODEL   = 6
const ALL_CELLS    = [(pm, α) for pm in (1, 2, 3, 4, 6) for α in (0.0, 0.25, 0.5, 0.75, 1.0)]
const TRAIN_CELLS  = [(1, 0.75), (2, 1.0), (3, 0.5), (3, 1.0), (4, 1.0), (6, 1.0)]   # = noise-search cells
const BASE = LiPOMDP()

n_seeds     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
train_seeds = collect(4001:(4000 + n_seeds))
test_seeds  = collect(6001:(6000 + n_seeds))
q           = parse(Int, strip(read(joinpath(OUT, "pomcpow_choice.txt"), String)))
rσ, dm      = parse.(Float64, split(strip(read(joinpath(OUT, "noise_choice.txt"), String))))
noise_kw    = (σ_obs_recovery_explore = rσ, σ_obs_d_dle = dm * BASE.σ_obs_d_dle, σ_obs_d_hard_rock = dm * BASE.σ_obs_d_hard_rock)
reward_size(α) = max(1.8α, 0.1)

heuristic_configs() = [
    "OneStep timing n=10"      => p -> OneStepLookaheadPolicy(p, n_rollouts=10, rollout_policy=TimingAwarePolicy(p)),
    "OneStep timing n=20"      => p -> OneStepLookaheadPolicy(p, n_rollouts=20, rollout_policy=TimingAwarePolicy(p)),
    "OneStep timing n=30"      => p -> OneStepLookaheadPolicy(p, n_rollouts=30, rollout_policy=TimingAwarePolicy(p)),
    "TimingAware"              => p -> TimingAwarePolicy(p),
    "OneStepLookahead"         => make_onestep,
    "DynamicProfitMaximizer"   => p -> DynamicProfitMaximizerPolicy(p),
    "DynamicEmissionMinimizer" => p -> DynamicEmissionMinimizerPolicy(p),
    "AlphaAware"               => p -> AlphaAwareRolloutPolicy(p),
    "ExploreOnly"              => p -> ExploreOnlyPolicy(p),
    "OneMineProfit"            => p -> ProfitMaximizerPolicy(p, top_k=1, times=[1]),
    "OneMineEmission"          => p -> EmissionMinimizerPolicy(p, top_k=1, times=[1]),
    "Random"                   => p -> RandomHeuristicPolicy(p, debug=false),
]
const HNAMES = first.(heuristic_configs())

# A POMCPOW setting; c_rel = nothing keeps the unscaled c = 1 of the default configuration
default_hp() = (c_rel = nothing, k_obs = 3.0, a_obs = 0.3, final = :maxq)
hp_name(hp) = "POMCPOW q=$q c=$(hp.c_rel === nothing ? "1(unscaled)" : "$(hp.c_rel)·R(α)") kobs=$(hp.k_obs) aobs=$(hp.a_obs) final=$(hp.final)"
pomcpow_from(hp) = p -> make_pomcpow(p; rollout=:timing, tree_queries=q,
                                     ucb_c = hp.c_rel === nothing ? 1.0 : hp.c_rel * reward_size(p.alpha),
                                     k_observation = hp.k_obs, alpha_observation = hp.a_obs,
                                     final_criterion = hp.final)

se(x) = length(x) > 1 ? std(x) / sqrt(length(x)) : NaN

function run_all(cells, configs, seeds; label="")
    jobs = [(c, name, seed) for (name, _) in configs for c in cells for seed in seeds]
    cfg = Dict(configs)
    results = Dict{Tuple{Tuple{Int,Float64},String,Int}, NamedTuple}()
    lk = ReentrantLock(); t0 = time()
    println("\n$label: $(length(jobs)) episodes on $(Threads.nthreads()) threads")
    Threads.@threads :greedy for (c, name, seed) in jobs
        planning   = LiPOMDP(; price_model_type=c[1],      alpha=c[2], noise_kw...)
        evaluation = LiPOMDP(; price_model_type=EVAL_MODEL, alpha=c[2], noise_kw...)
        r = run_scenario(cfg[name](planning), planning, evaluation, LiBeliefUpdater(P=planning), seed)
        lock(lk) do
            results[(c, name, seed)] = r
            @printf("  [%d/%d] %.0f min  p%d α=%.2f %s seed=%d  disc=%.3f expl=%d\n",
                    length(results), length(jobs), (time() - t0) / 60, c..., name, seed, r.disc, r.explores)
            flush(stdout)
        end
    end
    @printf("%d episodes in %.0f min\n", length(jobs), (time() - t0) / 60)
    return results
end

function write_csv(path, results, phase)
    open(path, "a") do io
        filesize(path) == 0 && println(io, "phase,planning_model,eval_model,alpha,policy,seed,discounted_reward,profit,emission_cost,co2_tonnes,met_demand_pct,mines_opened,capex,explorations")
        for ((c, name, seed), r) in sort(collect(results), by=first)
            println(io, join([phase, c[1], EVAL_MODEL, c[2], "\"$name\"", seed, r.disc, r.profit, r.emis, r.co2, r.met, r.mines, r.capex, r.explores], ","))
        end
    end
end

"""Mean paired margin over the best heuristic, cells won, clear wins/losses."""
function score(cells, seeds, disc, pname)
    Δs = map(cells) do c
        bh = argmax(h -> mean(disc(c, h)), HNAMES)
        Δ = disc(c, pname) .- disc(c, bh)
        (cell = c, best = bh, Δ = mean(Δ), se = se(Δ))
    end
    (mean_Δ = mean(v.Δ for v in Δs), wins = count(v -> v.Δ > 0, Δs),
     sig = count(v -> v.Δ > 2v.se, Δs), losses_sig = count(v -> v.Δ < -2v.se, Δs), cells = Δs)
end

# ---- TRAIN ---------------------------------------------------------------------------------------
println("POMCPOW HYPERPARAMETER TUNING  q=$q  noise: recovery σ=$rσ, deposit ×$dm  train seeds $(first(train_seeds))–$(last(train_seeds))")
search = DataFrame(CSV.File(joinpath(OUT, "noise_search_q$(q).csv")))
search = search[(search.recovery_sigma .== rσ) .& (search.deposit_mult .== dm), :]
train_disc = Dict{Tuple{Int,Float64,String,Int},Float64}(
    (r.planning_model, Float64(r.alpha), String(r.policy), r.seed) => r.discounted_reward for r in eachrow(search))
base_name = "POMCPOW timing q=$q"                       # untuned POMCPOW from the noise search
tdisc(c, name) = [train_disc[(c[1], c[2], name, s)] for s in train_seeds]

best_hp = default_hp()
best_name = base_name
best_score = score(TRAIN_CELLS, train_seeds, tdisc, base_name)
@printf("stage 0 (untuned): mean Δ %+.3f, wins %d/%d\n", best_score.mean_Δ, best_score.wins, length(TRAIN_CELLS))
csv_path = joinpath(OUT, "hyperparam_tuning_q$(q).csv")
# Resume: reuse train episodes already saved by an earlier, interrupted run
if isfile(csv_path)
    saved = DataFrame(CSV.File(csv_path))
    saved = saved[saved.phase .== "train", :]
    for r in eachrow(saved)
        train_disc[(r.planning_model, Float64(r.alpha), String(r.policy), r.seed)] = r.discounted_reward
    end
    println("resuming: $(nrow(saved)) train episodes reused from $csv_path")
end
complete(name) = all(haskey(train_disc, (c[1], c[2], name, s)) for c in TRAIN_CELLS for s in train_seeds)

stages = [
    ("stage 1: UCB c scaled to reward size", hp -> [merge(hp, (c_rel = c,)) for c in (0.25, 0.5, 1.0, 2.0)]),
    ("stage 2: observation widening",        hp -> [merge(hp, (k_obs = k, a_obs = a)) for (k, a) in ((1.0, 0.1), (2.0, 0.2), (6.0, 0.4))]),
    ("stage 3: final action",                hp -> [merge(hp, (final = :visits,))]),
]
for (label, candidates) in stages
    global best_hp, best_name, best_score
    cands = candidates(best_hp)
    todo = [hp_name(hp) => pomcpow_from(hp) for hp in cands if !complete(hp_name(hp))]
    res = isempty(todo) ? Dict() : run_all(TRAIN_CELLS, todo, train_seeds; label=label)
    isempty(res) || write_csv(csv_path, res, "train")
    for ((c, name, seed), r) in res
        train_disc[(c[1], c[2], name, seed)] = r.disc
    end
    println("\n$label (train cells):")
    @printf("  %-70s mean Δ %+.3f  wins %d/%d  (current best)\n", best_name, best_score.mean_Δ, best_score.wins, length(TRAIN_CELLS))
    for hp in cands
        s = score(TRAIN_CELLS, train_seeds, tdisc, hp_name(hp))
        @printf("  %-70s mean Δ %+.3f  wins %d/%d\n", hp_name(hp), s.mean_Δ, s.wins, length(TRAIN_CELLS))
        if (s.mean_Δ, s.wins) > (best_score.mean_Δ, best_score.wins)
            best_hp, best_name, best_score = hp, hp_name(hp), s
        end
    end
    println("  → keep: $best_name")
end
write(joinpath(OUT, "hyperparam_choice.txt"), best_name)

# ---- TEST ----------------------------------------------------------------------------------------
configs = vcat(["TUNED: " * best_name => pomcpow_from(best_hp), "UNTUNED: " * base_name => pomcpow_from(default_hp())],
               heuristic_configs())
res = run_all(ALL_CELLS, configs, test_seeds; label="TEST (fresh seeds $(first(test_seeds))–$(last(test_seeds)))")
write_csv(csv_path, res, "test")
xdisc(c, name) = [res[(c, name, s)].disc for s in test_seeds]
for p in first.(configs)[1:2]
    s = score(ALL_CELLS, test_seeds, xdisc, p)
    println("\n$p — all 25 cells, fresh seeds")
    for v in s.cells
        tag = (v.Δ == 0 && v.se == 0) ? "tie" : v.Δ > 2v.se ? "WIN (>2SE)" : v.Δ > 0 ? "win (noise)" : v.Δ > -2v.se ? "loss (noise)" : "LOSS (>2SE)"
        @printf("  p%d α=%-5.2f  best heuristic %-26s  Δ %+7.3f ± %.3f  %s\n", v.cell..., v.best, v.Δ, v.se, tag)
    end
    @printf("  ahead in %d/25 (%d beyond 2 SE); clear losses %d; mean Δ %+.3f\n", s.wins, s.sig, s.losses_sig, s.mean_Δ)
end
println("\nCSV: $csv_path")
