"""
Combined POMCPOW grid: UCB constant × max depth × widening constant × proposer, all with the
certainty-equivalent leaf value (rollout = :mean_path, see scripts/pomcpow_components.jl).

Stages (paired common-random-number scenarios, evaluation on historical prices, TRAINING seeds only;
test seeds 2001+ stay untouched for final_results_best_pomcpow.jl):
  screen   seeds 1001–1010, cells p4/p6 × α ∈ {0.5, 1}: every grid config + references
  confirm  seeds 1011–1030 (fresh, so the pick is not tuned to the screening noise), cells p1..p6 at
           α ∈ {0.25, 0.5, 1}: the top `TOP_K` screen configs by pooled mean + references
  summary  re-print a summary from the CSV (works on a partial run)

  julia --project=. -t 40 scripts/test_pomcpow_grid.jl screen
  julia --project=. -t 80 scripts/test_pomcpow_grid.jl screen_mismatch   (second machine; p1/p2/p3 cells)
  julia --project=. -t 40 scripts/test_pomcpow_grid.jl confirm
  julia --project=. scripts/test_pomcpow_grid.jl summary screen

Each finished episode is appended to outputs/pomcpow_grid/<stage>.csv at once; rerunning a stage skips
episodes already in the file (resume after a crash).
"""

include(joinpath(@__DIR__, "pomcpow_tuning_harness.jl"))

const STAGE   = length(ARGS) >= 1 ? ARGS[1] : "screen"
const SMOKE   = get(ENV, "GRID_SMOKE", "0") == "1"     # tiny run to check the script end to end
const Q       = SMOKE ? 200 : 10_000
const TOP_K   = SMOKE ? 2 : 6
const OUTDIR  = SMOKE ? get(ENV, "GRID_SMOKE_DIR", "/tmp/pomcpow_grid_smoke") : "outputs/pomcpow_grid"
mkpath(OUTDIR)

# ---------------------------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------------------------
# UCB c ∈ {0.1..20} and depth ≤ 25 were already swept with the stochastic rollout (outputs/ucb_check,
# outputs/pomcpow_tuning); here they are re-screened only in combination with the new leaf value and
# proposers, which change the value noise the UCB constant has to trade off against. With the
# mean-path leaf valued to the horizon, `max_depth` is the TREE depth only.
const UCB_CS    = [1.0, 4.0, 8.0]
const DEPTHS    = [15, 31]
const PROPOSERS = [(:timing_first, 5.0), (:timing_first, 2.0), (:timing_first, 1.0),
                   (:timing_tree, 5.0), (:timing_tree, 1.0)]
prop_name(p) = p == :timing_first ? "TAfirst" : "TAtree"
grid_name(c, d, p, k) = "MP c=$(Int(c)) d=$d $(prop_name(p)) k=$(Int(k))"

const GRID = Dict{String,Function}()
for c in UCB_CS, d in DEPTHS, (p, k) in PROPOSERS
    GRID[grid_name(c, d, p, k)] = P -> make_pomcpow(P; rollout=:mean_path, ucb_c=c, max_depth=d,
                                                    next_action=p, k_action=k, tree_queries=Q)
end
const REFERENCES = Dict{String,Function}(
    # the configuration in final_results_best_pomcpow.jl before this study
    "BASE TArollout c=4 d=25 random k=5" => P -> make_pomcpow(P; rollout=:timing, ucb_c=4.0, max_depth=25, tree_queries=Q),
    "TA"     => P -> TimingAwarePolicy(P),
    "OSL-TA" => P -> OneStepLookaheadPolicy(P, n_rollouts=10, debug=false, rollout_policy=TimingAwarePolicy(P)),
)
make_policy(name, P) = haskey(GRID, name) ? GRID[name](P) : REFERENCES[name](P)

# ---------------------------------------------------------------------------------------------
# CSV streaming
# ---------------------------------------------------------------------------------------------
const HEADER = "planning_model,alpha,policy,seed,discounted_reward,explorations,mines_opened,open_years,secs"

function read_results(path)
    res = Dict{Tuple{Int,Float64,String,Int},NamedTuple}()
    isfile(path) || return res
    for (i, line) in enumerate(eachline(path))
        i == 1 && continue
        f = split(line, ',')
        length(f) == 9 || continue
        res[(parse(Int, f[1]), parse(Float64, f[2]), String(f[3]), parse(Int, f[4]))] =
            (disc=parse(Float64, f[5]), explores=parse(Int, f[6]), mines=parse(Int, f[7]),
             opens=String(f[8]), secs=parse(Float64, f[9]))
    end
    return res
end

function run_detail(policy, planning, evaluation, updater, seed)
    t0 = time()
    Random.seed!(seed)
    s, b = create_initial_belief(planning)
    disc = 0.0; explores = 0; opens = fill(0, planning.n_deposits)
    for t in 1:planning.time_horizon
        a = action(policy, b)
        explores += count(==(1), a.type)
        a_exec = LiPOMDPs.effective_action(evaluation, s.have_opened, a)
        for i in 1:planning.n_deposits
            a_exec.type[i] == 2 && !s.have_opened[i] && (opens[i] = t)
        end
        Random.seed!(hash((seed, t)))
        env = MersenneTwister(hash((seed, t, :env)))
        sp, o, r = gen(evaluation, s, a, env)
        disc += evaluation.γ^(t - 1) * r
        isterminal(evaluation, sp) && break
        b = update(updater, b, a, o)
        s = sp
    end
    return (disc=disc, explores=explores, mines=count(>(0), opens), opens=join(opens, " "), secs=time() - t0)
end

function run_stage(path, names, cells, seeds)
    done = read_results(path)
    isfile(path) || open(io -> println(io, HEADER), path, "w")
    jobs = [(pm, α, n, s) for n in names for (pm, α) in cells for s in seeds if !haskey(done, (pm, α, n, s))]
    # slowest (stochastic-rollout) jobs first so the tail of the run stays parallel
    sort!(jobs, by = j -> startswith(j[3], "BASE") ? 0 : startswith(j[3], "OSL") ? 1 : 2)
    println("$(length(jobs)) episodes to run ($(length(done)) already done) on $(Threads.nthreads()) threads")
    lk = ReentrantLock(); t0 = time(); n_done = Threads.Atomic{Int}(0)
    Threads.@threads :greedy for (pm, α, n, s) in jobs
        P = LiPOMDP(price_model_type=pm, alpha=α); E = LiPOMDP(price_model_type=6, alpha=α)
        r = run_detail(make_policy(n, P), P, E, LiBeliefUpdater(P=P), s)
        lock(lk) do
            open(path, "a") do io
                println(io, join([pm, α, n, s, r.disc, r.explores, r.mines, r.opens, round(r.secs, digits=1)], ","))
            end
        end
        k = Threads.atomic_add!(n_done, 1) + 1
        k % 50 == 0 && @printf("  %d/%d episodes, %.0f min\n", k, length(jobs), (time() - t0) / 60)
    end
    @printf("stage finished in %.0f min\n", (time() - t0) / 60)
end

# ---------------------------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------------------------
se(d) = length(d) > 1 ? std(d) / sqrt(length(d)) : NaN

"""Pooled paired summary over the (cell, seed) pairs every listed policy has finished."""
function summarize(path; io=stdout)
    res = read_results(path)
    isempty(res) && return println(io, "no results in $path")
    names = unique(k[3] for k in keys(res))
    pairs = unique((k[1], k[2], k[4]) for k in keys(res))
    complete = [(pm, α, s) for (pm, α, s) in pairs
                if haskey(res, (pm, α, "TA", s)) && haskey(res, (pm, α, "OSL-TA", s))]
    v(n) = [haskey(res, (pm, α, n, s)) ? res[(pm, α, n, s)].disc : NaN for (pm, α, s) in complete]
    ta, osl = v("TA"), v("OSL-TA")
    rows = []
    for n in names
        x = v(n); m = .!isnan.(x)
        count(m) == 0 && continue
        push!(rows, (n=n, k=count(m), mean=mean(x[m]), dta=mean((x .- ta)[m]), sta=se((x .- ta)[m]),
                     dosl=mean((x .- osl)[m]), sosl=se((x .- osl)[m]),
                     expl=mean(res[(pm, α, n, s)].explores for (pm, α, s) in complete if haskey(res, (pm, α, n, s))),
                     secs=mean(res[(pm, α, n, s)].secs for (pm, α, s) in complete if haskey(res, (pm, α, n, s)))))
    end
    sort!(rows, by = r -> -r.dosl)
    cells = sort(unique((pm, α) for (pm, α, _) in complete))
    println(io, "\n$(path): $(length(complete)) paired (cell, seed) scenarios; cells $(cells)")
    @printf(io, "  %-36s %4s %7s %16s %16s %6s %6s\n", "config", "n", "mean", "Δ vs TA", "Δ vs OSL-TA", "expl", "s/ep")
    for r in rows
        @printf(io, "  %-36s %4d %7.3f %+8.3f ±%5.3f %+8.3f ±%5.3f %6.1f %6.0f\n",
                r.n, r.k, r.mean, r.dta, r.sta, r.dosl, r.sosl, r.expl, r.secs)
    end
    # per-cell means for the top configs
    top = [r.n for r in rows[1:min(10, end)]]
    println(io, "\n  per-cell mean discounted reward (top 10 + references):")
    shown = unique(vcat(top, [n for n in ("BASE TArollout c=4 d=25 random k=5", "TA", "OSL-TA") if n in names]))
    @printf(io, "  %-36s", "config"); for (pm, α) in cells; @printf(io, "  p%d α=%-4s", pm, α); end; println(io)
    for n in shown
        @printf(io, "  %-36s", n)
        for (pm, α) in cells
            x = [res[(pm, α, n, s)].disc for (p2, a2, s) in complete if (p2, a2) == (pm, α) && haskey(res, (pm, α, n, s))]
            @printf(io, "  %9.3f", isempty(x) ? NaN : mean(x))
        end
        println(io)
    end
    # factor marginals over the grid
    gridrows = [r for r in rows if startswith(r.n, "MP ")]
    if !isempty(gridrows)
        println(io, "\n  grid marginals (mean Δ vs OSL-TA averaged over the other factors):")
        for (label, key) in (("UCB c", r -> match(r"c=(\d+)", r.n)[1]), ("max depth", r -> match(r"d=(\d+)", r.n)[1]),
                             ("proposer, k", r -> match(r"(TA\w+ k=\d+)", r.n)[1]))
            groups = Dict{String,Vector{Float64}}()
            for r in gridrows; push!(get!(groups, key(r), Float64[]), r.dosl); end
            println(io, "    $label: ", join(["$(g)=$(@sprintf("%+.3f", mean(groups[g])))" for g in sort(collect(keys(groups)), by = x -> something(tryparse(Int, x), 0))], "  "))
        end
    end
    return rows
end

# ---------------------------------------------------------------------------------------------
if STAGE == "screen"
    path = joinpath(OUTDIR, "screen.csv")
    names = vcat(sort(collect(keys(GRID))), collect(keys(REFERENCES)))
    SMOKE && (names = vcat(names[[1, 9, 20, 30]], collect(keys(REFERENCES))))
    run_stage(path, names, [(4, 0.5), (4, 1.0), (6, 0.5), (6, 1.0)], SMOKE ? [1001] : collect(1001:1010))
    open(io -> summarize(path; io=io), joinpath(OUTDIR, "screen_summary.txt"), "w")
    summarize(path)
elseif STAGE == "screen_mismatch"
    # same grid on the planning-model-mismatch cells (static / linear / exponential planning), so the
    # pick is not tuned to GBM and historical planning only; runs on a second machine
    path = joinpath(OUTDIR, "screen_mismatch.csv")
    names = vcat(sort(collect(keys(GRID))), collect(keys(REFERENCES)))
    run_stage(path, names, [(1, 0.5), (1, 1.0), (2, 0.5), (2, 1.0), (3, 0.5), (3, 1.0)], collect(1001:1010))
    open(io -> summarize(path; io=io), joinpath(OUTDIR, "screen_mismatch_summary.txt"), "w")
    summarize(path)
elseif STAGE == "confirm"
    # rank by the pooled paired gap to OSL-TA over both screens (when the mismatch screen exists)
    rows = summarize(joinpath(OUTDIR, "screen.csv"); io=devnull)
    if isfile(joinpath(OUTDIR, "screen_mismatch.csv"))
        rows2 = summarize(joinpath(OUTDIR, "screen_mismatch.csv"); io=devnull)
        d2 = Dict(r.n => r for r in rows2)
        rows = [(n=r.n, dosl=(r.dosl * r.k + (haskey(d2, r.n) ? d2[r.n].dosl * d2[r.n].k : 0.0)) / (r.k + (haskey(d2, r.n) ? d2[r.n].k : 0)))
                for r in rows]
        sort!(rows, by = r -> -r.dosl)
    end
    top = [r.n for r in rows if startswith(r.n, "MP ")][1:TOP_K]
    println("confirming top $TOP_K screen configs: ", top)
    path = joinpath(OUTDIR, "confirm.csv")
    cells = [(pm, α) for pm in (1, 2, 3, 4, 6) for α in (0.25, 0.5, 1.0)]
    SMOKE && (cells = cells[1:2])
    run_stage(path, vcat(top, collect(keys(REFERENCES))), cells, SMOKE ? [1011] : collect(1011:1030))
    open(io -> summarize(path; io=io), joinpath(OUTDIR, "confirm_summary.txt"), "w")
    summarize(path)
elseif STAGE == "summary"
    summarize(joinpath(OUTDIR, (length(ARGS) >= 2 ? ARGS[2] : "screen") * ".csv"))
else
    error("unknown stage $STAGE")
end
