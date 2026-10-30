using Test
using LiPOMDPs
using POMDPs

@testset "LiPOMDPs" begin

    @testset "Package loads" begin
        @test true  # if we got here, the package loaded
    end

    @testset "Struct construction" begin
        s = State(
            deposit = [100_000.0, 100_000.0, 500_000.0, 500_000.0],
            time = 1.0,
            price = 10_000.0,
            company_demand = 5_000.0,
            total_mined = 0.0,
            have_opened = [false, false, false, false],
            is_depleted = [false, false, false, false],
            restored_mines = [false, false, false, false]
        )
        @test s.time == 1.0
        @test length(s.deposit) == 4

        a = Action(site = [0, 0, 0, 0], type = [0, 0, 0, 0])
        @test a.site == [0, 0, 0, 0]
        @test a.type == [0, 0, 0, 0]

        o = Observation(deposits = [1.0, 2.0, 3.0, 4.0],
                        observed_price = 10_000.0,
                        observed_demand = 5_000.0)
        @test o.observed_price == 10_000.0
    end

    @testset "LiPOMDP model" begin
        pomdp = LiPOMDP()
        @test pomdp.n_deposits == 4
        @test pomdp.time_horizon == 31   # 1995–2025
        @test 0.0 < POMDPs.discount(pomdp) < 1.0

        # Custom parameters
        pomdp2 = LiPOMDP(price_model_type = 4, alpha = 0.8)
        @test pomdp2.price_model_type == 4
        @test pomdp2.alpha == 0.8
    end

    @testset "Initial belief and state" begin
        pomdp = LiPOMDP()
        s0, belief = create_initial_belief(pomdp)

        @test length(s0.deposit) == 4
        @test s0.time == 1.0
        @test all(.!s0.have_opened)
        @test all(.!s0.is_depleted)
        @test belief.time == 1.0
        @test length(belief.deposits_distribution) == 4
    end

    @testset "Action space" begin
        pomdp = LiPOMDP()
        acts = POMDPs.actions(pomdp)
        @test length(acts) > 0
        @test all(a -> isa(a, Action), acts)
    end

    @testset "Terminal check" begin
        pomdp = LiPOMDP()
        s = State(
            deposit = [100_000.0, 100_000.0, 500_000.0, 500_000.0],
            time = 1.0,
            price = 10_000.0,
            company_demand = 5_000.0,
            total_mined = 0.0,
            have_opened = [false, false, false, false],
            is_depleted = [false, false, false, false],
            restored_mines = [false, false, false, false]
        )
        @test !POMDPs.isterminal(pomdp, s)

        # Past time horizon → terminal
        s_late = State(
            deposit = [100_000.0, 100_000.0, 500_000.0, 500_000.0],
            time = 100.0,
            price = 10_000.0,
            company_demand = 5_000.0,
            total_mined = 0.0,
            have_opened = [false, false, false, false],
            is_depleted = [false, false, false, false],
            restored_mines = [false, false, false, false]
        )
        @test POMDPs.isterminal(pomdp, s_late)
    end

    @testset "Pricing models" begin
        for model_type in [1, 2, 3, 4]
            pomdp = LiPOMDP(price_model_type = model_type)
            @test pomdp.price_model_type == model_type
            s0, _ = create_initial_belief(pomdp)
            @test s0.price > 0.0
        end
    end

    # Modules LiPOMDPs already loads; avoids extra test dependencies
    Random = LiPOMDPs.Random
    Distributions = LiPOMDPs.Distributions
    mean, std = Distributions.mean, Distributions.std
    MT = Random.MersenneTwister
    noop(P) = Action(site = collect(1:P.n_deposits), type = zeros(Int, P.n_deposits))

    @testset "Deposit prior: LogNormal with nominal mean and CV = uncertainty factor" begin
        P = LiPOMDP()
        d = POMDPs.initialstate(P)
        for i in 1:P.n_deposits
            nominal = P.deposit_types[i] ? P.init_deposit_size_dle : P.init_deposit_size_mining
            cv      = P.deposit_types[i] ? P.dle_uncertainty_factor : P.mining_uncertainty_factor
            @test mean(d.deposit_dists[i]) ≈ nominal
            @test std(d.deposit_dists[i]) / nominal ≈ cv
        end
        s, b = create_initial_belief(P)
        @test all(s.deposit .> 0)
        @test all(mean(b.deposits_distribution[i]) ≈ mean(d.deposit_dists[i]) for i in 1:P.n_deposits)
    end

    @testset "GBM price: one-step moments, driven by the current price" begin
        P = LiPOMDP(price_model_type = 4)
        a = noop(P)
        step_mean(p0) = begin
            s = State(deposit = fill(1e5, 4), time = 5.0, price = p0, company_demand = 1e4, total_mined = 0.0,
                      have_opened = falses(4), is_depleted = falses(4), restored_mines = falses(4))
            rng = MT(1)
            mean(LiPOMDPs.update_price(P, s, a; rng = rng) for _ in 1:20_000)
        end
        # E[S(t+1)] = S(t)·exp(μ); the old code used p₀ instead of S(t), so 20k and 10k gave the same mean
        @test isapprox(step_mean(10_000.0), 10_000.0 * exp(P.gbm_drift), rtol = 0.02)
        @test isapprox(step_mean(20_000.0), 20_000.0 * exp(P.gbm_drift), rtol = 0.02)
    end

    @testset "gen is reproducible from its rng" begin
        P = LiPOMDP(price_model_type = 6)
        s, _ = create_initial_belief(P)
        a = Action(site = [1, 2, 3, 4], type = [1, 2, 0, 0])
        g1 = POMDPs.gen(P, s, a, MT(42))
        g2 = POMDPs.gen(P, s, a, MT(42))
        @test g1.sp == g2.sp && g1.o == g2.o && g1.r == g2.r
    end

    @testset "Belief tracks price, demand and an open mine's deposit" begin
        for pm in (4, 6)
            P = LiPOMDP(price_model_type = pm)
            up = LiBeliefUpdater(P = P)
            Random.seed!(3)
            s, b = create_initial_belief(P)
            rng = MT(3)
            open1 = Action(site = [1, 2, 3, 4], type = [2, 0, 0, 0])
            for t in 1:28
                a = t == 1 ? open1 : noop(P)
                sp, o, r = POMDPs.gen(P, s, a, rng)
                b = POMDPs.update(up, b, a, o)
                @test abs(mean(b.price_distribution) - sp.price) < 5 * P.σ_obs_p
                @test abs(mean(b.demand_distribution) - sp.company_demand) < 5 * P.σ_obs_demand
                if !sp.is_depleted[1]
                    @test abs(mean(b.deposits_distribution[1]) - sp.deposit[1]) < 5_000
                end
                @test b.have_opened == sp.have_opened
                s = sp
            end
            @test b.time == 29.0
        end
    end

    @testset "Filter uses the observation noise the generator uses" begin
        P = LiPOMDP(price_model_type = 1)
        up = LiBeliefUpdater(P = P)
        s, b = create_initial_belief(P)
        explore1 = Action(site = [1, 2, 3, 4], type = [1, 0, 0, 0])
        o_dist = POMDPs.observation(P, s, explore1, s)
        σ_gen = std(o_dist.dists[1])                   # noise the environment draws for site 1
        @test σ_gen == LiPOMDPs.deposit_obs_std(P, 1, :explore)
        σ0 = std(b.deposits_distribution[1])
        sp, o, _ = POMDPs.gen(P, s, explore1, MT(1))
        b2 = POMDPs.update(up, b, explore1, o)
        expected = max(1 / sqrt(1 / σ0^2 + 1 / σ_gen^2), max(500.0, 0.01 * P.init_deposit_size_dle))
        @test b2.deposits_distribution[1].untruncated.σ ≈ expected   # σ before truncation at 0
        @test std(b2.deposits_distribution[2]) ≈ σ0                              # unexplored site unchanged
    end

    @testset "Historical prices and demand come from the same calendar year" begin
        P = LiPOMDP(price_model_type = 6)
        data = LiPOMDPs.lithium_data
        @test LiPOMDPs.get_p₀(P) == data[P.start_year][2]                     # 1995 price at t = 1
        s = State(deposit = fill(1e5, 4), time = 1.0, price = data[1995][2], company_demand = 1e4,
                  total_mined = 0.0, have_opened = falses(4), is_depleted = falses(4), restored_mines = falses(4))
        for t in 1:(P.time_horizon - 1)
            year = P.start_year + t                                           # calendar year of time t + 1
            @test LiPOMDPs.price_base(P, Float64(t)) == data[year][2]
            sp = LiPOMDPs.sample_transition(P, s, noop(P), MT(t))
            @test isapprox(sp.company_demand, data[year][1] * P.LCE_per_lithium * P.market_share_target, rtol = 0.25)
            s = sp
        end
        @test P.start_year + P.time_horizon - 1 == 2025
        @test data[2025] == (292_301.0, 9_000.0)        # 2025 production revised 2026-10-19
    end

    @testset "Opening year is construction: every tonne extracted is paid for" begin
        P = LiPOMDP(price_model_type = 6, alpha = 1.0)
        Random.seed!(11)
        s, _ = create_initial_belief(P)
        rng = MT(11)
        open_all = Action(site = [1, 2, 3, 4], type = [2, 2, 2, 2])
        sp, _, _ = POMDPs.gen(P, s, open_all, rng)
        @test sp.deposit == s.deposit                 # nothing mined while under construction
        @test sp.total_mined == 0.0
        paid = 0.0
        s = sp
        while !POMDPs.isterminal(P, s)
            paid += LiPOMDPs.calculate_total_production(P, s)[1]   # LCE `reward` pays for this step
            s, _, _ = POMDPs.gen(P, s, noop(P), rng)
        end
        @test s.total_mined ≈ paid                    # extracted (transition) == sold (reward)
    end

    @testset "Recovery: Beta prior, hidden state, learned from exploration and production" begin
        P = LiPOMDP(price_model_type = 1)
        d = POMDPs.initialstate(P)
        for i in 1:P.n_deposits
            @test mean(d.recovery_dists[i]) ≈ P.recovery[i]
            @test std(d.recovery_dists[i]) ≈ P.recovery_uncertainty[i]
        end
        Random.seed!(21)
        s, b = create_initial_belief(P)
        @test length(s.recovery) == P.n_deposits && all(0 .< s.recovery .<= 1)
        up = LiBeliefUpdater(P = P)
        rng = MT(21)
        explore1 = Action(site = [1, 2, 3, 4], type = [1, 0, 0, 0])
        sp, o, _ = POMDPs.gen(P, s, explore1, rng)
        b2 = POMDPs.update(up, b, explore1, o)
        @test length(o.recoveries) == P.n_deposits
        @test std(b2.recovery_distribution[1]) < std(b.recovery_distribution[1])   # explored site learns
        @test std(b2.recovery_distribution[2]) ≈ std(b.recovery_distribution[2])   # others unchanged
        # Known recovery (no uncertainty) keeps the old observation layout
        Pk = LiPOMDP(recovery_uncertainty = zeros(4))
        sk, bk = create_initial_belief(Pk)
        @test isempty(sk.recovery) && isempty(bk.recovery_distribution)
        @test isempty(POMDPs.gen(Pk, sk, explore1, MT(1)).o.recoveries)
    end

    @testset "Opex is per tonne of ore: lower recovery raises cost per tonne sold" begin
        P = LiPOMDP()
        base = State(deposit = fill(5e5, 4), time = 20.0, price = 10_000.0, company_demand = 1e6, total_mined = 0.0,
                     have_opened = [true, false, false, false], is_depleted = falses(4), restored_mines = falses(4),
                     recovery = copy(P.recovery))
        low = deepcopy(base); low.recovery[1] = 0.4
        a = noop(P)
        sold(s) = LiPOMDPs.calculate_total_production(P, s)[1]
        @test LiPOMDPs.compute_costs(P, base, a) / sold(base) ≈ P.dle_opex                    # nominal: quoted opex
        @test LiPOMDPs.compute_costs(P, low, a) / sold(low) ≈ P.dle_opex * 0.80 / 0.4         # per t LCE doubles
        @test sold(low) ≈ sold(base) / 2                                                      # same ore, half the LCE
    end

    @testset "Capital budget: opens beyond the budget are dropped everywhere" begin
        P = LiPOMDP(capital_budget = 2.5e9)    # DLE 1.23B, hard rock 1.512B
        Random.seed!(31)
        s, b = create_initial_belief(P)
        open_all = Action(site = [1, 2, 3, 4], type = [2, 2, 2, 2])
        eff = LiPOMDPs.effective_action(P, s.have_opened, open_all)
        @test eff.type == [2, 2, 0, 0]                       # 2.46B; adding a hard-rock mine would exceed 2.5B
        sp, o, r = POMDPs.gen(P, s, open_all, MT(31))
        @test sp.have_opened == [true, true, false, false]
        @test POMDPs.update(LiBeliefUpdater(P = P), b, open_all, o).have_opened == sp.have_opened
        @test r ≈ POMDPs.reward(P, s, eff)                   # only the executed capex is charged
        @test all(LiPOMDPs.committed_capex(P, a.type .== 2) <= P.capital_budget for a in POMDPs.actions(P, s))
        @test LiPOMDPs.effective_action(LiPOMDP(capital_budget = Inf), s.have_opened, open_all) == open_all
    end

    @testset "OneStepLookahead: feasible, deterministic given its rng" begin
        P = LiPOMDP(price_model_type = 1)
        _, b = create_initial_belief(P)
        a1 = POMDPs.action(OneStepLookaheadPolicy(P, n_rollouts = 5, rng = MT(7)), b)
        a2 = POMDPs.action(OneStepLookaheadPolicy(P, n_rollouts = 5, rng = MT(7)), b)
        @test a1 == a2
        @test all(t in 0:2 for t in a1.type)          # nothing to restore at t = 1
    end

    @testset "Mean-path transition is the average of the sampled transition" begin
        for pm in (4, 6)
            P = LiPOMDP(price_model_type = pm, alpha = 1.0)
            Random.seed!(5)
            s, _ = create_initial_belief(P)
            s.time = 10.0
            a = Action(site = [1, 2, 3, 4], type = [2, 0, 0, 0])
            m = LiPOMDPs.mean_transition(P, s, a)
            sps = [LiPOMDPs.sample_transition(P, s, a, MT(k)) for k in 1:20_000]
            @test isapprox(mean(sp.price for sp in sps), m.price, rtol = 0.01)
            @test isapprox(mean(sp.company_demand for sp in sps), m.company_demand, rtol = 0.01)
            @test m.have_opened == sps[1].have_opened && m.deposit == sps[1].deposit && m.time == s.time + 1
        end
        P = LiPOMDP(price_model_type = 4, alpha = 1.0)
        s, _ = create_initial_belief(P)
        @test LiPOMDPs.expected_price(P, s) == s.price * exp(P.gbm_drift)   # mean path, not the median
    end

    @testset "Mean-path value: deterministic, horizon-limited, equals the mean-path return" begin
        P = LiPOMDP(price_model_type = 6, alpha = 0.5)
        Random.seed!(7)
        s, _ = create_initial_belief(P)
        ta = TimingAwarePolicy(P)
        v1 = mean_path_value(P, ta, s, P.time_horizon)
        v2 = mean_path_value(P, ta, s, P.time_horizon)
        @test v1 == v2 && isfinite(v1)
        @test mean_path_value(P, ta, s, 0) == 0.0
        # replicate by hand
        total, disc, x = 0.0, 1.0, deepcopy(s)
        for _ in 1:P.time_horizon
            POMDPs.isterminal(P, x) && break
            a = POMDPs.action(ta, x)
            total += disc * POMDPs.reward(P, x, a)
            x = LiPOMDPs.mean_transition(P, x, a)
            disc *= P.γ
        end
        @test v1 ≈ total
        @test v1 > 0                                   # TA opens profitably on the historical path
        @test mean_path_value(P, ta, s, 5) != v1       # truncation changes the value
    end

end
