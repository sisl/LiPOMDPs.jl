"""
Fit the GBM price-model parameters (gbm_drift μ, gbm_volatility σ) to the historical lithium prices.

The model steps S(t+1) = S(t)·exp((μ − σ²/2) + σZ), so annual log-returns r_t = ln(S_t/S_{t−1})
have mean μ − σ²/2 and std σ. σ is estimated from 5%-winsorized returns to limit the influence of
the 2022 price spike. The drift is taken from the unwinsorized mean log-return, which equals
ln(S_end/S_start)/n and is therefore already insensitive to a spike that reverses, so that the
median simulated path matches the observed long-run growth: μ = mean(r) + σ²/2.

  julia --project=. scripts/fit_gbm.jl
"""

using LiPOMDPs, Statistics, Printf, Random

d = LiPOMDPs.lithium_data
yrs = sort(collect(keys(d)))
r = diff(log.([d[y][2] for y in yrs]))
n = length(r)

winsorize(x, q) = clamp.(x, quantile(x, [q, 1 - q])...)

r̄ = mean(r)
σ = std(winsorize(r, 0.05))
μ = r̄ + σ^2 / 2

@printf("%d annual log-returns, %d–%d\n", n, yrs[1], yrs[end])
@printf("mean log-return r̄ = %.4f (SE %.4f)\n", r̄, std(r) / sqrt(n))
@printf("σ (5%%-winsorized) = %.4f   (unwinsorized: %.4f)\n", σ, std(r))
@printf("μ = r̄ + σ²/2     = %.4f\n", μ)

# Bootstrap uncertainty of the same estimator: 30 points pin the drift down only loosely
rng = MersenneTwister(1)
boot = map(1:5000) do _
    x = r[rand(rng, 1:n, n)]
    s = std(winsorize(x, 0.05))
    (mean(x) + s^2 / 2, s)
end
@printf("bootstrap 90%% CI: μ [%.3f, %.3f], σ [%.3f, %.3f]\n",
        quantile(first.(boot), [0.05, 0.95])..., quantile(last.(boot), [0.05, 0.95])...)

p0 = d[yrs[1]][2]
T = yrs[end] - yrs[1]
@printf("implied %d price from %d: median %.0f, mean %.0f (observed %.0f)\n",
        yrs[end], yrs[1], p0 * exp((μ - σ^2 / 2) * T), p0 * exp(μ * T), d[yrs[end]][2])
