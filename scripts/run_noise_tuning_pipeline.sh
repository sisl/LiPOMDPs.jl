#!/usr/bin/env bash
# Waits for the compute sweeps and the exploration-cost run, then runs pick → search → confirm.
cd "$(dirname "$0")/.." || exit 1
L=outputs/pomcpow_vs_heuristics
for log in $L/compute_sweep_full_e6.log $L/compute_sweep_q20k_e6.log $L/exploration_cost_e6.log; do
    until [ -f "$log" ] && grep -q "^exit" "$log"; do sleep 60; done
done
O=outputs/noise_tuning
julia --project -t 72 scripts/tune_observation_noise.jl pick       > $O/pick.log 2>&1    && \
julia --project -t 72 scripts/tune_observation_noise.jl search 10  > $O/search.log 2>&1  && \
julia --project -t 72 scripts/tune_observation_noise.jl confirm 10 > $O/confirm.log 2>&1
echo "pipeline exit $?" >> $O/confirm.log
