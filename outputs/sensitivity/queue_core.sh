#!/bin/zsh
cd /home/edmondsa/LiPOMDPs.jl
while ps -eo args | grep "bin/julia" | grep -q "structure_fixes.jl 20 10000 gbm8"; do sleep 120; done
julia --project=. -t 40 scripts/sensitivity_analysis.jl 20 0.5 6 10000 core uncert > outputs/sensitivity/run_p6_n20_core_uncert_rec20-25_osl5.log 2>&1
