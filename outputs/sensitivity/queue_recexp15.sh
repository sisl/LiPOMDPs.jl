#!/bin/zsh
# after the oceanside finals finish, rerun both sensitivity sets on the σ_rec_explore = 0.15 model
cd /home/edmondsa/LiPOMDPs.jl
while ps -eo args | grep "bin/julia" | grep -q "final_results_best_pomcpow.jl [123]"; do sleep 300; done
julia --project=. -t 40 scripts/sensitivity_analysis.jl 20 0.5 6 10000 core uncert > outputs/sensitivity/run_p6_n20_core_uncert_rec20-25_osl5_recexp15.log 2>&1
julia --project=. -t 60 scripts/sensitivity_analysis.jl 20 0.5 6 10000 appendix > outputs/sensitivity/run_p6_n20_appendix_rec20-25_osl5_recexp15.log 2>&1
