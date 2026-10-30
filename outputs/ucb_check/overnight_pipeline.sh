#!/bin/zsh
# Overnight: UCB check (no controls) → pick best c → set it in final_results_best_pomcpow.jl → launch p1,2,3,4,6.
cd /home/edmondsa/LiPOMDPs.jl
LOG=outputs/ucb_check/pipeline.log
echo "[$(date)] starting UCB check (c = 1, 2, 4; no controls; training seeds 1001-1025)" >> $LOG
julia --project -t 80 outputs/ucb_check/test_ucb_nocontrols.jl > outputs/ucb_check/ucb_nocontrols.log 2>&1
grep -E "episodes|planning p|disc|POOLED|c=[124] |SELECTED" outputs/ucb_check/ucb_nocontrols.log > outputs/ucb_check/summary_nocontrols.txt
if [[ ! -s outputs/ucb_check/best_c.txt ]]; then
    echo "[$(date)] ABORT: UCB check failed (no best_c.txt); nothing launched. See ucb_nocontrols.log" >> $LOG; exit 1
fi
C=$(cat outputs/ucb_check/best_c.txt)
echo "[$(date)] selected c = $C" >> $LOG
python3 - "$C" <<'PY' >> $LOG 2>&1 || { echo "[$(date)] ABORT: edit failed; nothing launched" >> $LOG; exit 1; }
import sys
c = sys.argv[1]; p = "scripts/final_results_best_pomcpow.jl"; s = open(p).read()
for old, new in [("criterion = MaxUCB(0.5),", f"criterion = MaxUCB({c}.0),"),
                 ("_best_pomcpow_q10k_timing_0.5ucb.csv", f"_best_pomcpow_q10k_timing_{c}ucb.csv")]:
    assert s.count(old) == 1, f"pattern not found exactly once: {old}"
    s = s.replace(old, new)
open(p, "w").write(s)
print(f"edited {p}: MaxUCB({c}.0), output ..._{c}ucb.csv")
PY
grep -n "criterion = MaxUCB\|ucb.csv" scripts/final_results_best_pomcpow.jl >> $LOG
for pm in 1 2 3 4 6; do
    setsid nohup julia --project=. -t 16 scripts/final_results_best_pomcpow.jl $pm > outputs/test_train_final_results/final_${C}ucb_p$pm.log 2>&1 < /dev/null &
done
sleep 120
echo "[$(date)] launched final runs: $(pgrep -fc '[s]cripts/final_results_best_pomcpow.jl ') processes" >> $LOG
grep -h "Planning models:\|ERROR" outputs/test_train_final_results/final_${C}ucb_p*.log >> $LOG
echo "[$(date)] pipeline done" >> $LOG
