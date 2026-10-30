#!/bin/zsh
# After the no-controls final runs finish: set the selected c in the exploration-controls copy and launch p1,2,3,4,6.
cd /home/edmondsa/LiPOMDPs.jl
LOG=outputs/ucb_check/pipeline.log
until grep -q "pipeline done\|ABORT" $LOG 2>/dev/null; do sleep 60; done
if grep -q "ABORT" $LOG; then echo "[$(date)] [explorectl] not launched: first pipeline aborted" >> $LOG; exit 1; fi
echo "[$(date)] [explorectl] waiting for the no-controls final runs to finish" >> $LOG
while pgrep -f "[s]cripts/final_results_best_pomcpow.jl " > /dev/null; do sleep 120; done
C=$(cat outputs/ucb_check/best_c.txt)
echo "[$(date)] [explorectl] no-controls runs finished; setting c = $C in the exploration-controls copy" >> $LOG
python3 - "$C" <<'PY' >> $LOG 2>&1 || { echo "[$(date)] [explorectl] ABORT: edit failed; nothing launched" >> $LOG; exit 1; }
import sys
c = sys.argv[1]; p = "scripts/final_results_best_pomcpow_explorectl.jl"; s = open(p).read()
for old, new in [("criterion = MaxUCB(0.5),", f"criterion = MaxUCB({c}.0),"),
                 ("_best_pomcpow_q10k_timing_0.5ucb$(variant).csv", f"_best_pomcpow_q10k_timing_{c}ucb$(variant).csv")]:
    assert s.count(old) == 1, f"pattern not found exactly once: {old}"
    s = s.replace(old, new)
open(p, "w").write(s)
print(f"[explorectl] edited {p}: MaxUCB({c}.0), output ..._{c}ucb_infogate_cheapest.csv")
PY
for pm in 1 2 3 4 6; do
    setsid nohup julia --project=. -t 16 scripts/final_results_best_pomcpow_explorectl.jl $pm > outputs/test_train_final_results/final_explorectl_${C}ucb_p$pm.log 2>&1 < /dev/null &
done
sleep 120
echo "[$(date)] [explorectl] launched: $(pgrep -fc '[f]inal_results_best_pomcpow_explorectl.jl') processes" >> $LOG
grep -h "Planning models:\|ERROR" outputs/test_train_final_results/final_explorectl_${C}ucb_p*.log >> $LOG
echo "[$(date)] [explorectl] pipeline done" >> $LOG
