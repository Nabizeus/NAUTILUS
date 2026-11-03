#!/bin/bash
# monitor_ftorch.sh -- follow a pipeline job live.
#
#   bash monitor_ftorch.sh            # follow the most recently submitted job
#   bash monitor_ftorch.sh 47076764   # follow a specific job
#
# Shows the queue state while the job waits, then prints the job's output as
# it is written, and finishes with a summary when the job ends.
# Ctrl-C stops the monitoring only; the job keeps running.

cd "$(dirname "$0")" || exit 1
JOB="${1:-$(squeue -h -u "$USER" -n ftorch_pipeline -o %i --sort=-i | head -1)}"
if [ -z "${JOB}" ]; then
    JOB=$(ls -t server/ftorch_pipeline_*.out 2>/dev/null | head -1 | sed 's/.*ftorch_pipeline_\(.*\)\.out/\1/')
fi
[ -n "${JOB}" ] || { echo "No ftorch_pipeline job found. Submit one: sbatch sbatch_ftorch_pipeline.job"; exit 1; }
OUT="server/ftorch_pipeline_${JOB}.out"
ERR="server/ftorch_pipeline_${JOB}.err"
echo "Monitoring job ${JOB}  (output file: ${OUT})"

while [ ! -f "${OUT}" ] && squeue -h -j "${JOB}" 2>/dev/null | grep -q .; do
    echo "$(date +%H:%M:%S)  waiting in queue: $(squeue -h -j "${JOB}" -o '%T, reason: %R')"
    sleep 10
done

if [ -f "${OUT}" ]; then
    tail -n +1 -f "${OUT}" &
    TAIL_PID=$!
    trap 'kill ${TAIL_PID} 2>/dev/null; exit 0' INT
    while squeue -h -j "${JOB}" 2>/dev/null | grep -q .; do sleep 5; done
    sleep 3
    kill ${TAIL_PID} 2>/dev/null
fi

echo
echo "================ SUMMARY for job ${JOB} ================"
sacct -j "${JOB}" -X -n -o JobID,State,ExitCode,Elapsed 2>/dev/null
grep -h -E "^ ?(PASS|FAIL)|ERROR" "${OUT}" 2>/dev/null
if [ -s "${ERR}" ] && grep -q -i -E "error|traceback" "${ERR}"; then
    echo "--- last lines of ${ERR} ---"
    grep -v -E "^(load|unload|remove|Set INTEL|CPLUS_INCLUDE|PKG_CONFIG)" "${ERR}" | tail -15
fi
