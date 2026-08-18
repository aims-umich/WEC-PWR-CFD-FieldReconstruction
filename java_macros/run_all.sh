#!/bin/bash
set -euo pipefail

CSV_PATH="/mnt/data1/test_cases.csv"
SIM_TEMPLATE="CatawbaCFD_SS_20_20_Base_Test.sim"
JAVA_MACRO="RunTestCaseMacroV6.java"
CORES_PER_JOB=8
MAX_JOBS=3
LOGFILE="/mnt/data1/launch.log"

# count cases (subtract 1 for header)
NUM_CASES=$(( "$(wc -l < "$CSV_PATH")" - 1 ))

echo "Launching $NUM_CASES simulations with a max of $MAX_JOBS at a time..."

timestamp() { date +'%Y-%m-%dT%H:%M:%S'; }

run_case() {
  local case_num="$1"
  local case_dir="/mnt/data1/Test_Case_${case_num}"
  local sim_name="TestCase_${case_num}_CatawbaCFD.sim"
  local sim_path="${case_dir}/${sim_name}"

  mkdir -p "$case_dir"
  cp "$SIM_TEMPLATE" "$sim_path"

  local start_ts end_ts duration
  start_ts=$(date +%s)
  echo "$(timestamp) START Case_${case_num}" >> "$LOGFILE"

  # Run inside the case directory so relative paths in macros (if any) behave
  (
    cd "$case_dir"
    starccm+ -np "$CORES_PER_JOB" -batch "$JAVA_MACRO" "$sim_name" \
      > "${case_dir}/run.log" 2>&1
  )

  end_ts=$(date +%s)
  duration=$(( end_ts - start_ts ))
  echo "$(timestamp) END   Case_${case_num} duration=${duration}s" >> "$LOGFILE"
}

# Launch cases with simple in-script concurrency control
for i in $(seq 1 "$NUM_CASES"); do
  # throttle concurrent jobs
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
    sleep 5
  done
  echo "Starting Test Case $i…"
  run_case "$i" &
done

wait
echo "$(timestamp) ALL_FINISHED" >> "$LOGFILE"