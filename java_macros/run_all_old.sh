#!/bin/bash

CSV_PATH="/mnt/data1/test_cases.csv"
SIM_TEMPLATE="CatawbaCFD_SS_20_20_Base_Test.sim"
JAVA_MACRO="RunTestCaseMacroV5.java"
CORES_PER_JOB=8
MAX_JOBS=3

NUM_CASES=$(($(wc -l < "$CSV_PATH") - 1))

echo "Launching $NUM_CASES simulations with a max of $MAX_JOBS at a time..."

i=1
while [ $i -le $NUM_CASES ]; do
    RUNNING=$(pgrep -af starccm+ | grep -c 'RunTestCaseMacroV5')
    if [ "$RUNNING" -lt "$MAX_JOBS" ]; then
        echo "Starting Test Case $i..."
        echo "$(date): Launching Test Case $i" >> /mnt/data1/launch.log
        mkdir -p "/mnt/data1/Test_Case_$i"
        cp "$SIM_TEMPLATE" "TestCase_${i}_CatawbaCFD.sim"
        starccm+ -np $CORES_PER_JOB -batch "$JAVA_MACRO" "TestCase_${i}_CatawbaCFD.sim" > "/mnt/data1/Test_Case_$i/run.log" 2>&1 &
        ((i++))
    else
        sleep 5
    fi
done

wait
echo "All simulations finished." >> /mnt/data1/launch.log
