#!/bin/bash

log_filename_train="output_logs/TAPT_First.log"

echo "Starting pre-training TAPT"

nohup python3 TAPT.py > "${log_filename_train}" 2>&1 &
wait $!

echo "Training with has completed."

done

echo "All experiments have been completed."