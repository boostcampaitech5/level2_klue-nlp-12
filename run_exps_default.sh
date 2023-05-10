#!/bin/bash

configs=("configs/config_exp.yaml")
index=0

for config in "${configs[@]}"; do
  let index++
  log_filename="output_logs/output_exp_${index}.log"

  echo "Starting training with ${config}..."

  nohup python3 train.py ${config} > "${log_filename}" 2>&1 &
  wait $!

  echo "Training with ${config} has completed."

  nohup python3 inference.py ${config} > "${log_filename}" 2>&1 &
  wait $!

  echo "Inferencing with ${config} has completed."
done

echo "All experiments have been completed."
