#!/bin/bash

configs=("configs/config_1.yaml" "configs/config_2.yaml")
index=0

for config in "${configs[@]}"; do
  let index++
  log_filename_train="output_logs/output_train_${index}.log"
  log_filename_infer="output_logs/output_infer_${index}.log"

  echo "Starting training with ${config}..."

  nohup python3 train.py ${config} > "${log_filename_train}" 2>&1 &
  wait $!

  echo "Training with ${config} has completed."

  nohup python3 inference.py ${config} > "${log_filename_infer}" 2>&1 &
  wait $!

  echo "Inferencing with ${config} has completed."
done

echo "All experiments have been completed."
