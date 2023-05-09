#!/bin/bash

configs=("configs/config_exp1.yaml" "configs/config_exp2.yaml")
index=2

for config in "${configs[@]}"; do
  let index++
  log_filename="output_exp_${index}.log"

  echo "Starting training with ${config}..."

  cp ${config} config_exp.yaml
  nohup python3 train.py > "${log_filename}" 2>&1 &
  wait $!

  echo "Training with ${config} has completed."

  nohup python3 inference.py > "${log_filename}" 2>&1 &
  wait $!
  echo "Inferencing with ${config} has completed."
done

echo "All experiments have been completed."
