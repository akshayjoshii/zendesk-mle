#!/bin/bash

################################################################################
# WARNING: This script is intended to be run inside a Docker container. So do not
# run it directly on your host machine. It is designed to be the entrypoint for
# a Docker image that has the necessary environment and dependencies set up!!!
################################################################################

# Exit immediately if a command exits with a non-zero status.
set -e

# Prevent errors in a pipeline from being masked.
set -o pipefail

# These paths MUST match the volume mount points defined in the Dockerfile
# and used in the docker run command.
readonly DATA_DIR="/app/data"
readonly OUTPUT_DIR="/app/output"
readonly LOG_DIR="/app/logs" # Assuming logs are also mounted or handled

run_type="$1" # Expecting "multiclass" or "multilabel"

if [[ -z "$run_type" ]]; then
    echo "[ERROR] No run type specified. Please provide 'multiclass' or 'multilabel' as the first argument." >&2
    echo "Usage: docker run ... <image> [multiclass|multilabel]" >&2
    exit 1
fi

echo "[INFO] Entrypoint received run type: ${run_type}"
echo "[INFO] Using Data Directory: ${DATA_DIR}"
echo "[INFO] Using Output Directory: ${OUTPUT_DIR}"

# We will call the python module directly, but use the arguments defined
# in the corresponding shell script, replacing paths as needed.

echo "[INFO] Launching training..."

# Change to the app directory to ensure relative imports in coding_task work
cd /app

# Use exec to replace the current shell process with the python process
# as this ensures signals like SIGTERM from docker stop are passed correctly to Python
if [[ "$run_type" == "multiclass" ]]; then
    exec python -m coding_task.train.main \
        --dataset_path "${DATA_DIR}/train.tsv" \
        --test_dataset_path "${DATA_DIR}/test.tsv" \
        --output_dir "${OUTPUT_DIR}" \
        --model_name_or_path xlm-roberta-base \
        --task_type multiclass \
        --unpack_multi_labels False \
        --num_train_epochs 10 \
        --per_device_train_batch_size 16 \
        --learning_rate 1e-4 \
        --freeze_base_model True \
        --method lora \
        --lora_r 32 \
        --lora_alpha 16 \
        --report_to tensorboard \
        --logging_steps 25 \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --load_best_model_at_end True \
        --metric_for_best_model eval_f1_weighted \
        --fp16 # Keep based on GPU availability in container. Minimum V100 required
        # --use_dask True # Add if needed

elif [[ "$run_type" == "multilabel" ]]; then
    exec python -m coding_task.train.main \
        --dataset_path "${DATA_DIR}/train.tsv" \
        --test_dataset_path "${DATA_DIR}/test.tsv" \
        --output_dir "${OUTPUT_DIR}" \
        --model_name_or_path xlm-roberta-base \
        --task_type multilabel \
        --unpack_multi_labels True \
        --label_delimiter + \
        --num_train_epochs 10 \
        --per_device_train_batch_size 16 \
        --learning_rate 1e-4 \
        --freeze_base_model True \
        --method lora \
        --lora_r 64 \
        --lora_alpha 32 \
        --report_to tensorboard \
        --logging_steps 25 \
        --evaluation_strategy epoch \
        --save_strategy epoch \
        --load_best_model_at_end True \
        --metric_for_best_model eval_f1_micro \
        --greater_is_better True \
        --fp16 # Keep based on GPU availability in container

else
    echo "[ERROR] Unknown run type: '${run_type}'. Please use 'multiclass' or 'multilabel'." >&2
    exit 1
fi

# The exec command replaces this script, so anything below here won't run unless exec fails
echo "[ERROR] Failed to execute training command." >&2
exit 1
