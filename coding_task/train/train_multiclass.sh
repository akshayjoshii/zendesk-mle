#!/bin/bash
# Run this file fro project rootm: /zendesk-mle

python -m coding_task.train.main \
    --dataset_path ./coding_task/data/atis/train.tsv \
    --test_dataset_path ./coding_task/data/atis/test.tsv \
    --output_dir ./results/atis_multiclass_xlmr_lora \
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
    --logging_steps 20 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model eval_f1_weighted \
    --fp16 # add if you have atleast a V100 GPU
    # --use_dask True # if dataset is very large
