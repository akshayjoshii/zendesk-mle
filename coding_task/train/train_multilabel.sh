#!/bin/bash
# Run this file fro project rootm: /zendesk-mle

python -m coding_task.train.main \
    --dataset_path ./coding_task/data/atis/train.tsv \
    --output_dir ./results/atis_multilabel_xlmr_lora \
    --model_name_or_path xlm-roberta-base \
    --task_type multilabel \
    --unpack_multi_labels True \
    --label_delimiter + \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-5 \
    --freeze_base_model True \
    --peft_config.method lora \
    --peft_config.lora_r 16 \
    --peft_config.lora_alpha 32 \
    --report_to tensorboard \
    --logging_steps 50 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model eval_f1_micro \
    --greater_is_better True \
    --fp16 # ass if you hve V100 or better GPU
