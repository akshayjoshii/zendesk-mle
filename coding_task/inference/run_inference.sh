#!/bin/bash
# Run this file from project rootm: /zendesk-mle

# Single Text Input (prints to console):
python -m coding_task.inference.main \
    --model_path ./results/BEST_atis_multilabel_xlmr_lora \
    --input_text "which flights are leaving from kansas city to atlanta early monday morning" \
    --device cpu # or cuda
    # --include_probabilities True # Optional

#################################################################

# Single Text Input (saves to CSV):
python -m coding_task.inference.main \
    --model_path ./results/BEST_atis_multilabel_xlmr_lora \
    --input_text "find flights and fares from denver to pittsburgh" \
    --output_file ./inference_outputs/single_pred.csv \
    --device cuda \
    --include_probabilities True

#################################################################

# File Input (CSV/TSV, print to console): works only if input_data.csv has a column named 'query'
python -m coding_task.inference.main \
    --model_path ./results/BEST_atis_multilabel_xlmr_lora \
    --input_file ./data/new_queries.csv \
    --text_column query \
    --batch_size 16 \
    --device cuda

#################################################################

# File Input (saves results to CSV): works only if input_data.csv has a column named 'text'
python -m coding_task.inference.main \
    --model_path ./results/BEST_atis_multilabel_xlmr_lora \
    --input_file ./data/unseen_atis.tsv \
    --text_column text \
    --output_file ./inference_outputs/predictions.csv \
    --batch_size 32 \
    --device cuda \
    --include_probabilities True