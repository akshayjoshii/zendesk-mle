# Training & Evaluation Summary

This document summarizes the key insights and performance metrics from the best model that we trained and evaluated on Validation & provided Test sets.

## Key Configuration Insights:

* **Model:** Multilingual `xlm-roberta-base` was used as the base model. This model knows 100+ languages and is suitable for multilingual intent classification tasks.
* **Task:** Multi-label classification (`multilabel`).
* **Data:** Training data from `atis/train.tsv`, test data from `atis/test.tsv`. Labels are delimited by `+`.
* **Training Method:** LoRA (method `lora`) was used for parameter-efficient fine-tuning with rank `r=256` and `alpha=32`. The base model weights were frozen (`freeze_base_model True`).
* **Hyperparameters:**
    * Epochs requested: 50 (but logs show it finished at epoch 22 - potentially due to `early_stopping` mechanism cutting the training short to prevent the model from overfitting on the training set).
    * Batch Size: 32
    * Learning Rate: 1e-4
* **Evaluation:** Performed every epoch (`evaluation_strategy epoch`). The best model was selected based on `eval_f1_micro`, saving occurred every epoch, and the best checkpoint was loaded at the end.
* **Efficiency:** Mixed-precision training (`fp16`) was enabled.

## I. Training Summary

The model training completed after 22 epochs.

| Metric                     | Value         |
| :------------------------- | :------------ |
| **Epochs Completed** | 22.0          |
| **Final Training Loss** | ~0.0559       |
| **Total Training Runtime** | 0:05:18.91    |
| **Training Samples/sec** | ~581.19       |
| **Training Steps/sec** | ~18.19        |

* **Observation:** Training progressed through 22 epochs with a final loss of approximately 0.0559.

## II. Validation Set Performance (Final @ Epoch 22)

The final evaluation on the validation set was performed after training completion.

| Metric                | Micro Avg. | Macro Avg. | Weighted Avg. | Samples Avg. | Other                 |
| :-------------------- | :--------- | :--------- | :------------ | :----------- | :-------------------- |
| **Precision** | ~0.9892    | ~0.8087    | ~0.9882       | ~0.9854      |                       |
| **Recall** | ~0.9818    | ~0.7086    | ~0.9818       | ~0.9842      |                       |
| **F1-Score** | **~0.9855**| ~0.7409    | ~0.9838       | ~0.9844      |                       |
| **Jaccard Score** | ~0.9714    | ~0.6961    | ~0.9717       | ~0.9836      |                       |
| **Avg. Precision** | ~0.9939    | `nan`      | ~0.9921       | -            |                       |
| **ROC AUC** | `nan`      | `nan`      | `nan`         | -            |                       |
| **Subset Accuracy** | -          | -          | -             | -            | **~0.9817** |
| **Hamming Loss** | -          | -          | -             | -            | **~0.0017** |
| **Loss** | -          | -          | -             | -            | **~0.0093** |
| **Runtime** | -          | -          | -             | -            | ~0.84 sec             |
| **Samples/sec** | -          | -          | -             | -            | ~1108                 |

* **Key Insight:** The model achieves very high micro-averaged F1 (~98.55%) and subset accuracy (~98.17%) on the validation set, indicating strong overall performance.
* **Observation:** Macro-averaged scores (F1: ~0.74, Jaccard: ~0.69) are significantly lower than micro-averaged ones. This suggests potential performance variation across different labels, with potentially lower performance on less frequent labels.
* **Observation:** ROC AUC metrics could not be computed (`nan`). Average Precision was computable for micro and weighted averages but not macro.

## III. Test Set Performance (Final @ Epoch 22)

The final evaluation was also performed on the test set.

| Metric                | Micro Avg. | Macro Avg. | Weighted Avg. | Samples Avg. | Other                 |
| :-------------------- | :--------- | :--------- | :------------ | :----------- | :-------------------- |
| **Precision** | ~0.9801    | ~0.8159    | ~0.9859       | ~0.9788      |                       |
| **Recall** | ~0.9722    | ~0.8075    | ~0.9722       | ~0.9759      |                       |
| **F1-Score** | **~0.9761**| ~0.7955    | ~0.9772       | ~0.9757      |                       |
| **Jaccard Score** | ~0.9534    | ~0.7417    | ~0.9590       | ~0.9724      |                       |
| **Avg. Precision** | ~0.9921    | `nan`      | ~0.9929       | -            |                       |
| **ROC AUC** | `nan`      | `nan`      | `nan`         | -            |                       |
| **Subset Accuracy** | -          | -          | -             | -            | **~0.9624** |
| **Hamming Loss** | -          | -          | -             | -            | **~0.0028** |
| **Loss** | -          | -          | -             | -            | **~0.0120** |
| **Runtime** | -          | -          | -             | -            | ~0.65 sec             |
| **Samples/sec** | -          | -          | -             | -            | ~1314                 |

* **Key Insight:** Performance on the test set is strong and generally consistent with the validation set, though slightly lower (e.g., Micro F1: ~97.61% vs ~98.55%, Subset Accuracy: ~96.24% vs ~98.17%).
* **Observation:** Macro F1 (~0.7955) is closer to Micro F1 on the test set compared to the validation set, but still indicates some variance across labels.
* **Observation:** ROC AUC metrics remain `nan`, and Macro Average Precision is also `nan`.
* **Observation:** Test set evaluation was slightly faster (higher samples/sec) than validation.

## IV. Comparison: Validation vs. Test

| Metric             | Validation (@ Ep 22) | Test (@ Ep 22) | Delta       | Observation                      |
| :----------------- | :------------------- | :------------- | :---------- | :------------------------------- |
| **Micro F1** | ~0.9855              | ~0.9761        | -0.0094     | Slight drop on test set        |
| **Macro F1** | ~0.7409              | ~0.7955        | +0.0546     | *Improved* macro F1 on test    |
| **Subset Accuracy**| ~0.9817              | ~0.9624        | -0.0193     | Noticeable drop on test set    |
| **Hamming Loss** | ~0.0017              | ~0.0028        | +0.0011     | Higher (worse) loss on test    |
| **Eval Loss** | ~0.0093              | ~0.0120        | +0.0027     | Higher (worse) loss on test    |

* **Crucial Insight:** While most metrics show a slight decrease in performance from validation to test (as expected), the **Macro F1 score surprisingly *improves***. This warrants further investigation into the label distribution differences between validation and test sets or the specific errors made. The drop in subset accuracy is also notable.

## V. Important Warnings & Logs

* **`UndefinedMetricWarning: Jaccard is ill-defined...`:** This warning occurred during both val and test evaluations. It indicates that for some labels, there were no true *or* predicted samples, leading to a Jaccard score of 0.0 for those specific labels (controlled by `zero_division` parameter in sklearn). This often happens with rare labels.
* **`RuntimeWarning: invalid value encountered in divide`:** This occurred in `sklearn/metrics/_ranking.py` during recall calculation - likely related to AUC/Avg. Precision calculations. It's mostly division by zero, possibly cuz there were no positive samples for some classes/queries when calculating these ranking metrics.
* **`UserWarning: unknown class(es) ['day_name'] will be ignored`:** This occurred during the processing of the *test* dataset (`./coding_task/data/atis/test.tsv`). It means the label 'day_name' was present in the test data but was not seen during training. This is a **critical issue**, as the model cannot predict labels it wasn't trained on or aware of.