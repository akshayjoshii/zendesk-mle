import numpy as np
import evaluate
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import hamming_loss, jaccard_score, matthews_corrcoef, balanced_accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score, cohen_kappa_score
import torch # sigmoid activation in multilabel type
from typing import Dict, Tuple, Callable

try:
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
except Exception as e:
    raise RuntimeError(f"Failed to load evaluation metrics: {e}")


def compute_metrics_fn(task_type: str, 
    id2label: Dict[int, str], 
    threshold: float = 0.5
) -> Callable[[Tuple[np.ndarray, np.ndarray]], Dict[str, float]]:
    """
    Factory function to create the compute_metrics function for the Trainer,
    handling both multiclass and multilabel scenarios with specified metrics.

    Args:
        task_type (str): 'multiclass' or 'multilabel'.
        id2label (Dict[int, str]): Mapping from label ID to label name.
        threshold (float): Threshold for converting probabilities to binary predictions
                           in the multilabel case.

    Returns:
        Callable: A function that takes EvalPrediction (logits, labels) and
                  returns a dictionary of computed metrics.

    Raises:
        ValueError: If task_type is not 'multiclass' or 'multilabel'.
        RuntimeError: If evaluate metrics failed to load.
    """
    # if not all([accuracy_metric, precision_metric, recall_metric, f1_metric]):
    #     raise RuntimeError("Essential metrics from 'evaluate' library could not be loaded.")

    num_labels = len(id2label)

    def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """
        Computes evaluation metrics based on task type.

        Args:
            eval_pred (Tuple[np.ndarray, np.ndarray]): A tuple containing logits and labels.
                                                For multilabel, labels are expected
                                                to be multi-hot encoded floats/ints.

        Returns:
            Dict[str, float]: Dictionary of computed metrics.
        """
        logits, labels = eval_pred
        results = {}

        if task_type == "multiclass":
            predictions = np.argmax(logits, axis=-1)
            # Ensure labels are integers for metric calculation
            labels = labels.astype(int)

            results["accuracy"] = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]

            # Calculate micro/macro/weighted F1, precision, recall
            for average in ["micro", "macro", "weighted"]:
                results[f"precision_{average}"] = precision_metric.compute(predictions=predictions, references=labels, average=average)["precision"]
                results[f"recall_{average}"] = recall_metric.compute(predictions=predictions, references=labels, average=average)["recall"]
                results[f"f1_{average}"] = f1_metric.compute(predictions=predictions, references=labels, average=average)["f1"]

            # Set the primary metric for comparison (e.g., weighted F1 or macro f1)
            results["f1"] = results["f1_weighted"]
            
            # Balanced accuracy - accuracy that accounts for class imbalance
            results["balanced_accuracy"] = balanced_accuracy_score(labels, predictions)
            
            # Cohen's Kappa - agreement between predicted and actual, accounting for chance
            results["cohen_kappa"] = cohen_kappa_score(labels, predictions)
            
            # Matthews Correlation Coefficient - balanced measure for multiclass
            results["matthews_corrcoef"] = matthews_corrcoef(labels, predictions)
            
            # try to compute ROC AUC for multiclass (one-vs-rest)
            try:
                # Convert logits to probabilities using softmax
                probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
                results["roc_auc_ovr"] = roc_auc_score(
                    y_true=np.eye(num_labels)[labels], 
                    y_score=probs, 
                    average="weighted", 
                    multi_class="ovr"
                )
            except Exception as e:
                # Skip this metric if it fails e.g., if only one class is present in batch
                results["roc_auc_ovr"] = float('nan')

        elif task_type == "multilabel":
            # apply sigmoid to logits to get probabilities
            probs = torch.sigmoid(torch.Tensor(logits)).numpy()
            # Apply threshold to get binary predictions
            predictions = (probs > threshold).astype(int)
            # mke sure labels are also binary
            labels = labels.astype(int)

            # 'micro': Calculates metrics globally by counting total TP, FN, FP. Good overall measure.
            # 'macro': Calculates metrics for each label, then finds unweighted mean. Treats all labels equally.
            # 'weighted': Calculates metrics for each label, weighted by support. Accounts for imbalance.
            # 'samples': Calculates metrics for each instance, then finds average. Good for instance-level performance.
            for average in ["micro", "macro", "weighted", "samples"]:
                # Use zero_division=0 or 1 to avoid warnings/errors if a class has no predictions/labels
                results[f"precision_{average}"] = precision_score(y_true=labels, y_pred=predictions, average=average, zero_division=0)
                results[f"recall_{average}"] = recall_score(y_true=labels, y_pred=predictions, average=average, zero_division=0)
                results[f"f1_{average}"] = f1_score(y_true=labels, y_pred=predictions, average=average, zero_division=0)

            # Set the primary metric for comparison (e.g., micro F1 or weighted F1)
            results["f1"] = results["f1_micro"]

            # Subset Accuracy (exact match ratio)
            # Computed manually: fraction of samples where prediction exactly matches label vector
            subset_accuracy = np.mean([np.array_equal(p, l) for p, l in zip(predictions, labels)])
            results["subset_accuracy"] = subset_accuracy

            # Hamming Loss: fraction of labels that are incorrectly predicted (lower is better)
            results["hamming_loss"] = hamming_loss(labels, predictions)
            
            # Jaccard similarity (IoU) - measures overlap between label sets
            for average in ["micro", "macro", "weighted", "samples"]:
                try:
                    results[f"jaccard_{average}"] = jaccard_score(
                        y_true=labels, 
                        y_pred=predictions, 
                        average=average
                    )
                except Exception:
                    results[f"jaccard_{average}"] = float('nan')
            
            # ROC AUC - area under ROC curve for each label then averaged
            try:
                results["roc_auc_micro"] = roc_auc_score(labels, probs, average="micro")
                results["roc_auc_macro"] = roc_auc_score(labels, probs, average="macro")
                results["roc_auc_weighted"] = roc_auc_score(labels, probs, average="weighted")
                
                # set primary ROC AUC metric
                results["roc_auc"] = results["roc_auc_micro"]
            except Exception:
                # Skip if calculation fails - e.g., if a class has no positive samples
                results["roc_auc_micro"] = results["roc_auc_macro"] = results["roc_auc_weighted"] = float('nan')
                results["roc_auc"] = float('nan')
                
            # Average Precision - average precision for each label
            try:
                results["avg_precision_micro"] = average_precision_score(labels, probs, average="micro")
                results["avg_precision_macro"] = average_precision_score(labels, probs, average="macro")
                results["avg_precision_weighted"] = average_precision_score(labels, probs, average="weighted")
                
                # st primary average precision metric
                results["avg_precision"] = results["avg_precision_micro"]
            except Exception:
                results["avg_precision_micro"] = results["avg_precision_macro"] = results["avg_precision_weighted"] = float('nan')
                results["avg_precision"] = float('nan')

        else:
            raise ValueError(f"Invalid task_type '{task_type}' encountered in compute_metrics.")

        return results

    # Validate task_type before returning the function
    if task_type not in ["multiclass", "multilabel"]:
        raise ValueError(f"Unsupported task_type for metrics: {task_type}. Choose 'multiclass' or 'multilabel'.")

    return compute_metrics