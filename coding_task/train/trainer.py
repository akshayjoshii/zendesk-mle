import os
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    PrinterCallback,
    ProgressCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase
)
import numpy as np
from datasets import Dataset
from typing import Dict, Optional, List, Callable, Any, Tuple

from coding_task.logging_utils import get_logger
from coding_task.constants import LOG_DIR

from coding_task.train.config import TrainingConfig

logger = get_logger(
    logger_name="TrainerSetup",
    log_file_path=os.path.join(LOG_DIR, "trainer_setup.log"),
    stream=True
)

def create_trainer(
    model: PreTrainedModel, # could be q PeftModel or a standard model
    tokenizer: PreTrainedTokenizerBase,
    train_config: TrainingConfig,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    compute_metrics: Optional[Callable[[Tuple[np.ndarray, np.ndarray]], Dict[str, float]]],
    custom_callbacks: Optional[List[Any]] = None
) -> Trainer:
    """
    Initializes and returns a HF Trainer instance configured for sequence classification with PEFT.

    Args:
        model: The model to train (can be a PEFT model).
        tokenizer: The tokenizer used for preprocessing.
        train_config: Configuration object for training parameters.
        train_dataset: The processed training dataset.
        eval_dataset: The processed evaluation dataset (optional).
        compute_metrics: The function to compute metrics during evaluation.
        custom_callbacks: Optional list of additional callbacks for the Trainer.

    Returns:
        A configured Trainer instance.
    """
    logger.info("Configuring HuggingFace Trainer...")

    # Data collator handles dynamic padding within each batch
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Pass on Training Arguments from the config object
    os.makedirs(train_config.output_dir, exist_ok=True)
    if train_config.logging_dir:
        os.makedirs(train_config.logging_dir, exist_ok=True)

    # check if evaluation and load_best_model should be enabled
    do_eval = eval_dataset is not None and train_config.evaluation_strategy != "no"
    load_best = do_eval and train_config.load_best_model_at_end

    training_args = TrainingArguments(
        output_dir=train_config.output_dir,
        num_train_epochs=train_config.num_train_epochs,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        per_device_eval_batch_size=train_config.per_device_eval_batch_size,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        lr_scheduler_type=train_config.lr_scheduler_type,
        warmup_ratio=train_config.warmup_ratio,
        logging_dir=train_config.logging_dir,
        logging_steps=train_config.logging_steps,
        evaluation_strategy=train_config.evaluation_strategy if do_eval else "no",
        eval_steps=train_config.eval_steps if train_config.evaluation_strategy == "steps" and do_eval else None,
        save_strategy=train_config.save_strategy,
        save_steps=train_config.save_steps if train_config.save_strategy == "steps" else None,
        save_total_limit=train_config.save_total_limit,
        load_best_model_at_end=load_best,
        metric_for_best_model=train_config.metric_for_best_model if load_best else None,
        greater_is_better=train_config.greater_is_better if load_best else None,
        fp16=train_config.fp16,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        gradient_checkpointing=train_config.gradient_checkpointing,
        seed=train_config.seed,
        report_to=train_config.report_to,
        remove_unused_columns=True, # useful with PEFT/custom columns
        label_names=["labels"],
        # TODO: Accelerator/Distributed like DeepSpeed or FSDP
        # WARNING: For now to reduce code complexity, we are not passing these
        # deepspeed=train_config.deepspeed, # Pass config path if provided
        # fsdp=train_config.fsdp, # Pass FSDP strategy if provided
        # fsdp_transformer_layer_cls_to_wrap=train_config.fsdp_transformer_layer_cls_to_wrap,
    )

    callbacks_to_use = [
        PrinterCallback(), # console logging
        ProgressCallback(), # TQDM progress bar
    ]

    # ass early stopping only if loading the best model based on evaluation
    if load_best:
        early_stopping_patience = 3
        logger.info(f"Adding EarlyStoppingCallback with patience={early_stopping_patience} based on '{training_args.metric_for_best_model}'.")
        callbacks_to_use.append(
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
        )
    if custom_callbacks:
        callbacks_to_use.extend(custom_callbacks)

    # init the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if do_eval else None, # only pass if evaluating
        callbacks=callbacks_to_use,
    )
    logger.info("Trainer configured.")
    return trainer