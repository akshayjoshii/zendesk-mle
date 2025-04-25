import os
from transformers import AutoModelForSequenceClassification, AutoConfig
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
    IA3Config,
    PeftConfig as PeftLibConfig
    # TODO: Import other PEFT configs like IA3Config, AdapterFusion, Compactor, PrefixTuning etc
    # See more here: https://docs.adapterhub.ml/methods.html or https://huggingface.co/docs/peft/index
)
from typing import Dict, Optional
from coding_task.logging_utils import get_logger
from coding_task.constants import LOG_DIR

from coding_task.train.config import ModelConfig, PeftConfig, DataConfig

logger = get_logger(
    logger_name="ModelLoader",
    log_file_path=os.path.join(LOG_DIR, "model_loader.log"),
    stream=True
)

def load_model_for_training(
    model_config: ModelConfig,
    peft_config: PeftConfig,
    data_config: DataConfig,
    num_labels: int,
    id2label: Dict[int, str],
    label2id: Dict[str, int]
) -> PeftModel:
    """
    Loads the base model, applies the specified PEFT adapter, and freezes
    base layers if configured.

    Args:
        model_config: Configuration for the base model.
        peft_config: Configuration for the PEFT method.
        data_config: Configuration for data (needed for task type).
        num_labels: Number of unique labels.
        id2label: Mapping from label ID to label name.
        label2id: Mapping from label name to label ID.

    Returns:
        The PEFT-enhanced model ready for training.

    Raises:
        ValueError: If an unsupported PEFT method or task type is specified.
    """
    logger.info(f"Loading base model: {model_config.model_name_or_path}")

    # Determine problem type for HF model head and loss function selection
    if data_config.task_type == "multiclass":
        problem_type = "single_label_classification"
        hf_task_type = TaskType.SEQ_CLS # PEFT task type
    elif data_config.task_type == "multilabel":
        problem_type = "multi_label_classification"
        hf_task_type = TaskType.SEQ_CLS # same PEFT task type
    else:
        raise ValueError(f"Unsupported task_type for model loading: {data_config.task_type}")

    logger.info(f"Configuring model for '{problem_type}' (num_labels={num_labels}).")

    # Load model config, setting label mappings and problem type
    config = AutoConfig.from_pretrained(
        model_config.model_name_or_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type=problem_type,
        cache_dir=model_config.cache_dir,
        # FOR EXTENSIBILITY: can be useful for some models/pipelines
        # finetuning_task=data_config.task_type
    )

    # Load the base model with the configured head
    # ignore_mismatched_sizes=True can be useful if loading a checkpoint saved
    # without a classification head or with a head of a different size.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path,
        config=config,
        cache_dir=model_config.cache_dir,
        ignore_mismatched_sizes=True
    )

    # Freeze base model parameters (if you'd like to train only the adapter)
    if model_config.freeze_base_model:
        logger.info("Freezing base model parameters.")
        for param in model.base_model.parameters():
            param.requires_grad = False

        # Verify classifier head is trainable
        for name, param in model.named_parameters():
            if 'classifier' in name or 'classification_head' in name:
                logger.debug(f"Classifier param '{name}' requires_grad: {param.requires_grad}")

    # Apply config and PEFT adapter
    logger.info(f"Applying PEFT method: {peft_config.method}")
    peft_lib_config: Optional[PeftLibConfig] = None

    if peft_config.method.lower() == "lora":
        # Target modules can often be auto-inferred for common archs like BERT, RoBERTa
        # Specifying them explicitly gives us more control over
        # Common targets for BERT/RoBERTa are 'query', 'key', 'value', 'dense' in attention layers
        # If None, peft library will try to find suitable ones
        target_modules = peft_config.lora_target_modules
        if target_modules is None:
            logger.info("lora_target_modules not specified, attempting auto-detection by PEFT library.")
            # This is a common practice, but we can specify them explicitly
            # TODO: I'm skiping this for now

        peft_lib_config = LoraConfig(
            task_type=hf_task_type,
            r=peft_config.lora_r,
            lora_alpha=peft_config.lora_alpha,
            lora_dropout=peft_config.lora_dropout,
            bias="none", # can be "all" or "lora_only"
            target_modules=target_modules
        )
        logger.info(f"LoRA Config: r={peft_config.lora_r}, alpha={peft_config.lora_alpha}, dropout={peft_config.lora_dropout}, target_modules={target_modules or 'auto'}")

    # example for IA3 Adapter
    elif peft_config.method.lower() == "ia3":
        # in the original implementations IA3 targets include feed-forward layers too
        target_modules = peft_config.ia3_target_modules or ["key", "value", "output.dense"]
        peft_lib_config = IA3Config(
            task_type=hf_task_type,
            target_modules=target_modules,
            feedforward_modules=["output.dense"], # adjust based on model
            inference_mode=False
        )
        logger.info(f"IA3 Config: target_modules={target_modules}, feedforward_modules={peft_lib_config.feedforward_modules}")

    # exmple for Adapters - using the 'adapters' library instead of 'peft'
    # TODO: Enable this later, for now I'm keeping it commented out to prevent further
    # headaches within these 3 days of the challenge
    elif peft_config.method.lower() == "adapter":
        # # This would require using the 'adapters' library instead of 'peft'
        # from adapters import BnConfig # Or other adapter configs
        # logger.info("Using 'adapters' library for Bottleneck Adapters.")
        # adapter_config = BnConfig(reduction_factor=peft_config.reduction_factor)
        # model.add_adapter("my_adapter", config=adapter_config)
        # model.train_adapter("my_adapter")
        # # Note: Freezing needs to be handled differently with 'adapters' library
        # model.freeze_model(freeze=model_config.freeze_base_model)
        # logger.info(f"Adapter Config: reduction_factor={peft_config.reduction_factor}")
        # # Return the model directly, not a PeftModel
        # return model # Type hint would need adjustment for this case
        raise NotImplementedError("PEFT method 'adapter' requires using the 'adapters' library, which is not integrated here.")

    else:
        raise ValueError(f"Unsupported PEFT method: {peft_config.method}. Choose 'lora' or add implementation for others.")

    # Apply the PEFT config to the base model
    peft_model = get_peft_model(model, peft_lib_config)

    logger.info("PEFT model created successfully.")
    peft_model.print_trainable_parameters()

    return peft_model

