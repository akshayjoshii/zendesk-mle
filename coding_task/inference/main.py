import os
import sys
import pandas as pd
from transformers import HfArgumentParser

from coding_task.inference.config import InferenceConfig
from coding_task.inference.predictor import TextClassifierPredictor
from logging_utils import get_logger


def main():
    # Parse Arguments
    parser = HfArgumentParser((InferenceConfig,))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        inference_args, = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # from CMD line
        inference_args, = parser.parse_args_into_dataclasses()

    logger = get_logger(
        logger_name="InferenceMain",
        log_file_path=inference_args.log_file,
        stream=True
    )

    logger.info("--- Inference Configuration ---")
    logger.info(inference_args)

    # validate inputs: must provide either text or file, but not both
    if not inference_args.input_text and not inference_args.input_file:
        logger.error("No input provided. Please specify either --input_text or --input_file.")
        sys.exit(1)
    if inference_args.input_text and inference_args.input_file:
        logger.error("Both --input_text and --input_file provided. Please specify only one.")
        sys.exit(1)

    # Initialize model 
    try:
        predictor = TextClassifierPredictor(inference_args)
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}", exc_info=True)
        sys.exit(1)

    # Perform Prediction
    results_df = None
    results_single = None

    try:
        if inference_args.input_text:
            logger.info(f"Predicting for single input text...")
            results_single = predictor.predict(inference_args.input_text)
            logger.info(f"Input Text: '{inference_args.input_text}'")
            logger.info(f"Prediction: {results_single}")

        elif inference_args.input_file:
            logger.info(f"Predicting from input file: {inference_args.input_file}")
            results_df = predictor.predict_from_file(
                input_file=inference_args.input_file,
                text_column=inference_args.text_column
            )
            logger.info(f"Predictions generated for {len(results_df)} rows.")
            logger.info("Sample predictions (first 5 rows):")
            logger.info(f"\n{results_df.head().to_string()}")

    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}", exc_info=True)
        sys.exit(1)

    # Handle Output
    if inference_args.output_file:
        output_path = inference_args.output_file
        logger.info(f"Saving results to: {output_path}")
        try:
            if results_df is not None:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                results_df.to_csv(output_path, index=False)
            elif results_single is not None:
                 single_df = pd.DataFrame([{
                     "input_text": inference_args.input_text,
                     **results_single # unpack the prediction dict
                 }])
                 os.makedirs(os.path.dirname(output_path), exist_ok=True)
                 single_df.to_csv(output_path, index=False)
            else:
                 logger.warning("No results generated to save.")

            logger.info(f"Results saved successfully.")

        except Exception as e:
            logger.error(f"Failed to save results to {output_path}: {e}", exc_info=True)
    else:
        if results_df is not None:
            logger.info("Predictions (DataFrame):")
            print(results_df.to_string())

    logger.info("Inference script finished.")


if __name__ == "__main__":
    main()

