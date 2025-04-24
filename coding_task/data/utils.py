"""
This script contains a class for loading and preprocessing ML datasets.
It supports both pandas and Dask for handling large files.
It also provides a func for basic text cleanup.
"""

import os
import pandas as pd
import dask.dataframe as dd
from datasets import load_dataset, Dataset
from typing import Callable, Optional, List, Dict, Any, Union

# import constants from root directory
from coding_task.constants import LOG_DIR
from coding_task.logging_utils import get_logger

logger = get_logger(
    logger_name="Data Utils",
    log_file_path=os.path.join(LOG_DIR, "data_utils.log"),
    stream=True
)


class CustomTextDataset:
    def __init__(self, 
        file_path: str, 
        file_type: Optional[str] = None, 
        column_names: Optional[List[str]] = None,
        split: Optional[str] = 'train', 
        use_dask: bool = False,
        unpack_multi_labels: bool = False, # needed if unpacking
        label_column_name: Optional[str] = None, # needed if unpacking
        label_delimiter: str = '+' # delimiter for unpacking
        ):
        """
        CustomTextDataset is a class for loading and preprocessing text datasets.
        It supports both pandas and Dask for handling large files.
        Args:
            file_path (str): Path to the dataset file.
            file_type (str, optional): Type of the file ('csv', 'tsv', 'json'). 
                            If None, it will be inferred from the file extension.
            column_names (List[str], optional): List of column names for the dataset. 
                            If None, it will be inferred from the file.
            split (str, optional): The split of the dataset to load (e.g., 'train', 'test').
            use_dask (bool): Whether to use Dask for loading large files. 
                            Default is False.
            unpack_multi_labels (bool): If True, rows with delimiter-separated labels
                            in the label_column_name will be duplicated, one row per label.
                            Default is False.
            label_column_name (str, optional): Name of the column containing labels 
                            to unpack.
                            Required if unpack_multi_labels is True.
            label_delimiter (str): The delimiter used to separate combined labels
                            (e.g., '+'). Default is '+'.
                            Required if unpack_multi_labels is True.
        """
        self.file_path = file_path
        self.file_type = file_type or self._infer_file_type()
        self.column_names = column_names
        self.use_dask = use_dask
        self.split = split
        self.unpack_multi_labels = unpack_multi_labels
        self.label_column_name = label_column_name
        self.label_delimiter = label_delimiter

        if self.unpack_multi_labels and not (self.label_column_name and self.label_delimiter):
            raise ValueError("label_column_name & label_delimiter must be provided if unpack_multi_labels is True")
        
        if self.unpack_multi_labels and self.column_names and \
            self.label_column_name not in self.column_names:
             raise ValueError(f"label_column_name '{self.label_column_name}' not found in provided column_names: {self.column_names}")
        
        # log all the args
        self._log_init()

    def _log_init(self):
        logger.info(f"CustomTextDataset initialized:")
        logger.info(f"  file_path: {self.file_path}")
        logger.info(f"  file_type: {self.file_type}")
        logger.info(f"  column_names: {self.column_names}")
        logger.info(f"  use_dask: {self.use_dask}")
        logger.info(f"  unpack_multi_labels: {self.unpack_multi_labels}")
        if self.unpack_multi_labels:
            logger.info(f"  label_column_name: {self.label_column_name}")
            logger.info(f"  label_delimiter: {self.label_delimiter}")

    def _infer_file_type(self) -> str:
        """ Infer the file type based on the file extension."""
        ext = os.path.splitext(self.file_path)[-1].lower()
        logger.info(f"File extension: {ext}")
        if ext in ['.tsv']:
            return 'tsv'
        elif ext in ['.csv']:
            return 'csv'
        elif ext in ['.json', '.jsonl']:
            return 'json'
        else:
            raise ValueError(f"unsupported file extension: {ext}")
        
    def _unpack_dataframe(self, 
        df: Union[pd.DataFrame, dd.DataFrame]
    ) -> Union[pd.DataFrame, dd.DataFrame]:
        """
        Helper function to unpack multi-labels in the specified label column by 
        using the efficient 'explode' method.

        Args:
            df (Union[pd.DataFrame, dd.DataFrame]): The df to unpack
        Returns:
            Union[pd.DataFrame, dd.DataFrame]: The unpacked df
        """
        if not self.label_column_name: # sanity check before actually unpacking
             logger.error("Cannot unpack labels without label_column_name.")
             return df

        logger.info(f"Unpacking multi-labels in column '{self.label_column_name}' using delimiter '{self.label_delimiter}'.")
        initial_rows = len(df)

        # make sure label column is treated as string before splitting
        # In dask specifying meta helps ensure type consistency
        if self.use_dask:
            label_series = df[self.label_column_name].astype(str).str.split(self.label_delimiter)
        else:
            # same thing but with pandas
            df[self.label_column_name] = df[self.label_column_name].astype(str)
            label_series = df[self.label_column_name].str.split(self.label_delimiter)

        # assign the list back (or create new df with it) before exploding
        df = df.assign(**{self.label_column_name: label_series})

        # explode the df based on the list in the label column
        df_exploded = df.explode(self.label_column_name)

        # Only reset index if it's a Pandas DataFrame
        if isinstance(df_exploded, pd.DataFrame):
            df_exploded = df_exploded.reset_index(drop=True)

        # Use isinstance check for logging sample rows too
        final_rows = len(df_exploded) if isinstance(df_exploded, pd.DataFrame) else None
        logger.info(f"Unpacking complete. Initial rows: {initial_rows}, Final rows: {final_rows or 'unknown (Dask)'}.")

        if isinstance(df_exploded, pd.DataFrame) and final_rows > initial_rows:
             try:
                  # log duplicated index before reset_index if using PD
                  # Or just log head after reset_index
                  logger.debug(f"Sample rows after unpacking (Pandas):\n{df_exploded.head()}")
             except Exception:
                  logger.debug("Could not log duplicated rows sample.")

        return df_exploded

    def load_to_dataframe(self,
        chunksize: Optional[int] = None
    ) -> Union[pd.DataFrame, dd.DataFrame]:
        """
        Load the dataset into a pandas or Dask df
        Optionally unpacks multi-label entries if configured

        Args:
            chunksize (int, optional): Number of rows to read at a time (pandas only).
                            If None, the entire file is read into memory (pandas).
                            Ignored if use_dask is True.
        Returns:
            Union[pd.DataFrame, dd.DataFrame]: The loaded (and perhaps unpacked) dataframe.
        """
        df_raw: Union[pd.DataFrame, dd.DataFrame]
        if self.use_dask:
            logger.info("Loading data using Dask...")
            # dask handles large files automagically (both in a single node or distributed)
            if self.file_type == 'tsv':
                df_raw = dd.read_csv(self.file_path,
                                sep='\t',
                                names=self.column_names,
                                header=None if self.column_names else 'infer',
                                blocksize=None # let Dask choose blocksize based on
                                )
            elif self.file_type == 'csv':
                df_raw = dd.read_csv(self.file_path,
                                names=self.column_names,
                                header=None if self.column_names else 'infer',
                                blocksize=None
                                )
            elif self.file_type == 'json':
                # assuming line-delimited JSON
                df_raw = dd.read_json(self.file_path, lines=True, blocksize=None)
            else:
                raise ValueError("Unsupported file type for Dask.")
            logger.info(f"Loaded initial Dask df with {len(df_raw.columns)} columns.")
        else:
            logger.info(f"Loading data using Pandas (chunksize: {chunksize})...")

            # pandas for smaller files or chunked reading (if dataset is too large)
            if self.file_type == 'tsv':
                df_raw = pd.read_csv(self.file_path,
                                   sep='\t', names=self.column_names,
                                   header=None if self.column_names else 'infer',
                                   chunksize=chunksize
                                )
            elif self.file_type == 'csv':
                df_raw = pd.read_csv(self.file_path,
                                   names=self.column_names,
                                   header=None if self.column_names else 'infer',
                                   chunksize=chunksize
                                )
            elif self.file_type == 'json':
                df_raw = pd.read_json(self.file_path, lines=True, chunksize=chunksize)
            else:
                raise ValueError("Unsupported file type for pandas.")

            # handle chunked reading - if chunksize is used, we get an iterator
            if chunksize:
                logger.warning("Chunked reading enabled with Pandas. Unpacking will be applied per chunk.")
                # TODO: Decide how to handle chunked unpacking.
                # Option 1: Raise error - Simplest, forces user to load all data or use Dask.
                # Option 2: Apply unpacking per chunk - requires modifying downstream processing.
                # Option 3: Concatenate chunks first - defeats chunking purpose.
                # DECISION: due to time limitation, i'll just raise error as unpacking implies processing the whole structure.
                raise NotImplementedError("Unpacking multi-labels is not supported with Pandas chunked reading. "
                "Load the full file (chunksize=None) or use Dask (use_dask=True)!")
            logger.info(f"Loaded initial Pandas df with {len(df_raw)} rows and {len(df_raw.columns)} columns.")


        # apply unpacking
        if self.unpack_multi_labels:
            if self.label_column_name not in df_raw.columns:
                 logger.error(f"Label column '{self.label_column_name}' not found in loaded df columns: {df_raw.columns}")
                 raise ValueError(f"Label column '{self.label_column_name}' not found in loaded df.")
            df_processed = self._unpack_dataframe(df_raw)
        else:
            df_processed = df_raw

        return df_processed


    def preprocess_dataframe(
        self,
        df: Union[pd.DataFrame, dd.DataFrame],
        preprocess_col_indices: Optional[List[int]] = None,
        preprocess_col_names: Optional[List[str]] = None,
        cleanup_fn: Optional[Callable[[str], str]] = None
    ) -> Union[pd.DataFrame, dd.DataFrame]:
        """
        Preprocess the dataframe by applying custom cleanup functions to specified columns.

        Args:
            df (Union[pd.DataFrame, dd.DataFrame]): The dataframe to preprocess.
            preprocess_col_indices (List[int], optional): List of column indices
            to apply the cleanup function. Deprecated if preprocess_col_names used.
            preprocess_col_names (List[str], optional): List of column names to apply
                                        the cleanup function. Takes precedence over indices.
            cleanup_fn (Callable[[str], str], optional): A function to clean up data.
                    If None, no cleanup is applied.

        Returns:
            Union[pd.DataFrame, dd.DataFrame]: The preprocessed dataframe.
        """
        if cleanup_fn:
            cols_to_process: List[str] = []
            if preprocess_col_names:
                cols_to_process = [col for col in preprocess_col_names \
                                   if col in df.columns]
                missing_cols = [col for col in preprocess_col_names 
                                if col not in df.columns]
                if missing_cols:
                    logger.warning(f"Columns specified in preprocess_col_names not found: {missing_cols}")

            elif preprocess_col_indices:
                logger.warning("preprocess_col_indices is deprecated, prefer preprocess_col_names.")
                valid_indices = [idx for idx in preprocess_col_indices \
                                if 0 <= idx < len(df.columns)]
                cols_to_process = [df.columns[idx] for idx in valid_indices]
                invalid_indices = [idx for idx in preprocess_col_indices \
                                    if not (0 <= idx < len(df.columns))]
                if invalid_indices:
                     logger.warning(f"Invalid column indices specified: {invalid_indices}")
            else:
                 logger.warning("cleanup_fn provided but no columns specified for preprocessing.")
                 return df # return early


            if not cols_to_process:
                 logger.warning("No valid columns found for preprocessing.")
                 return df

            logger.info(f"Applying cleanup function '{cleanup_fn.__name__}' to columns: {cols_to_process}")

            for col_name in cols_to_process:
                # make sure column exists
                if col_name not in df.columns:
                    logger.warning(f"Column '{col_name}' not found during cleanup, skipping.")
                    continue

                # Check the type of the DataFrame *passed to this function*
                if isinstance(df, dd.DataFrame):
                    # Apply Dask map with meta
                    df[col_name] = df[col_name].map(cleanup_fn, meta=(col_name, 'object'))
                elif isinstance(df, pd.DataFrame):
                    # Apply Pandas apply (convert to string first for safety)
                    df[col_name] = df[col_name].astype(str).apply(cleanup_fn)
                else:
                    # Handle unexpected types if necessary
                    logger.warning(f"DataFrame type {type(df)} not supported for cleanup. Skipping column {col_name}.")

            # check for empty dataframe after potential filtering within cleanup_fn
            # WARNING: with dask, df.empty check is tricky without computing!
            if not self.use_dask and df.empty:
                logger.warning("Pandas dataframe is empty after cleanup. Returning empty dataframe.")
                return pd.DataFrame(columns=df.columns)

        # drop rows where *any* column is NaN/NaT after potential cleanup or unpacking
        logger.info("Dropping rows with any NaN values.")
        df_cleaned = df.dropna(how='any')

        # Reset index if dask or pandas
        if not self.use_dask:
            df_cleaned = df_cleaned.reset_index(drop=True)
            logger.info(f"Resetting index after cleanup. New shape: {df_cleaned.shape}")
        else:
            # dask doesn't require explicit index reset if done within partition
            # TODO: If a global index is needed later, compute() then reset_index() on PD
            pass

        return df_cleaned

    # WARNING: this will load the entire df into memory so be careful with 
    # large datasets & when using Dask along with HF Datasets
    def to_hf_dataset_from_dataframe(self, 
        df: Union[pd.DataFrame, dd.DataFrame]) -> Dataset:
        """ Wrap the pandas dataframe into a Huggingface Dataset.
        Args:
            df (Union[pd.DataFrame, dd.DataFrame]): The df to convert.
        Returns:
            Dataset: Huggingface Dataset.
        """
        if isinstance(df, dd.DataFrame) and self.use_dask:
            logger.info("Converting Dask df to pandas df...")
            df = df.compute()
        return Dataset.from_pandas(df, preserve_index=False)

    def load_to_hf_dataset(
        self,
        streaming: bool = False, 
        features: Optional[Dict[str, Any]] = None, 
        **kwargs
    ) -> Dataset:
        """ Load the dataset into a Huggingface Dataset.

        NOTE: Multi-label unpacking is NOT applied in this method. Use
        load_to_dataframe() followed by to_hf_dataset_from_dataframe()
        if unpacking is needed with Huggingface Datasets.

        Args:
            streaming (bool): Whether to stream the dataset.
            features (Dict[str, Any], optional): Features for the dataset.
            **kwargs: Additional arguments for loading the dataset.
        Returns:
            Dataset: Huggingface Dataset.
        """
        if self.unpack_multi_labels:
            logger.warning("unpack_multi_labels is True, but load_to_hf_dataset does not support "
            "unpacking directly. Labels will remain combined. Use load_to_dataframe first.")
        
        data_files = {self.split: self.file_path}
        load_args = {
            "data_files": data_files,
            "split": self.split,
            "column_names": self.column_names,
            "features": features,
            "streaming": streaming,
            **kwargs
        }

        # hugingface Datasets uses csv for tsv files & need to specify the delimiter
        if self.file_type == 'tsv':
            logger.info("Loading TSV file as CSV with tab delimiter via HF Datasets...")
            load_args["delimiter"] = "\t"
            dataset = load_dataset('csv', **load_args)
        else:
            logger.info(f"Loading {self.file_type} file directly via HF Datasets...")
            dataset = load_dataset(self.file_type, **load_args)

        return dataset

def basic_text_cleanup(text: str) -> str:
    """ Basic text cleanup function to remove unwanted characters and normalize text.
    Args:
        text (str): The text to clean up.
    Returns:
        str: The cleaned-up text.

    TODO: Add more sophisticated text cleanup functions as needed (based on the ML model type)
    """
    if pd.isna(text): # handle NaN values passed to cleanup
        return ""
    return str(text).lower().strip()


# Test the CustomTextDataset class
def main(file_path: str):

    """ 
    Let's run a few test cases how our utility class works with the ATIS dataset

    Cases:
        1. Pandas without unpacking
        2. Pandas with unpacking
        3. Dask with unpacking
        4. Convert unpacked Pandas to HF Dataset
    """

    if not os.path.exists(FILE_PATH):
        logger.error(f"Dataset file not found at: {FILE_PATH}")
        logger.error("Please ensure the path is correct.")
    else:
        logger.info("--- 1: Pandas without Unpacking ---")
        loader_pd = CustomTextDataset(
            FILE_PATH,
            column_names=["atis_text", "atis_labels"],
            label_column_name="atis_labels",
            use_dask=False,
            unpack_multi_labels=False
        )
        df_pd = loader_pd.load_to_dataframe()
        df_pd_clean = loader_pd.preprocess_dataframe(
            df_pd,
            preprocess_col_names=["atis_text"],
            cleanup_fn=basic_text_cleanup
        )
        logger.info(f"Pandas DF (no unpack) Head:\n{df_pd_clean.head()}")
        logger.info(f"Pandas DF (no unpack) Shape: {df_pd_clean.shape}")

        # check a combined label e.g.,
        combined_pd = df_pd_clean[df_pd_clean['atis_labels'].str.contains(r'\+', na=False)]
        logger.info(f"Sample combined labels (no unpack):\n{combined_pd.head()}")


        logger.info("\n\n--- 2: Pandas with Unpacking ---")
        loader_pd_unpack = CustomTextDataset(
            FILE_PATH,
            column_names=["atis_text", "atis_labels"],
            label_column_name="atis_labels",
            use_dask=False,
            unpack_multi_labels=True,
            label_delimiter='+'
        )
        df_pd_unpacked = loader_pd_unpack.load_to_dataframe()
        df_pd_unpacked_clean = loader_pd_unpack.preprocess_dataframe(
            df_pd_unpacked,
            preprocess_col_names=["atis_text"],
            cleanup_fn=basic_text_cleanup
        )
        logger.info(f"Pandas DF (unpack) Head:\n{df_pd_unpacked_clean.head()}")
        logger.info(f"Pandas DF (unpack) Shape: {df_pd_unpacked_clean.shape}")

        # find rows derived from originally combined label
        original_text_example = df_pd[df_pd['atis_labels'].str.contains(r'\+', 
                                na=False)]['atis_text'].iloc[0]
        unpacked_example_rows = df_pd_unpacked_clean[df_pd_unpacked_clean['atis_text'] == original_text_example]
        logger.info(f"Example unpacked rows for text '{original_text_example}':\n{unpacked_example_rows}")

        logger.info("\n\n--- 3: Dask with Unpacking ---")
        loader_dd_unpack = CustomTextDataset(
            FILE_PATH,
            column_names=["atis_text", "atis_labels"],
            label_column_name="atis_labels",
            use_dask=True,
            unpack_multi_labels=True,
            label_delimiter='+'
        )

        ddf_unpacked = loader_dd_unpack.load_to_dataframe()
        ddf_unpacked_clean = loader_dd_unpack.preprocess_dataframe(
            ddf_unpacked,
            preprocess_col_names=["atis_text"],
            cleanup_fn=basic_text_cleanup
        )
        logger.info(f"Dask DF (unpack) defined (lazy operations).")

        logger.info(f"Computing Dask DF head...")
        computed_head = ddf_unpacked_clean.head()
        logger.info(f"Dask DF (unpack) Head:\n{computed_head}")


        logger.info("\n\n--- 4: Convert Unpacked Pandas to HF Dataset ---")
        hf_dataset_unpacked = loader_pd_unpack.to_hf_dataset_from_dataframe(df_pd_unpacked_clean)
        logger.info(f"Unpacked Huggingface ATIS Dataset: {hf_dataset_unpacked}")
        logger.info(f"First example from unpacked HF Dataset:\n{hf_dataset_unpacked[0]}")

        # Show the unpacked example again from HF dataset
        hf_unpacked_example = hf_dataset_unpacked.filter(lambda example: 
                            example['atis_text'] == original_text_example)
        logger.info(f"Example unpacked rows in HF Dataset:\n{hf_unpacked_example[:]}")


if __name__ == "__main__":
    FILE_PATH = "/workspaces/zendesk-mle/coding_task/data/atis/train.tsv"
    main(FILE_PATH)
