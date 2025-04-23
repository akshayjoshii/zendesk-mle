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
        use_dask: bool = False
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
            use_dask (bool): Whether to use Dask for loading large files. 
                            Default is False.
        """
        self.file_path = file_path
        self.file_type = file_type or self._infer_file_type()
        self.column_names = column_names
        self.use_dask = use_dask
        self.split = split

    def __post_init__(self):
        logger.info(f"CustomTextDataset initialized with file_path: {self.file_path}, "
                    f"file_type: {self.file_type}, column_names: {self.column_names}, "
                    f"use_dask: {self.use_dask}")

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

    def load_to_dataframe(self, 
        chunksize: Optional[int] = None
    ) -> Union[pd.DataFrame, dd.DataFrame]:
        """
        Load the dataset into a pandas or Dask dataframe.
        Args:
            chunksize (int, optional): Number of rows to read at a time.
                            If None, the entire file is read into memory.
        Returns:
            Union[pd.DataFrame, dd.DataFrame]: The loaded dataframe.
        """
        if self.use_dask:
            # dask handles large files automagically (both in a single node or distributed)
            if self.file_type == 'tsv':
                df = dd.read_csv(self.file_path, 
                                sep='\t', 
                                names=self.column_names, 
                                header=None if self.column_names else 'infer'
                                )
            elif self.file_type == 'csv':
                df = dd.read_csv(self.file_path, 
                                names=self.column_names, 
                                header=None if self.column_names else 'infer'
                                )
            elif self.file_type == 'json':
                df = dd.read_json(self.file_path, 
                                lines=True)
            else:
                raise ValueError("Unsupported file type for Dask.")
            logger.info(f"Loaded Dask dataframe with {len(df)} rows and {len(df.columns)} columns.")
            return df
        else:
            logger.info(f"Loading file with PD chunksize: {chunksize}")
            # pandas for smaller files or chunked reading (if dataset is too large)
            if self.file_type == 'tsv':
                return pd.read_csv(self.file_path, 
                                   sep='\t', names=self.column_names, 
                                   header=None if self.column_names else 'infer', 
                                   chunksize=chunksize
                                )
            elif self.file_type == 'csv':
                return pd.read_csv(self.file_path, 
                                   names=self.column_names, 
                                   header=None if self.column_names else 'infer', 
                                   chunksize=chunksize
                                )
            elif self.file_type == 'json':
                return pd.read_json(self.file_path, lines=True, chunksize=chunksize)
            else:
                raise ValueError("Unsupported file type for pandas.")

    def preprocess_dataframe(
        self, 
        df: Union[pd.DataFrame, dd.DataFrame], 
        preprocess_col_indices: List[int] = [0], 
        cleanup_fn: Optional[Callable[[str], str]] = None
    ) -> Union[pd.DataFrame, dd.DataFrame]:
        """
        Preprocess the dataframe by applying custom cleanup functions to specified columns.
        Args:
            df (Union[pd.DataFrame, dd.DataFrame]): The dataframe to preprocess.
            preprocess_col_indices (List[int]): List of column indices to apply the cleanup function.
            cleanup_fn (Callable[[str], str], optional): A function to clean up data.
                    If None, no cleanup is applied. Data could be of any modality
        Returns:
            Union[pd.DataFrame, dd.DataFrame]: The preprocessed dataframe.
        """
        if cleanup_fn:
            logger.info(f"Preprocessing dataframe with cleanup function: {cleanup_fn.__name__}")
            logger.info(f"Preprocessing columns: {preprocess_col_indices}")
            logger.info(f"Dataframe before cleanup has {len(df)} rows and {len(df.columns)} columns.")
            for idx in preprocess_col_indices:
                col = df.columns[idx]  # Get the column name at this index
                if self.use_dask:
                    df[col] = df[col].map(cleanup_fn, meta=('x', 'object'))
                else:
                    df[col] = df[col].apply(cleanup_fn)
        
        # If cleanup deletes all rows, return an empty DataFrame with the same columns
        if hasattr(df, "empty") and df.empty:
            logger.warning("Dataframe is empty after cleanup. Returning empty dataframe.")
            return pd.DataFrame(columns=df.columns)
        else:
            logger.info(f"Dataframe after cleanup has {len(df)} rows and {len(df.columns)} columns.")
            # drop any empty rows (if any col is NAN, then drop the row)
            df = df.dropna(how='any')
            df = df.reset_index(drop=True)
            return df

    # WARNING: this will load the entire dataframe into memory so be careful with 
    # large datasets & when using Dask along with HF Datasets
    def to_hf_dataset_from_dataframe(self, 
        df: Union[pd.DataFrame, dd.DataFrame]) -> Dataset:
        """ Wrap the pandas dataframe into a Huggingface Dataset.
        Args:
            df (Union[pd.DataFrame, dd.DataFrame]): The dataframe to convert.
        Returns:
            Dataset: Huggingface Dataset.
        """
        if isinstance(df, dd.DataFrame) and self.use_dask:
            logger.info("Converting Dask dataframe to pandas dataframe...")
            df = df.compute()
        return Dataset.from_pandas(df, preserve_index=False)

    def load_to_hf_dataset(
        self,
        streaming: bool = False, 
        features: Optional[Dict[str, Any]] = None, 
        **kwargs
    ) -> Dataset:
        """ Load the dataset into a Huggingface Dataset.
        Args:
            streaming (bool): Whether to stream the dataset.
            features (Dict[str, Any], optional): Features for the dataset.
            **kwargs: Additional arguments for loading the dataset.
        Returns:
            Dataset: Huggingface Dataset.
        """
        data_files = {self.split: self.file_path}

        # hugingface Datasets uses csv for tsv files & need to specify the delimiter
        if self.file_type == 'tsv':
            logger.info("Loading TSV file as CSV with tab delimiter...")
            delimiter = "\t"
            dataset = load_dataset(
                'csv',
                data_files=data_files,
                split=self.split,
                column_names=self.column_names,
                features=features,
                streaming=streaming,
                delimiter=delimiter,
                **kwargs
            )
        else:
            logger.info(f"Loading {self.file_type} file directly as HF dataset...")
            dataset = load_dataset(
                self.file_type,
                data_files=data_files,
                split=self.split,
                column_names=self.column_names,
                features=features,
                streaming=streaming,
                **kwargs
        )
        return dataset

def basic_text_cleanup(text: str) -> str:
    """ Basic text cleanup function to remove unwanted characters and normalize text.
    Args:
        text (str): The text to clean up.
    Returns:
        str: The cleaned-up text.

    TODO: Add more sophisticated text cleanup functions as needed (based on the ML model type)
    """
    return text.lower().strip()



if __name__ == "__main__":
    FILE_PATH = "/workspaces/zendesk-mle/coding_task/data/atis/train.tsv"

    # let's try with Dask
    loader = CustomTextDataset(FILE_PATH,
                               column_names=["atis_text", "atis_labels"], 
                               use_dask=True)
    df = loader.load_to_dataframe()
    df = loader.preprocess_dataframe(df, 
            preprocess_col_indices=[0], cleanup_fn=basic_text_cleanup)
    hf_dataset = loader.to_hf_dataset_from_dataframe(df)
    logger.info(f"Huggingface ATIS Dataset: {hf_dataset}")

"""     # or, for direct Huggingface streaming (for very large files)
    hf_streaming_dataset = loader.load_to_hf_dataset(streaming=True)
    print(hf_streaming_dataset) # yields 1 sample """
