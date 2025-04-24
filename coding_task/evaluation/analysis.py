"""
This Exploratory Data Analysis (EDA) script provides a strctured way to undrstand data, its distribution, class characteristics &  potential insights relevant for model building.
"""

import os
import re
import string
from collections import Counter

import pandas as pd
import dask.dataframe as dd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from typing import List, Optional, Union, Any

from coding_task.data.utils import CustomTextDataset
from coding_task.logging_utils import get_logger
from coding_task.constants import LOG_DIR, PLOT_DIR

logger = get_logger(
    logger_name="EDA Utils",
    log_file_path=os.path.join(LOG_DIR, "eda_utils.log"),
    stream=True
)

# TODO: For other major languages, consider using langdetect & extend this accordingky
def download_nltk_data(packages: List[str] = ['punkt', 'stopwords', 'punkt_tab']):
    """Download required NLTK data packages if not already present in the env"""
    try:
        for package in packages:
            nltk.data.find(f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
        logger.info("NLTK data packages already present")
    except LookupError:
        logger.warning("NLTK data packages not found. Attempting to download...")
        try:
            for package in packages:
                nltk.download(package, quiet=True)
            logger.info("NLTK data packages downloaded succssfuly")
        except Exception as e:
            logger.error(f"Failed to download NLTK data: {e}")
            logger.warning("Proceeding w/o NLTK packages. Some analyses might fail")

download_nltk_data()
try:
    # TODO: Extend this for other languages
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    logger.warning("NLTK stopwords not found. Stopword removal will be skipped")
    STOPWORDS = set()


def basic_text_cleanup(text: str) -> str:
    """
    Performs basic text cleaning:
    1. Lowercasing
    2. Removing punctuation
    3. Removing extra whitespace
    Args:
        text (str): Input text string.
    Returns:
        str: Cleaned text string.
    """
    if not isinstance(text, str):
        return "" # handle non-string data
    text = text.lower()

    # keep spaces but remove other puncs
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class TextDatasetExplorer:
    """
    Performs comprehensive EDA on a text classification dataset.

    Attributes:
        data_loader (CustomTextDataset): Instance of the data loader.
        df (pd.DataFrame): Loaded and preprocessed Pandas dataframe.
                           (Note: Dask df is computed to Pandas during init).
        text_column (str): Name of the column containing the text data.
        label_column (str): Name of the column containing the class labels.
        analysis_results (Dict[str, Any]): Stores results of various analyses.
        plot_output_dir (Optional[str]): Directory to save plots.
        split (str): Dataset split being analyzed.
    """
    def __init__(self,
                file_path: str,
                text_column: str,
                label_column: str,
                file_type: Optional[str] = None,
                column_names: Optional[List[str]] = None,
                split: Optional[str] = 'train',
                use_dask: bool = False,
                unpack_multi_labels: bool = False,
                label_delimiter: str = '+',
                apply_basic_cleanup: bool = True,
                plot_output_dir: Optional[str] = PLOT_DIR):
        """
        Initializes the TextDatasetExplorer.

        Args:
            file_path (str): Path to the dataset file.
            text_column (str): Name of the text column.
            label_column (str): Name of the label column (used for analysis AND unpacking).
            file_type (str, optional): Type of the file ('csv', 'tsv', 'json'). Inferred if None.
            column_names (List[str], optional): List of column names. Inferred if None.
            split (str, optional): Dataset split name. Defaults to 'train'.
            use_dask (bool): Whether to use Dask for *initial loading*. Defaults to False.
                             Note: The dataframe is computed to Pandas within this init.
            unpack_multi_labels (bool): Whether to unpack combined labels during loading. Defaults to False.
            label_delimiter (str): Delimiter used for combined labels if unpacking. Defaults to '+'.
            apply_basic_cleanup (bool): Whether to apply basic_text_cleanup to the text column. Defaults to True.
            plot_output_dir (str, optional): Directory to save plots. Defaults to PLOT_DIR constant.
        """
        logger.info(f"Initializing TextDatasetExplorer for file: {file_path}")
        
        self.data_loader = CustomTextDataset(
            file_path=file_path,
            file_type=file_type,
            column_names=column_names,
            split=split,
            use_dask=use_dask,
            unpack_multi_labels=unpack_multi_labels,
            label_column_name=label_column,
            label_delimiter=label_delimiter
        )
        self.text_column = text_column
        self.label_column = label_column
        self.plot_output_dir = plot_output_dir
        self.analysis_results = {}
        self.split = split

        if self.plot_output_dir:
            try:
                os.makedirs(self.plot_output_dir, exist_ok=True)
                logger.info(f"Plots will be saved to: {os.path.abspath(self.plot_output_dir)}")
            except OSError as e:
                logger.error(f"Could not create plot directory {self.plot_output_dir}: {e}")
                self.plot_output_dir = None

        # Load Data (unpacking happens inside, if configured)
        try:
            # Load using the loader (handles dask/pandas & unpacking)
            df_loaded = self.data_loader.load_to_dataframe()

            # force compute if Dask was used for loading
            # This simplifies subsquent analysis steps as self.df will be Pandas
            if use_dask and isinstance(df_loaded, dd.DataFrame):
                logger.warning("Dask used for loading, computing DataFrame to Pandas for EDA...")
                self.df = df_loaded.compute()
                logger.info("Dask compute complete.")
            elif isinstance(df_loaded, pd.DataFrame):
                self.df = df_loaded
            else:
                raise TypeError(f"Loaded data is not a Pandas or Dask DataFrame: {type(df_loaded)}")

            cleanup_func = basic_text_cleanup if apply_basic_cleanup else None

            self.df = self.data_loader.preprocess_dataframe(
                self.df, # pass the potentially computed Pandas df
                preprocess_col_names=[self.text_column],
                cleanup_fn=cleanup_func
            )
            logger.info("Data loaded and basic preprocessing applied.")

            if self.text_column not in self.df.columns:
                 raise ValueError(f"Text column '{self.text_column}' missing after preprocessing")
            if self.label_column not in self.df.columns:
                 raise ValueError(f"Label column '{self.label_column}' missing after preprocessing")

        except Exception as e:
            logger.error(f"Error during data loading or initial preprocessing: {e}")
            raise


    def display_basic_info(self, head_n: int = 5):
        """Displays basic information about the dataset"""
        logger.info("--- EDA: Basic Dataset Information ---")
        # no need for compute checks, self.df is Pandas
        computed_df = self.df

        logger.info(f"Dataset Shape: {computed_df.shape}")
        logger.info(f"Columns: {computed_df.columns.tolist()}")
        logger.info("Data Types:\n" + str(computed_df.dtypes))

        missing_values = computed_df.isnull().sum()
        missing_filtered = missing_values[missing_values > 0]
        if not missing_filtered.empty:
            logger.info("Missing Values per Column:\n" + str(missing_filtered))
        else:
            logger.info("No missing values found")

        logger.info(f"Dataset Head (Top {head_n} rows):\n" + str(computed_df.head(head_n)))
        self.analysis_results['shape'] = computed_df.shape
        self.analysis_results['columns'] = computed_df.columns.tolist()
        self.analysis_results['dtypes'] = computed_df.dtypes
        self.analysis_results['missing_values'] = missing_values

    def analyze_target_distribution(self, plot: bool = True):
        """Analyzes and optionally plots the distribution of the target variable"""
        logger.info("--- Target Variable (Intent) Analysis ---")
        target_counts = self.df[self.label_column].value_counts()

        num_classes = len(target_counts)
        logger.info(f"Number of unique classes (intents): {num_classes}")
        logger.info("Class Distribution:\n" + str(target_counts))

        self.analysis_results['target_distribution'] = target_counts
        self.analysis_results['num_classes'] = num_classes

        if plot and self.plot_output_dir:
            fig, ax = plt.subplots(figsize=(12, max(6, num_classes // 4)))
            sns.barplot(x=target_counts.values,
                        y=target_counts.index, 
                        orient='h', 
                        palette="viridis", 
                        ax=ax
                    )
            ax.set_title(f'Distribution of Target Variable ({self.label_column})')
            ax.set_xlabel('Frequency')
            ax.set_ylabel('Class Label')
            plt.tight_layout()

            filename = f"target_distribution_{self.label_column}_{self.split}.png"
            filepath = os.path.join(self.plot_output_dir, filename)
            try:
                fig.savefig(filepath, bbox_inches='tight', dpi=150)
                logger.info(f"Saved plot: {filepath}")
            except Exception as e:
                logger.error(f"Failed to save plot {filepath}: {e}")
            plt.close(fig)

        # check for class imbalance
        if num_classes > 1:
            min_count = target_counts.min()
            max_count = target_counts.max()
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            logger.info(f"Class Imbalance Ratio (Max/Min): {imbalance_ratio:.2f}")
            if imbalance_ratio > 10:
                logger.warning("Significant class imbalance detected. Consider resampling techniques")
            self.analysis_results['imbalance_ratio'] = imbalance_ratio


    def analyze_text_length(self, plot: bool = True):
        """Analyzes and optionally plots the distribution of text lengths."""
        logger.info("--- Text Length Analysis ---")

        char_lengths_pd = self.df[self.text_column].astype(str).str.len()
        try:
            word_lengths_pd = self.df[self.text_column].astype(str).apply(
                lambda x: len(word_tokenize(x)) if pd.notna(x) else 0
            )
        except LookupError:
            logger.error("NLTK 'punkt' tokenizer data not found. Cannot calculate word lengths accurately.")
            word_lengths_pd = pd.Series([np.nan] * len(self.df))

        logger.info("Character Length Statistics:\n" + str(char_lengths_pd.describe()))
        logger.info("Word Length Statistics:\n" + str(word_lengths_pd.describe()))
        self.analysis_results['char_length_stats'] = char_lengths_pd.describe()
        self.analysis_results['word_length_stats'] = word_lengths_pd.describe()

        if plot and self.plot_output_dir:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            sns.histplot(char_lengths_pd, bins=50, ax=axes[0], kde=True)
            axes[0].set_title(f'Character Lengths ({self.text_column})')
            axes[0].set_xlabel('Characters'); axes[0].set_ylabel('Frequency')
            sns.histplot(word_lengths_pd, bins=50, ax=axes[1], kde=True)
            axes[1].set_title(f'Word Lengths ({self.text_column})')
            axes[1].set_xlabel('Words'); axes[1].set_ylabel('Frequency')
            plt.tight_layout()

            filename = f"text_length_distribution_{self.text_column}_{self.split}.png"
            filepath = os.path.join(self.plot_output_dir, filename)
            try:
                fig.savefig(filepath, bbox_inches='tight', dpi=150)
                logger.info(f"Saved plot: {filepath}")
            except Exception as e:
                logger.error(f"Failed to save plot {filepath}: {e}")
            plt.close(fig)

    def _get_tokens(self,
        series: pd.Series,
        remove_stopwords: bool = True
    ) -> List[str]:
        """Helper function to tokenize a series of text and flatten the list"""
        all_tokens = []
        computed_series = series
        logger.info(f"Tokenizing {len(computed_series)} text entries...")
        try:
            tokenized_series = computed_series.astype(str).apply(word_tokenize)
        except LookupError:
            logger.error("NLTK 'punkt' tokenizer data not found. Cannot perform token-based analysis.")
            return []

        for tokens in tokenized_series:
            if remove_stopwords:
                tokens = [token for token in tokens if token not in STOPWORDS and token.isalnum()]
            else:
                tokens = [token for token in tokens if token.isalnum()]
            all_tokens.extend(tokens)
        logger.info(f"Total tokens extracted: {len(all_tokens)}")
        return all_tokens

    def analyze_vocabulary(self, 
        top_n: int = 25, 
        plot: bool = True
    ) -> None:
        """Analyzes vocabulary size and frequency distribution."""
        logger.info("--- Vocabulary Analysis ---")

        # get tokens (stopwords removed by default for this analysis)
        all_tokens = self._get_tokens(self.df[self.text_column], remove_stopwords=True)
        if not all_tokens:
             logger.warning("No tokens found, skipping vocabulary analysis.")
             return

        vocab = set(all_tokens)
        vocab_size = len(vocab)
        logger.info(f"Total Vocabulary Size (unique tokens, stopwords removed): {vocab_size}")

        # calculate frequency distribution
        freq_dist = Counter(all_tokens)
        most_common = freq_dist.most_common(top_n)
        least_common = freq_dist.most_common()[:-top_n-1:-1]

        logger.info(f"Top {top_n} Most Common Words:\n{most_common}")
        logger.info(f"Top {top_n} Least Common Words:\n{least_common}")

        self.analysis_results['vocabulary_size'] = vocab_size
        self.analysis_results['most_common_words'] = most_common
        self.analysis_results['least_common_words'] = least_common

        if plot and self.plot_output_dir and most_common:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x=[word for word, count in most_common],
                        y=[count for word, count in most_common],
                        palette="magma", ax=ax)
            ax.set_title(f'Top {top_n} Most Common Words (Stopwords Removed)')
            ax.set_xlabel('Words'); ax.set_ylabel('Frequency')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            filename = f"top_{top_n}_words_{self.split}.png"
            filepath = os.path.join(self.plot_output_dir, filename)
            try:
                fig.savefig(filepath, bbox_inches='tight', dpi=150)
                logger.info(f"Saved plot: {filepath}")
            except Exception as e:
                logger.error(f"Failed to save plot {filepath}: {e}")
            plt.close(fig)

    def analyze_ngrams(self, 
        n: int = 2, 
        top_n: int = 25, 
        remove_stopwords: bool = True, 
        plot: bool = True
    ) -> None:
        """Analyzes n-grams and saves frequency plot if enabled"""
        if n < 2:
            logger.warning("N-gram analysis requires n >= 2")
            return
        stopword_status = "stopwords_removed" if remove_stopwords else "stopwords_included"
        logger.info(f"--- N-gram Analysis (n={n}, {stopword_status.replace('_', ' ')}) ---")

        computed_series = self.df[self.text_column]
        all_ngrams = []
        num_processed = 0
        try:
            for text in computed_series.astype(str):
                 if pd.notna(text) and isinstance(text, str):
                    current_tokens = word_tokenize(text)
                    if remove_stopwords: current_tokens = [tok for tok in \
                    current_tokens if tok not in STOPWORDS and tok.isalnum()]
                    else: 
                        current_tokens = [tok for tok in current_tokens if tok.isalnum()]
                    if len(current_tokens) >= n: 
                        all_ngrams.extend(list(ngrams(current_tokens, n)))
                 num_processed += 1
                 if num_processed % 5000 == 0: 
                    logger.debug(f"Processed {num_processed}/{len(computed_series)} for {n}-grams...")
        except LookupError: 
            logger.error("NLTK data for ngrams not found")
            return

        if not all_ngrams:
            logger.warning(f"No {n}-grams generated ({stopword_status.replace('_', ' ')})")
            return

        ngram_freq = Counter(all_ngrams)
        most_common_ngrams = ngram_freq.most_common(top_n)
        most_common_ngrams_str = [(" ".join(gram), count) for gram, count in most_common_ngrams]
        logger.info(f"Top {top_n} Most Common {n}-grams:\n{most_common_ngrams_str}")
        self.analysis_results[f'most_common_{n}grams_{stopword_status}'] = most_common_ngrams_str

        if plot and self.plot_output_dir and most_common_ngrams_str:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x=[gram for gram, count in most_common_ngrams_str],
                        y=[count for word, count in most_common_ngrams_str],
                        palette="coolwarm", ax=ax)
            ax.set_title(f'Top {top_n} Most Common {n}-grams ({stopword_status.replace("_", " ")})')
            ax.set_xlabel(f'{n}-grams'); ax.set_ylabel('Frequency')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            filename = f"top_{top_n}_{n}grams_{stopword_status}_{self.split}.png"
            filepath = os.path.join(self.plot_output_dir, filename)
            try:
                fig.savefig(filepath, bbox_inches='tight', dpi=150)
                logger.info(f"Saved plot: {filepath}")
            except Exception as e:
                logger.error(f"Failed to save plot {filepath}: {e}")
            plt.close(fig)

    def analyze_length_by_class(self, plot: bool = True):
        """Analyzes text length per class and saves plots if enabled."""
        logger.info("--- Text Length Analysis by Class ---")
        computed_df = self.df.copy()

        if 'char_length' not in computed_df.columns: 
            computed_df['char_length'] = computed_df[self.text_column].astype(str).str.len()
        if 'word_length' not in computed_df.columns:
            try: 
                computed_df['word_length'] = computed_df[self.text_column].astype(str).apply(lambda x: len(word_tokenize(x)) if pd.notna(x) and isinstance(x, str) else 0)
            except LookupError: 
                logger.error("NLTK 'punkt' not found.")
                computed_df['word_length'] = np.nan
        computed_df['word_length'] = pd.to_numeric(computed_df['word_length'], errors='coerce')
        avg_word_len_class = computed_df.groupby(self.label_column)['word_length'].mean().sort_values(ascending=False)

        logger.info("Average Word Length per Class:\n" + str(avg_word_len_class))
        self.analysis_results['avg_word_length_by_class'] = avg_word_len_class

        if plot and self.plot_output_dir:
            num_classes_to_plot = 20
            if computed_df[self.label_column].nunique() > num_classes_to_plot:
                top_classes = computed_df[self.label_column].value_counts().nlargest(num_classes_to_plot).index.tolist()
                plot_df = computed_df[computed_df[self.label_column].isin(top_classes)]
                plot_title_suffix = f"(Top {num_classes_to_plot} Classes)"
            else:
                plot_df = computed_df
                # sort for consistent order
                top_classes = sorted(plot_df[self.label_column].unique().tolist())
                plot_title_suffix = "(All Classes)"

            fig, axes = plt.subplots(1, 2, figsize=(18, 7))
            sns.boxplot(data=plot_df, 
                        x=self.label_column, 
                        y='char_length', 
                        ax=axes[0], 
                        palette="Set2", 
                        order=top_classes)
            axes[0].set_title(f'Character Length by Class {plot_title_suffix}')
            axes[0].set_xlabel('Class Label'); axes[0].set_ylabel('Characters')
            axes[0].tick_params(axis='x', rotation=45)
            sns.boxplot(data=plot_df, 
                        x=self.label_column, 
                        y='word_length', 
                        ax=axes[1], 
                        palette="Set3", 
                        order=top_classes)
            axes[1].set_title(f'Word Length by Class {plot_title_suffix}')
            axes[1].set_xlabel('Class Label'); axes[1].set_ylabel('Words')
            axes[1].tick_params(axis='x', rotation=45)
            plt.tight_layout()

            filename = f"length_by_class_{self.label_column}_{self.split}.png"
            filepath = os.path.join(self.plot_output_dir, filename)
            try:
                fig.savefig(filepath, bbox_inches='tight', dpi=150)
                logger.info(f"Saved plot: {filepath}")
            except Exception as e:
                logger.error(f"Failed to save plot {filepath}: {e}")
            plt.close(fig)

    def analyze_zipf_distribution(self, 
        remove_stopwords: bool = False, 
        plot: bool = True
    ) -> None:
        """Analyzes Zipf distribution and saves plot if enabled
        Zipf's Law states that the frequency of any word is inversely proportional to its rank in the frequency table.
        The most common word will occur with a frequency of 1/rank.
        """
        stopword_status = "stopwords_removed" if remove_stopwords else "stopwords_included"

        logger.info(f"--- Zipf's Law Analysis ({stopword_status.replace('_', ' ')}) ---")
        all_tokens = self._get_tokens(self.df[self.text_column], remove_stopwords=remove_stopwords)

        if not all_tokens: 
            logger.warning(f"No tokens found ({stopword_status.replace('_', ' ')}).")
            return
        freq_dist = Counter(all_tokens)
        if not freq_dist:
            logger.warning(f"Token counter empty ({stopword_status.replace('_', ' ')}).")
            return
        sorted_freqs = sorted(freq_dist.items(), key=lambda item: item[1], reverse=True)
        frequencies = [count for word, count in sorted_freqs]
        ranks = list(range(1, len(frequencies) + 1))

        logger.info(f"Found {len(ranks)} unique terms ({stopword_status.replace('_', ' ')}). \
                    Freq range: {frequencies[0]} to {frequencies[-1]}")
        result_key = f'zipf_data_{stopword_status}'
        self.analysis_results[result_key] = {'ranks': ranks, 'frequencies': frequencies}

        if plot and self.plot_output_dir:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.loglog(ranks, 
                    frequencies, 
                    marker='.', 
                    markersize=4, 
                    linestyle='', 
                    label='Observed Frequencies')
            
            if ranks and frequencies:
                 C = frequencies[0]; x_ref = np.array([ranks[0], ranks[-1]], dtype=float)
                 x_ref = x_ref[x_ref > 0]
                 if len(x_ref) > 0: y_ref = C / x_ref; ax.loglog(x_ref, y_ref, color='r', linestyle='--', label='Ideal Zipf (slope = -1)')

            ax.set_title(f"Zipf Distribution ({stopword_status.replace('_', ' ')}) - Log-Log Scale")
            ax.set_xlabel("Rank (Log Scale)"); ax.set_ylabel("Frequency (Log Scale)")
            ax.grid(True, which="both", ls="--", linewidth=0.5); ax.legend()
            plt.tight_layout()

            filename = f"zipf_distribution_{stopword_status}_{self.split}.png"
            filepath = os.path.join(self.plot_output_dir, filename)
            try:
                fig.savefig(filepath, bbox_inches='tight', dpi=150)
                logger.info(f"Saved plot: {filepath}")
            except Exception as e:
                logger.error(f"Failed to save plot {filepath}: {e}")
            plt.close(fig)

    def run_eda(self, run_deep_analysis: bool = True):
        """Runs the full EDA pipeline."""
        logger.info("--- Starting Exploratory Data Analysis ---")
        self.display_basic_info()
        self.analyze_target_distribution(plot=True)
        self.analyze_text_length(plot=True)
        self.analyze_vocabulary(plot=True)
        self.analyze_ngrams(n=2, top_n=25, plot=True) # Bi-grams
        self.analyze_ngrams(n=3, top_n=25, plot=True) # Tri-grams

        if run_deep_analysis:
            self.analyze_length_by_class(plot=True)
            self.analyze_zipf_distribution(remove_stopwords=False, plot=True)
            self.analyze_zipf_distribution(remove_stopwords=True, plot=True)
            # TODO: add more deeper analysis funcs like TF-IDF analysis per class, 
            # topic modeling, word embeddings clusters, tsne, etc.

        logger.info("=== EDA Finished ===")


        # Save the analysis results to a file
        if self.plot_output_dir:
            # Include split name in results filename
            results_file = os.path.join(self.plot_output_dir, f"eda_analysis_results_{self.split}.txt")
            try:
                with open(results_file, 'w') as f:
                    for key, value in self.analysis_results.items():
                        # Handle potential complex objects in results (like Series/DF dtypes)
                        f.write(f"{key}:\n")
                        if isinstance(value, (pd.Series, pd.DataFrame)):
                             f.write(f"{value.to_string()}\n\n")
                        else:
                             f.write(f"{value}\n\n")
                logger.info(f"Saved analysis results to {results_file}")
            except Exception as e:
                 logger.error(f"Failed to save analysis results to {results_file}: {e}")
        return self.analysis_results


if __name__ == "__main__":
    FILE_PATH = "/workspaces/zendesk-mle/coding_task/data/atis/test.tsv"
    SPLIT_NAME = 'test'

    TEXT_COLUMN = 'atis_text'
    LABEL_COLUMN = 'atis_labels'
    COLUMN_NAMES = [TEXT_COLUMN, LABEL_COLUMN]

    USE_DASK_LOADING = True # use Dask for initial load - will be computed to PD
    UNPACK_LABELS = False # set to True to run EDA on unpacked data
    APPLY_CLEANUP = True

    logger.info(f"Running EDA for {SPLIT_NAME} split. Unpack labels: {UNPACK_LABELS}")

    try:
        explorer = TextDatasetExplorer(
            file_path=FILE_PATH,
            text_column=TEXT_COLUMN,
            label_column=LABEL_COLUMN,
            column_names=COLUMN_NAMES,
            use_dask=USE_DASK_LOADING,
            apply_basic_cleanup=APPLY_CLEANUP,
            split=SPLIT_NAME,
            unpack_multi_labels=UNPACK_LABELS,
            label_delimiter='+'
        )
        explorer.run_eda(run_deep_analysis=True)

    except ImportError:
        logger.error("Failed to run EDA due to missing CustomTextDataset class")
    except FileNotFoundError:
            logger.error(f"File not found: {FILE_PATH}")
    except ValueError as ve:
            logger.error(f"Configuration or Data Error: {ve}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during EDA: {e}", exc_info=True)

