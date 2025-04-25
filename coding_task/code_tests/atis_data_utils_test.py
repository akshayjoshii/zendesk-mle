import pytest
import pandas as pd
from datasets import Dataset

from coding_task.data.utils import CustomTextDataset, basic_text_cleanup


# some sample test data covering a few different cases
SAMPLE_DATA_TSV = """text1\tlabelA
text2 with UPPERCASE\tlabelB+labelC
text3\tlabelA
text4 needs cleanup  \tlabelC+labelD+labelE
text5\tlabelB
text6 only plus\t+
text7 empty label\t
text8 trailing plus\tlabelF+
text9 leading plus\t+labelG
text10 single\tlabelH
text11 multi again\tlabelA+labelC
"""

SAMPLE_DATA_TSV_PIPE_DELIM = """text1|labelA
text2 with UPPERCASE|labelB|labelC
text3|labelA
text4 needs cleanup  |labelC|labelD|labelE
text5|labelB
"""

COLUMN_NAMES = ["text", "label"]
LABEL_COL = "label"


# Fixtures
@pytest.fixture
def create_test_file(tmp_path):
    """Creates a temporary TSV file with sample data."""
    def _create_file(filename="test_data.tsv", data=SAMPLE_DATA_TSV, delimiter='\t'):
        file_path = tmp_path / filename
        # make sure data uses the correct delimiter if specified differently
        if delimiter != '\t':
             data_to_write = data.replace('\t', delimiter)
        else:
             data_to_write = data
        file_path.write_text(data_to_write)
        return str(file_path)
    return _create_file

def test_initialization_errors():
    """Tests that initialization raises errors for invalid unpack config."""
    # Missing label_column_name when unpacking
    with pytest.raises(ValueError, match="label_column_name must be provided"):
        CustomTextDataset("dummy.tsv", unpack_multi_labels=True)

    # label_column_name not in column_names when unpacking
    with pytest.raises(ValueError, match="not found in provided column_names"):
        CustomTextDataset(
            "dummy.tsv",
            column_names=["col1", "col2"],
            label_column_name="wrong_label_col",
            unpack_multi_labels=True
        )

@pytest.mark.parametrize("use_dask", [False, True], ids=["Pandas", "Dask"])
def test_loading_no_unpack(create_test_file, use_dask):
    """Tests loading without unpacking for both Pandas and Dask."""
    file_path = create_test_file()
    loader = CustomTextDataset(
        file_path,
        column_names=COLUMN_NAMES,
        label_column_name=LABEL_COL,
        use_dask=use_dask,
        unpack_multi_labels=False
    )
    df = loader.load_to_dataframe()

    if use_dask:
        df = df.compute() # need to compute dask df for assertons

    assert isinstance(df, pd.DataFrame)
    # check shape - should match original number of lines
    assert df.shape[0] == SAMPLE_DATA_TSV.strip().count('\n') + 1
    assert df.shape[1] == len(COLUMN_NAMES)
    assert list(df.columns) == COLUMN_NAMES

    # check specific multi-label row is preserved
    assert "labelB+labelC" in df[LABEL_COL].tolist()
    assert df[df['text'] == 'text2 with UPPERCASE'][LABEL_COL].iloc[0] == "labelB+labelC"

@pytest.mark.parametrize("use_dask", [False, True], ids=["Pandas", "Dask"])
def test_unpacking_correctness(create_test_file, use_dask):
    """Tests the core unpacking logic for correctness (shape, content)."""
    file_path = create_test_file()
    loader = CustomTextDataset(
        file_path,
        column_names=COLUMN_NAMES,
        label_column_name=LABEL_COL,
        use_dask=use_dask,
        unpack_multi_labels=True,
        label_delimiter='+'
    )
    df_unpacked = loader.load_to_dataframe()

    if use_dask:
        df_unpacked = df_unpacked.compute()

    # assertions
    # expected rows: 1+2+1+3+1+1+1+1+1+1+2 = 16 (explode handles single/empty labels correctly)
    expected_rows_after_unpack = 16
    assert df_unpacked.shape[0] == expected_rows_after_unpack, "Unexpected number of rows after unpacking"
    assert list(df_unpacked.columns) == COLUMN_NAMES

    # 1. Check a row that was originally multi-label ('text2')
    text2_rows = df_unpacked[df_unpacked['text'] == 'text2 with UPPERCASE']
    assert len(text2_rows) == 2, "text2 should have been duplicated into 2 rows"
    assert set(text2_rows[LABEL_COL].tolist()) == {'labelB', 'labelC'}, "text2 rows have incorrect labels"

    # 2. Check another multi-label row ('text4')
    text4_rows = df_unpacked[df_unpacked['text'] == 'text4 needs cleanup  ']
    assert len(text4_rows) == 3, "text4 should have been duplicated into 3 rows"
    assert set(text4_rows[LABEL_COL].tolist()) == {'labelC', 'labelD', 'labelE'}, "text4 rows have incorrect labels"

    # 3. Check a row that was originally single-label ('text1')
    text1_rows = df_unpacked[df_unpacked['text'] == 'text1']
    assert len(text1_rows) == 1, "text1 should remain a single row"
    assert text1_rows[LABEL_COL].iloc[0] == 'labelA', "text1 label incorrect"

    # 4. Check that NO combined labels remain in the label column
    assert not df_unpacked[LABEL_COL].str.contains(r'\+', na=False).any(), "Combined labels still exist after unpacking"

    # 5. Check handling edge cases (empty strings from split)
    # 'text6 only plus\t+' -> splits to ['', ''] -> explodes to two rows with '' label
    text6_rows = df_unpacked[df_unpacked['text'] == 'text6 only plus']
    assert len(text6_rows) == 2, "text6 should explode into 2 rows"
    assert set(text6_rows[LABEL_COL].tolist()) == {''}, "text6 rows should have empty string label"

    # 'text7 empty label\t' -> splits to [''] -> explodes to one row with '' label
    text7_rows = df_unpacked[df_unpacked['text'] == 'text7 empty label']
    assert len(text7_rows) == 1, "text7 should explode into 1 row"
    assert text7_rows[LABEL_COL].iloc[0] == '', "text7 row should have empty string label"

    # 'text8 trailing plus\tlabelF+' -> splits to ['labelF', ''] -> explodes to two rows
    text8_rows = df_unpacked[df_unpacked['text'] == 'text8 trailing plus']
    assert len(text8_rows) == 2, "text8 should explode into 2 rows"
    assert set(text8_rows[LABEL_COL].tolist()) == {'labelF', ''}, "text8 rows have incorrect labels"

    # 'text9 leading plus\t+labelG' -> splits to ['', 'labelG'] -> explodes to two rows
    text9_rows = df_unpacked[df_unpacked['text'] == 'text9 leading plus']
    assert len(text9_rows) == 2, "text9 should explode into 2 rows"
    assert set(text9_rows[LABEL_COL].tolist()) == {'', 'labelG'}, "text9 rows have incorrect labels"


def test_unpacking_different_delimiter(create_test_file):
    """Tests unpacking works with a non-default delimiter."""
    file_path = create_test_file(filename="pipe_data.tsv", data=SAMPLE_DATA_TSV_PIPE_DELIM, delimiter='|')
    loader = CustomTextDataset(
        file_path,
        file_type='tsv',
        column_names=COLUMN_NAMES,
        label_column_name=LABEL_COL,
        use_dask=False, # let's keep it simple (because of time constrainsts) & just test with Pandas
        unpack_multi_labels=True,
        label_delimiter='|'
    )
    # override file type infrence for this test case since we used .tsv extension
    loader.file_type = 'csv'
    loader.delimiter = '|' # set delimiter for pandas loading

    # Need to adjust loading method call for pandas CSV with delimiter
    df_raw = pd.read_csv(
        loader.file_path,
        sep=loader.delimiter,
        names=loader.column_names,
        header=None
    )
    df_unpacked = loader._unpack_dataframe(df_raw) # Test unpack directly

    # Assertions
    # Expected rows: 1+2+1+3+1 = 8
    assert df_unpacked.shape[0] == 8, "Unexpected rows after unpacking with pipe delimiter"

    text2_rows = df_unpacked[df_unpacked['text'] == 'text2 with UPPERCASE']
    assert len(text2_rows) == 2
    assert set(text2_rows[LABEL_COL].tolist()) == {'labelB', 'labelC'}

    text4_rows = df_unpacked[df_unpacked['text'] == 'text4 needs cleanup  ']
    assert len(text4_rows) == 3
    assert set(text4_rows[LABEL_COL].tolist()) == {'labelC', 'labelD', 'labelE'}

    assert not df_unpacked[LABEL_COL].str.contains(r'\|', na=False).any(), "Pipe delimiter still exists"


@pytest.mark.parametrize("use_dask", [False, True], ids=["Pandas", "Dask"])
def test_unpacking_and_preprocessing(create_test_file, use_dask):
    """Tests that preprocessing is applied correctly AFTER unpacking."""
    file_path = create_test_file()
    loader = CustomTextDataset(
        file_path,
        column_names=COLUMN_NAMES,
        label_column_name=LABEL_COL,
        use_dask=use_dask,
        unpack_multi_labels=True,
        label_delimiter='+'
    )
    df_unpacked = loader.load_to_dataframe()
    df_processed = loader.preprocess_dataframe(
        df_unpacked,
        preprocess_col_names=["text"], # Specify by name
        cleanup_fn=basic_text_cleanup
    )

    if use_dask:
        df_processed = df_processed.compute()

    # 1. Check text cleanup on an unpacked row ('text2')
    text2_rows = df_processed[df_processed[LABEL_COL] == 'labelB']
    assert len(text2_rows) == 1
    assert text2_rows['text'].iloc[0] == 'text2 with uppercase', "Cleanup not applied correctly to unpacked text2"

    # 2. Check text cleanup on another unpacked row ('text4')
    text4_rows = df_processed[df_processed[LABEL_COL] == 'labelD']
    assert len(text4_rows) == 1
    assert text4_rows['text'].iloc[0] == 'text4 needs cleanup', "Cleanup not applied correctly to unpacked text4"

    # 3. Check empty label rows still exist (assuming dropna doesn't remove them)
    assert '' in df_processed[LABEL_COL].unique()
    text7_processed = df_processed[df_processed['text'] == 'text7 empty label']
    assert len(text7_processed) == 1
    assert text7_processed[LABEL_COL].iloc[0] == ''


def test_to_hf_dataset_after_unpack(create_test_file):
    """Tests conversion to Hugging Face Dataset after unpacking (Pandas)."""
    file_path = create_test_file()
    loader = CustomTextDataset(
        file_path,
        column_names=COLUMN_NAMES,
        label_column_name=LABEL_COL,
        use_dask=False,
        unpack_multi_labels=True,
        label_delimiter='+'
    )
    df_unpacked = loader.load_to_dataframe()
    df_processed = loader.preprocess_dataframe(
        df_unpacked,
        preprocess_col_names=["text"],
        cleanup_fn=basic_text_cleanup
    )

    hf_dataset = loader.to_hf_dataset_from_dataframe(df_processed)

    assert isinstance(hf_dataset, Dataset)
    assert len(hf_dataset) == df_processed.shape[0] # check row count matches
    assert list(hf_dataset.features.keys()) == COLUMN_NAMES

    # verify content of unpacked example in rhe HF Dataset
    # Find idx corresponding to original 'text2' after processing
    text2_example_in_hf = hf_dataset.filter(lambda example: example['text'] == 'text2 with uppercase')
    assert len(text2_example_in_hf) == 2
    assert set(text2_example_in_hf[LABEL_COL]) == {'labelB', 'labelC'}

