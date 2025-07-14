import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
import polars as pl

class BagOfWordsFormatter:
    def __init__(
        self,
        term_encoder: LabelEncoder
    ):
        self.term_encoder = term_encoder

    def _convert_to_polars(self, bag_of_words: dict[str, int]) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "term": list(bag_of_words.keys()),
                "exists": list(bag_of_words.values())
            }
        )

    def _apply_label_encoder(self, df: pl.DataFrame) -> pl.DataFrame:
        term_indices = pl.Series(
            name="term_idx",
            values=self.term_encoder.transform(df["term"].to_numpy())
        )
        # Return new dataframe with exists and term_idx columns
        return df.with_columns(term_indices)

    def _convert_to_sparse_matrix(self, df: pl.DataFrame) -> csr_matrix:
        row_indices = np.zeros(len(df), dtype=int)  # all zero, since it's row 0
        col_indices = df["term_idx"].to_numpy()
        data = df["exists"].to_numpy()

        return csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(1, len(self.term_encoder.classes_))
        )
    def format_bag_of_words(
        self,
        bag_of_words: dict[str, int]
    ) -> csr_matrix:

        df = self._convert_to_polars(bag_of_words)
        df = self._apply_label_encoder(df)
        return self._convert_to_sparse_matrix(df)