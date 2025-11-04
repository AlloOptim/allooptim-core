from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class AbstractCovarianceTransformer(ABC):
    """
    Abstract base class for covariance matrix transformations with asset name preservation.

    Covariance transformers apply various statistical techniques to improve covariance matrix
    estimates, such as shrinkage, noise filtering, or dimensionality reduction. All transformers
    maintain asset name consistency and return properly formatted pandas DataFrames.

    Examples:
        Basic usage with asset name preservation:

        >>> transformer = SimpleShrinkageCovarianceTransformer(shrinkage=0.2)
        >>> cov_clean = transformer.transform(cov_matrix, n_observations=252)
        >>> print(cov_clean.index.tolist())  # Asset names preserved
        ['AAPL', 'GOOGL', 'MSFT', ...]
    """

    def fit(self, df_prices: pd.DataFrame) -> None:
        """
        Optional method to fit the transformer to the data
        :param df_prices: DataFrame of historical asset prices
        """
        pass

    @abstractmethod
    def transform(
        self,
        df_cov: pd.DataFrame,
        n_observations: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Transforms a covariance matrix
        :param df_cov: covariance matrix
        :param n_observations: number of observations used to create the covariance matrix
        :return: transformed covariance matrix as pandas DataFrame with preserved asset names
        """

        pass

    @property
    def name(self) -> str:
        """
        Name of this optimizer. The name will be displayed in the MCOS results DataFrame.
        """

        return self.__class__.__name__
