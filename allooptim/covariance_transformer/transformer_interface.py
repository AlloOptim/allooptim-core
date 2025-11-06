from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class AbstractCovarianceTransformer(ABC):
    """Abstract base class for all covariance matrix transformations.

    Covariance transformers improve covariance matrix estimates through statistical techniques
    like shrinkage, denoising, or regularization. This ensures better-conditioned matrices
    for portfolio optimization, especially with limited historical data.

    The transformer interface maintains asset name consistency and provides a standardized
    approach to covariance preprocessing across different optimization strategies.

    Subclassing Guide:
        1. Inherit from AbstractCovarianceTransformer
        2. Implement transform() method
        3. Optionally implement fit() for data-dependent transformations
        4. Ensure asset names are preserved in output DataFrame

    Examples:
        Basic transformer usage:

        >>> transformer = SimpleShrinkageCovarianceTransformer(shrinkage=0.3)
        >>> clean_cov = transformer.transform(sample_cov, n_observations=60)

        Fitting transformer to data:

        >>> transformer.fit(historical_prices)
        >>> clean_cov = transformer.transform(sample_cov)

    See Also:
        - :class:`SimpleShrinkageCovarianceTransformer`: Basic shrinkage estimation
        - :class:`LedoitWolfCovarianceTransformer`: Optimal shrinkage
        - :mod:`allooptim.covariance_transformer`: Available transformers
    """

    def fit(self, df_prices: pd.DataFrame) -> None:
        """Optional method to fit the transformer to the data
        :param df_prices: DataFrame of historical asset prices
        """
        pass

    @abstractmethod
    def transform(
        self,
        df_cov: pd.DataFrame,
        n_observations: Optional[int] = None,
    ) -> pd.DataFrame:
        """Transforms a covariance matrix
        :param df_cov: covariance matrix
        :param n_observations: number of observations used to create the covariance matrix
        :return: transformed covariance matrix as pandas DataFrame with preserved asset names
        """
        pass

    @property
    def name(self) -> str:
        """Name of this optimizer. The name will be displayed in the MCOS results DataFrame."""
        return self.__class__.__name__
