import numpy as np
import pandas as pd
import pytest
from numpy import linalg
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pypfopt.risk_models import sample_cov

from allooptim.covariance_transformer.covariance_transformer import (
    DeNoiserCovarianceTransformer,
    DetoneCovarianceTransformer,
    cov_to_corr,
)
from allooptim.covariance_transformer.transformer_interface import AbstractCovarianceTransformer
from allooptim.covariance_transformer.transformer_list import TRANSFORMER_LIST


@pytest.fixture
def cov_matrix():
    asset_names = ["Asset_0", "Asset_1", "Asset_2"]
    stdevs = np.array([0.8, 1.2, 0.5])
    stdev_matrix = np.diag(stdevs)
    correlations = np.array(
        [
            [1.0, -0.5, 0.0],
            [-0.5, 1.0, 0.2],
            [0.0, 0.2, 1.0],
        ]
    )

    cov_data = np.dot(np.dot(stdev_matrix, correlations), np.transpose(stdev_matrix))
    return pd.DataFrame(cov_data, index=asset_names, columns=asset_names)


@pytest.fixture
def detoned_results():
    asset_names = ["Asset_0", "Asset_1", "Asset_2"]
    detoned_data = np.array(
        [
            [0.64, 0.737716, 0.193364],
            [0.737716, 1.44, -0.113214],
            [0.193364, -0.113214, 0.25],
        ]
    )
    return pd.DataFrame(detoned_data, index=asset_names, columns=asset_names)


class TestDetoneCovarianceTransformer:
    def test_transform(self, cov_matrix, detoned_results):
        results = DetoneCovarianceTransformer(n_remove=1).transform(cov_matrix, None)

        # Results should be a pandas DataFrame
        assert isinstance(results, pd.DataFrame)
        assert list(results.index) == list(cov_matrix.index)
        assert list(results.columns) == list(cov_matrix.columns)

        # Compare values
        assert_array_almost_equal(results.values, detoned_results.values)

        # Test eigenvalues
        w, v = linalg.eig(results.values)
        sort_index = np.argsort(-np.abs(w))
        assert_almost_equal(w[sort_index][-1], 0.0)

        # Test n_remove=0 returns original matrix
        results_no_remove = DetoneCovarianceTransformer(n_remove=0).transform(cov_matrix, None)
        assert isinstance(results_no_remove, pd.DataFrame)
        assert_array_almost_equal(results_no_remove.values, cov_matrix.values)


class TestDeNoiserCovarianceTransformer:
    def test_transform(self, prices_df):
        covariance_matrix = sample_cov(prices_df)  # This returns a pandas DataFrame
        n_observations = prices_df.size
        results = DeNoiserCovarianceTransformer().transform(covariance_matrix, n_observations)

        # Results should be a pandas DataFrame
        assert isinstance(results, pd.DataFrame)
        assert results.shape == (4, 4)
        assert list(results.index) == list(covariance_matrix.index)
        assert list(results.columns) == list(covariance_matrix.columns)

        expected_values = np.array(
            [
                [0.0971403, -0.0063762, 0.0060754, -0.0031562],
                [-0.0063762, 0.0988573, -0.0047028, 0.0024431],
                [0.0060754, -0.0047028, 0.1128172, -0.0023279],
                [-0.0031562, 0.0024431, -0.0023279, 0.0888692],
            ]
        )
        assert_almost_equal(results.values, expected_values)


def test_cov_to_corr(prices_df):
    covariance_matrix = sample_cov(prices_df).values
    correlation_matrix = cov_to_corr(covariance_matrix)
    assert correlation_matrix.shape == covariance_matrix.shape
    # values should be between [-1,1]
    assert correlation_matrix.max() <= 1
    assert correlation_matrix.min() >= -1
    for i in range(len(correlation_matrix)):
        # the diagonal should all be 1 since an asset is perfectly correlated with itself
        assert_almost_equal(correlation_matrix[i, i], 1.0)


@pytest.mark.parametrize("transformer_class", TRANSFORMER_LIST)
def test_transformers(transformer_class):
    assert issubclass(transformer_class, AbstractCovarianceTransformer)

    n_assets = 3
    n_periods = 30

    df_prices = pd.DataFrame(
        np.random.rand(n_periods, n_assets),
        columns=[f"A{i}" for i in range(n_assets)],
    )

    df_cov = df_prices.cov()

    # Test the transform method
    transformer = transformer_class()

    if transformer.name == "AutoencoderCovarianceTransformer":
        pytest.skip("Skipping AutoencoderCovarianceTransformer test here due to long training")

    # Check that fit and transform don't raise warnings
    import warnings
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        transformer.fit(
            df_prices,
        )

        transformed_cov = transformer.transform(
            df_cov=df_cov,
            n_observations=n_periods,
        )

    # Fail the test if any warnings were raised
    if warning_list:
        warning_messages = [str(w.message) for w in warning_list]
        pytest.fail(
            f"Transformer {transformer.name} raised warnings during fit/transform: {warning_messages}"
        )

    assert isinstance(transformed_cov, pd.DataFrame)
    assert transformed_cov.shape == (n_assets, n_assets)
    assert not transformed_cov.isna().any().any()
    assert isinstance(transformer.name, str)
