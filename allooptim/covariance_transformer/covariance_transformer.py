from typing import Optional, Union

import numpy as np
import pandas as pd
from numpy import linalg
from scipy.optimize import minimize
from sklearn.covariance import OAS, EllipticEnvelope, EmpiricalCovariance, LedoitWolf, ShrunkCovariance
from sklearn.neighbors import KernelDensity

from allooptim.covariance_transformer.transformer_interface import (
    AbstractCovarianceTransformer,
)

# Constants for numerical thresholds
NEAR_SINGULARITY_CONDITION_THRESHOLD = 1e12


def _extract_cov_info(cov: Union[np.ndarray, pd.DataFrame]) -> tuple[np.ndarray, list]:
    """Extract numpy array and asset names from covariance matrix
    :param cov: covariance matrix (numpy array or pandas DataFrame)
    :return: tuple of (numpy array, asset names list).
    """
    if isinstance(cov, pd.DataFrame):
        asset_names = cov.index.tolist()
        cov_array = cov.values.astype(float)
    else:
        cov_array = np.array(cov, dtype=float)
        asset_names = [f"Asset_{i}" for i in range(cov_array.shape[0])]

    return cov_array, asset_names


def _create_cov_dataframe(cov_array: np.ndarray, asset_names: list) -> pd.DataFrame:
    """Create pandas DataFrame from numpy covariance matrix with asset names
    :param cov_array: covariance matrix as numpy array
    :param asset_names: list of asset names
    :return: pandas DataFrame with asset names as index/columns.
    """
    return pd.DataFrame(cov_array, index=asset_names, columns=asset_names)


def _ensure_symmetric(matrix: np.array) -> np.array:
    """Ensure a matrix is symmetric by averaging with its transpose."""
    return (matrix + matrix.T) / 2


def cov_to_corr(cov: np.array) -> np.array:
    """Derive the correlation matrix from a covariance matrix
    :param cov: covariance matrix
    :return: correlation matrix.
    """
    # Ensure input is numpy array (not DataFrame)
    cov = np.asarray(cov)

    # Ensure the covariance matrix is symmetric
    cov = _ensure_symmetric(cov)

    std = np.sqrt(np.diag(cov))

    # Handle zero standard deviations
    std = np.where(std == 0, 1e-8, std)  # Replace zeros with small positive values

    corr = cov / np.outer(std, std)
    corr[corr < -1], corr[corr > 1] = -1, 1  # numerical error

    # Ensure diagonal is exactly 1
    np.fill_diagonal(corr, 1.0)

    # Ensure the correlation matrix is symmetric
    corr = _ensure_symmetric(corr)

    return corr


def _corr_to_cov(corr: np.array, std: np.array) -> np.array:
    """Recovers the covariance matrix from the de-noise correlation matrix
    :param corr: de-noised correlation matrix
    :param std: standard deviation of the correlation matrix
    :return: a recovered covariance matrix.
    """
    cov = corr * np.outer(std, std)
    return cov


def _reorder_matrix(m: np.array, sort_index: np.array) -> np.array:
    m = m[sort_index, :]
    m = m[:, sort_index]
    return m


class SimpleShrinkageCovarianceTransformer(AbstractCovarianceTransformer):
    """Simple shrinkage covariance transformer that shrinks towards identity matrix.

    This transformer applies a simple linear shrinkage of the covariance matrix towards
    the identity matrix. The shrinkage intensity is controlled by the shrinkage parameter.

    Attributes:
        shrinkage (float): Shrinkage intensity between 0 and 1. Higher values result in
            more shrinkage towards the identity matrix.
    """

    def __init__(self, shrinkage: float = 0.2):
        """Initialize the simple shrinkage covariance transformer.

        Args:
            shrinkage: Shrinkage intensity between 0 and 1. Default is 0.2.
        """
        self.shrinkage = shrinkage

    def transform(self, df_cov: pd.DataFrame, n_observations: Optional[int] = None) -> pd.DataFrame:
        """Apply simple shrinkage transformation to the covariance matrix.

        Shrinks the input covariance matrix towards the identity matrix using linear
        interpolation. The formula is: (1-shrinkage) * cov + shrinkage * I

        Args:
            df_cov: Input covariance matrix as pandas DataFrame.
            n_observations: Number of observations used to estimate the covariance
                (unused in this transformer).

        Returns:
            Transformed covariance matrix as pandas DataFrame with preserved asset names.
        """
        # Extract numpy array and asset names
        cov_array, asset_names = _extract_cov_info(df_cov)

        # Apply shrinkage transformation
        transformed_cov = cov_array * (1 - self.shrinkage) + np.eye(len(cov_array)) * self.shrinkage
        transformed_cov = _ensure_symmetric(transformed_cov)

        # Return as pandas DataFrame with preserved asset names
        return _create_cov_dataframe(transformed_cov, asset_names)


class SklearnShrinkageCovarianceTransformer(AbstractCovarianceTransformer):
    """Scikit-learn shrinkage covariance transformer using Ledoit-Wolf shrinkage.

    This transformer uses scikit-learn's ShrunkCovariance estimator which applies
    the Ledoit-Wolf shrinkage method to improve covariance matrix estimation.
    """

    def transform(self, df_cov: pd.DataFrame, n_observations: Optional[int] = None) -> pd.DataFrame:
        """Apply scikit-learn shrinkage transformation to the covariance matrix.

        Uses sklearn.covariance.ShrunkCovariance to estimate the covariance matrix
        with automatic shrinkage intensity selection based on the Ledoit-Wolf method.

        Args:
            df_cov: Input covariance matrix as pandas DataFrame.
            n_observations: Number of observations used to estimate the covariance
                (unused in this transformer).

        Returns:
            Transformed covariance matrix as pandas DataFrame with preserved asset names.
        """
        # Extract numpy array and asset names
        cov_array, asset_names = _extract_cov_info(df_cov)

        # Apply sklearn shrinkage transformation
        transformed_cov = ShrunkCovariance().fit(cov_array).covariance_
        transformed_cov = _ensure_symmetric(transformed_cov)

        # Return as pandas DataFrame with preserved asset names
        return _create_cov_dataframe(transformed_cov, asset_names)


class EllipticEnvelopeShrinkageCovarianceTransformer(AbstractCovarianceTransformer):
    """Elliptic envelope covariance transformer for robust covariance estimation.

    This transformer uses scikit-learn's EllipticEnvelope estimator which fits
    a robust covariance estimate using the Minimum Covariance Determinant (MCD) method.
    This is useful for handling outliers in the data.
    """

    def transform(self, df_cov: Union[np.ndarray, pd.DataFrame], n_observations: Optional[int] = None) -> pd.DataFrame:
        """Apply elliptic envelope transformation to the covariance matrix.

        Uses sklearn.covariance.EllipticEnvelope to estimate a robust covariance matrix
        that is less sensitive to outliers in the data.

        Args:
            df_cov: Input covariance matrix as pandas DataFrame or numpy array.
            n_observations: Number of observations used to estimate the covariance
                (unused in this transformer).

        Returns:
            Transformed covariance matrix as pandas DataFrame with preserved asset names.
        """
        # Extract numpy array and asset names
        cov_array, asset_names = _extract_cov_info(df_cov)

        # Apply elliptic envelope transformation
        transformed_cov = EllipticEnvelope().fit(cov_array).covariance_
        transformed_cov = _ensure_symmetric(transformed_cov)

        # Return as pandas DataFrame with preserved asset names
        return _create_cov_dataframe(transformed_cov, asset_names)


class EmpiricalCovarianceTransformer(AbstractCovarianceTransformer):
    """Empirical covariance transformer with regularization for high-dimensional data.

    This transformer computes empirical covariance estimates with automatic regularization
    for cases where the number of observations is less than the number of assets (n << p).
    It applies various regularization techniques to ensure numerical stability.
    """

    def __init__(
        self,
        regularization_method: str = "diagonal_loading",
        reg_param: float = 1e-4,
        fallback_shrinkage: float = 0.1,
    ) -> None:
        """Initialize EmpiricalCovarianceTransformer with regularization for n>>p scenarios.

        Args:
            regularization_method: Method to handle singularity ('diagonal_loading', 'shrinkage', 'eigenvalue_clip')
            reg_param: Regularization parameter
            fallback_shrinkage: Shrinkage to use if matrix is singular and regularization fails
        """
        self.regularization_method = regularization_method
        self.reg_param = reg_param
        self.fallback_shrinkage = fallback_shrinkage

    def transform(self, df_cov: pd.DataFrame, n_observations: Optional[int] = None) -> pd.DataFrame:
        """Transform covariance matrix using empirical estimation with regularization for singular cases.

        For n>>p regimes (like 500x60), the empirical covariance will be singular. This method
        applies regularization to make it invertible and numerically stable.
        """
        # Extract numpy array and asset names
        cov_array, asset_names = _extract_cov_info(df_cov)
        n_assets = cov_array.shape[0]

        # Check if we're in a problematic regime
        if n_observations < n_assets:
            print(
                f"Warning: n_observations ({n_observations}) < n_assets ({n_assets}). "
                f"Empirical covariance will be singular. Applying {self.regularization_method} regularization."
            )

        try:
            # Try empirical covariance first
            empirical_cov = EmpiricalCovariance().fit(cov_array).covariance_

            # Check condition number to detect near-singularity
            cond_num = np.linalg.cond(empirical_cov)

            # Apply regularization if needed
            if (
                cond_num > NEAR_SINGULARITY_CONDITION_THRESHOLD or n_observations < n_assets
            ):  # Singular or near-singular
                if self.regularization_method == "diagonal_loading":
                    # Add small value to diagonal
                    regularized_cov = empirical_cov + np.eye(n_assets) * self.reg_param

                elif self.regularization_method == "shrinkage":
                    # Apply shrinkage towards identity
                    regularized_cov = (1 - self.reg_param) * empirical_cov + self.reg_param * np.eye(n_assets)

                elif self.regularization_method == "eigenvalue_clip":
                    # Clip eigenvalues to minimum threshold
                    eigvals, eigvecs = np.linalg.eigh(empirical_cov)
                    eigvals = np.maximum(eigvals, self.reg_param)
                    regularized_cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

                else:
                    raise ValueError(f"Unknown regularization method: {self.regularization_method}")

                transformed_cov = _ensure_symmetric(regularized_cov)
            else:
                transformed_cov = _ensure_symmetric(empirical_cov)

        except Exception as e:
            print(f"EmpiricalCovariance failed with error: {e}")
            print(f"Falling back to shrinkage regularization with shrinkage={self.fallback_shrinkage}")

            # Fallback to simple shrinkage
            fallback_cov = (1 - self.fallback_shrinkage) * cov_array + self.fallback_shrinkage * np.eye(n_assets)
            transformed_cov = _ensure_symmetric(fallback_cov)

        # Return as pandas DataFrame with preserved asset names
        return _create_cov_dataframe(transformed_cov, asset_names)


class OracleCovarianceTransformer(AbstractCovarianceTransformer):
    """Oracle Approximating Shrinkage (OAS) covariance transformer.

    This transformer uses the Oracle Approximating Shrinkage method which provides
    an optimal shrinkage intensity estimate when the true covariance matrix is known
    (oracle case). In practice, it provides a good approximation without knowing the truth.
    """

    def transform(self, df_cov: pd.DataFrame, n_observations: Optional[int] = None) -> pd.DataFrame:
        """Apply Oracle Approximating Shrinkage transformation to the covariance matrix.

        Uses sklearn.covariance.OAS to estimate the covariance matrix with optimal
        shrinkage intensity based on the Oracle Approximating Shrinkage method.

        Args:
            df_cov: Input covariance matrix as pandas DataFrame.
            n_observations: Number of observations used to estimate the covariance
                (unused in this transformer).

        Returns:
            Transformed covariance matrix as pandas DataFrame with preserved asset names.
        """
        # Extract numpy array and asset names
        cov_array, asset_names = _extract_cov_info(df_cov)

        # Apply OAS transformation
        transformed_cov = OAS().fit(cov_array).covariance_
        transformed_cov = _ensure_symmetric(transformed_cov)

        # Return as pandas DataFrame with preserved asset names
        return _create_cov_dataframe(transformed_cov, asset_names)


class LedoitWolfCovarianceTransformer(AbstractCovarianceTransformer):
    """Ledoit-Wolf shrinkage covariance transformer.

    This transformer uses the classical Ledoit-Wolf shrinkage method which provides
    a well-performing shrinkage intensity estimate for covariance matrix regularization.
    It shrinks towards the identity matrix with an analytically derived shrinkage intensity.
    """

    def transform(self, df_cov: pd.DataFrame, n_observations: Optional[int] = None) -> pd.DataFrame:
        """Apply Ledoit-Wolf shrinkage transformation to the covariance matrix.

        Uses sklearn.covariance.LedoitWolf to estimate the covariance matrix with
        analytically optimal shrinkage intensity towards the identity matrix.

        Args:
            df_cov: Input covariance matrix as pandas DataFrame.
            n_observations: Number of observations used to estimate the covariance
                (unused in this transformer).

        Returns:
            Transformed covariance matrix as pandas DataFrame with preserved asset names.
        """
        # Extract numpy array and asset names
        cov_array, asset_names = _extract_cov_info(df_cov)

        # Apply Ledoit-Wolf transformation
        transformed_cov = LedoitWolf().fit(cov_array).covariance_
        transformed_cov = _ensure_symmetric(transformed_cov)

        # Return as pandas DataFrame with preserved asset names
        return _create_cov_dataframe(transformed_cov, asset_names)


class MarcenkoPasturCovarianceTransformer(AbstractCovarianceTransformer):
    """Marchenko-Pastur covariance transformer for random matrix theory denoising.

    This transformer applies Random Matrix Theory to identify and filter out noise
    eigenvalues that fall within the Marchenko-Pastur distribution bounds. Signal
    eigenvalues outside these bounds are preserved while noise eigenvalues are
    replaced with their average value.
    """

    def __init__(self, variance_scaling: float = 1.0):
        """Initialize Marchenko-Pastur covariance transformer.

        Args:
            variance_scaling: Scaling factor for MP distribution (default assumes σ² = 1)
        """
        self.variance_scaling = variance_scaling

    def transform(self, df_cov: pd.DataFrame, n_observations: Optional[int] = None) -> pd.DataFrame:
        """Apply Marchenko-Pastur eigenvalue filtering to remove noise.

        Based on Random Matrix Theory, this method identifies and filters out eigenvalues
        that fall within the Marchenko-Pastur distribution bounds, which correspond to noise.

        CORRECTED: The MP distribution works for ANY λ = m/n > 0, including λ > 1.
        For 500 assets and 60 observations, λ = 500/60 = 8.33 is perfectly valid.
        """
        # Extract numpy array and asset names
        cov_array, asset_names = _extract_cov_info(df_cov)
        n_assets = cov_array.shape[0]

        # Use the actual number of observations, not the matrix dimension
        n_samples = n_observations

        # Calculate λ = m/n (ratio of assets to observations)
        # This is the key parameter for MP distribution
        lam = n_assets / n_samples  # λ in MP theory

        print(f"MP Transform: n_assets={n_assets}, n_observations={n_samples}, λ={lam:.3f}")

        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(cov_array)

        # Sort eigenvalues in ascending order (as they come from np.linalg.eigh)
        sort_idx = np.argsort(eigvals)
        eigvals = eigvals[sort_idx]
        eigvecs = eigvecs[:, sort_idx]

        # Marchenko-Pastur bounds with variance scaling
        # For correlation matrices, σ² = variance_scaling (typically 1)
        sigma_sq = self.variance_scaling

        # MP distribution bounds: λ± = σ²(1 ± √λ)²
        lambda_min = sigma_sq * (1 - np.sqrt(lam)) ** 2
        lambda_max = sigma_sq * (1 + np.sqrt(lam)) ** 2

        print(f"MP bounds: λ_min={lambda_min:.4f}, λ_max={lambda_max:.4f}")
        print(f"Eigenvalue range: {eigvals.min():.4f} to {eigvals.max():.4f}")

        # Identify noise eigenvalues (those within MP bounds)
        # Signal eigenvalues are those OUTSIDE the MP bounds (typically above λ_max)
        noise_mask = (eigvals >= lambda_min) & (eigvals <= lambda_max)
        signal_mask = ~noise_mask

        noise_idxs = np.where(noise_mask)[0]
        signal_idxs = np.where(signal_mask)[0]

        print(f"Found {len(noise_idxs)} noise eigenvalues, {len(signal_idxs)} signal eigenvalues")

        # Create denoised eigenvalues
        denoised_eigvals = eigvals.copy()

        if len(noise_idxs) > 0:
            # Replace noise eigenvalues with their mean
            avg_noise_eig = np.mean(eigvals[noise_idxs])
            denoised_eigvals[noise_idxs] = avg_noise_eig
            print(f"Replaced {len(noise_idxs)} noise eigenvalues with mean: {avg_noise_eig:.6f}")
        else:
            print("No eigenvalues identified as noise within MP bounds")

        # Ensure all eigenvalues are positive
        denoised_eigvals = np.maximum(denoised_eigvals, 1e-8)

        # Reconstruct covariance matrix
        cov_denoised = eigvecs @ np.diag(denoised_eigvals) @ eigvecs.T

        # Ensure symmetry
        transformed_cov = _ensure_symmetric(cov_denoised)

        # Return as pandas DataFrame with preserved asset names
        return _create_cov_dataframe(transformed_cov, asset_names)


class PCACovarianceTransformer(AbstractCovarianceTransformer):
    """PCA-based covariance transformer for dimensionality reduction.

    This transformer applies Principal Component Analysis to denoise the covariance
    matrix by retaining only the top principal components that explain a specified
    fraction of the total variance. This reduces noise while preserving the most
    important signal components.
    """

    def __init__(self, n_components: int = None, variance_threshold: float = 0.95):
        """Initialize PCA covariance transformer.

        Args:
            n_components: Number of components to retain. If None, uses variance_threshold
            variance_threshold: Fraction of variance to retain when n_components is None
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold

    def transform(self, df_cov: pd.DataFrame, n_observations: Optional[int] = None) -> pd.DataFrame:
        """Applies PCA to denoise the covariance matrix by retaining only the top n_components
        principal components.

        :param cov: the covariance matrix we want to denoise
        :param n_observations: the number of observations used to create the covariance matrix
        :return: denoised covariance matrix
        """
        # Extract numpy array and asset names
        cov_array, asset_names = _extract_cov_info(df_cov)
        n_assets = cov_array.shape[0]
        eigvals, eigvecs = np.linalg.eigh(cov_array)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sorted_idx]
        eigvecs = eigvecs[:, sorted_idx]

        # Determine number of components to retain
        n_components = self.n_components
        if n_components is None:
            # Automatic estimation: retain components explaining variance_threshold of total variance
            total_var = np.sum(eigvals[eigvals > 0])  # Only positive eigenvalues
            running_var = 0
            n_components = 0
            for val in eigvals:
                if val > 0:  # Only count positive eigenvalues
                    running_var += val
                    n_components += 1
                    if running_var / total_var >= self.variance_threshold:
                        break

        # Ensure n_components is within valid bounds
        max_components = min(n_assets, n_observations, np.sum(eigvals > 0))
        n_components = min(n_components, max_components)
        n_components = max(n_components, 1)  # At least 1 component

        # Keep only top components
        signal_eigvals = eigvals[:n_components]
        signal_eigvecs = eigvecs[:, :n_components]

        # Ensure all eigenvalues are positive
        signal_eigvals = np.maximum(signal_eigvals, 1e-8)

        # Reconstruct covariance matrix
        denoised_cov = signal_eigvecs @ np.diag(signal_eigvals) @ signal_eigvecs.T

        # Ensure symmetric
        transformed_cov = _ensure_symmetric(denoised_cov)

        # Return as pandas DataFrame with preserved asset names
        return _create_cov_dataframe(transformed_cov, asset_names)


class DeNoiserCovarianceTransformer(AbstractCovarianceTransformer):
    """Kernel density estimation based covariance denoiser.

    This transformer uses Kernel Density Estimation to fit the Marchenko-Pastur
    distribution to the empirical eigenvalue distribution, then shrinks eigenvalues
    associated with noise while preserving signal eigenvalues. Based on the method
    described in "A Robust Estimator of the Efficient Frontier".
    """

    def __init__(
        self,
        bandwidth: float = 0.25,
        n_observations: int = 1,
    ) -> None:
        """Initialize DeNoiser covariance transformer.

        Args:
            bandwidth: Bandwidth hyper-parameter for KernelDensity
            n_observations: Number of observations used to estimate covariance
        """
        self.bandwidth = bandwidth
        self.n_observations = n_observations

    def transform(self, df_cov: pd.DataFrame, n_observations: Optional[int] = None) -> pd.DataFrame:
        """Computes the correlation matrix associated with a given covariance matrix,
        and derives the eigenvalues and eigenvectors for that correlation matrix.
        Then shrinks the eigenvalues associated with noise, resulting in a de-noised correlation matrix
        which is then used to recover the covariance matrix.

        In summary, this step shrinks only the eigenvalues
        associated with noise, leaving the eigenvalues associated with signal unchanged.

        For more info see section 4.2 of "A Robust Estimator of the Efficient Frontier",
        this function and the functions it calls are all modified from this section

        :param cov: the covariance matrix we want to de-noise
        :param n_observations: the number of observations used to create the covariance matrix
        :return: de-noised covariance matrix
        """
        # Extract numpy array and asset names
        cov_array, asset_names = _extract_cov_info(df_cov)

        self.n_observations = n_observations

        #  q=T/N where T=sample length and N=number of variables
        q = self.n_observations / cov_array.shape[1]

        # get correlation matrix based on covariance matrix
        correlation_matrix = cov_to_corr(cov_array)

        # Get eigenvalues and eigenvectors in the correlation matrix
        eigenvalues, eigenvectors = self._get_PCA(correlation_matrix)

        # Find max random eigenvalue
        max_eigenvalue = self._find_max_eigenvalue(np.diag(eigenvalues), q)

        # de-noise the correlation matrix
        n_facts = eigenvalues.shape[0] - np.diag(eigenvalues)[::-1].searchsorted(max_eigenvalue)
        correlation_matrix = self._de_noised_corr(eigenvalues, eigenvectors, n_facts)

        # recover covariance matrix from correlation matrix
        de_noised_covariance_matrix = _corr_to_cov(correlation_matrix, np.diag(cov_array) ** 0.5)

        # make symmetric
        transformed_cov = _ensure_symmetric(de_noised_covariance_matrix)

        # Return as pandas DataFrame with preserved asset names
        return _create_cov_dataframe(transformed_cov, asset_names)

    def _get_PCA(self, matrix: np.array) -> tuple[np.array, np.array]:
        """Gets eigenvalues and eigenvectors from a Hermitian matrix
        :param matrix: a Hermitian matrix
        :return: array of eigenvalues and array of eigenvectors.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        indices = eigenvalues.argsort()[::-1]  # arguments for sorting eigenvalues desc
        eigenvalues, eigenvectors = eigenvalues[indices], eigenvectors[:, indices]
        eigenvalues = np.diagflat(eigenvalues)
        return eigenvalues, eigenvectors

    def _find_max_eigenvalue(self, eigenvalues: np.array, q: float) -> float:
        """Uses a Kernel Density Estimate (KDE) algorithm to fit the
        Marcenko-Pastur distribution to the empirical distribution of eigenvalues.
        This has the effect of separating noise-related eigenvalues from signal-related eigenvalues.
        :param eigenvalues: array of eigenvalues
        :param q: q=T/N where T=sample length and N=number of variables
        :return: max random eigenvalue, variance.
        """
        # Find max random eigenvalues by fitting Marcenko's dist to the empirical one
        out = minimize(
            lambda *x: self._err_PDFs(*x),
            0.5,
            args=(eigenvalues, q),
            bounds=((1e-5, 1 - 1e-5),),
        )
        var = out["x"][0] if out["success"] else 1
        max_eigenvalue = var * (1 + (1.0 / q) ** 0.5) ** 2
        return max_eigenvalue

    def _err_PDFs(self, var: float, eigenvalues: pd.Series, q: float, pts: int = 1000) -> float:
        """Calculates a theoretical Marcenko-Pastur probability density function and
        an empirical Marcenko-Pastur probability density function,
        and finds the error between the two by squaring the difference of the two
        :param var: variance 𝜎^2
        :param eigenvalues: array of eigenvalues
        :param q: q=T/N where T=sample length and N=number of variables
        :param pts: number of points in the distribution
        :return: the error of the probability distribution functions obtained by squaring the difference
        of the theoretical and empirical Marcenko-Pastur probability density functions.
        """
        # Fit error
        theoretical_pdf = self._mp_PDF(var, q, pts)  # theoretical probability density function
        empirical_pdf = self._fit_KDE(
            eigenvalues, x=theoretical_pdf.index.values
        )  # empirical probability density function
        sse = np.sum((empirical_pdf - theoretical_pdf) ** 2)
        return sse

    def _mp_PDF(self, var: float, q: float, pts: int) -> pd.Series:
        """Creates a theoretical Marcenko-Pastur probability density function
        :param var: variance 𝜎^2
        :param q: q=T/N where T=sample length and N=number of variables
        :param pts: number of points in the distribution
        :return: a theoretical Marcenko-Pastur probability density function.
        """
        min_eigenvalue, max_eigenvalue = (
            var * (1 - (1.0 / q) ** 0.5) ** 2,
            var * (1 + (1.0 / q) ** 0.5) ** 2,
        )
        eigenvalues = np.linspace(min_eigenvalue, max_eigenvalue, pts).flatten()
        pdf = (
            q
            / (2 * np.pi * var * eigenvalues)
            * ((max_eigenvalue - eigenvalues) * (eigenvalues - min_eigenvalue)) ** 0.5
        )
        pdf = pdf.flatten()
        pdf = pd.Series(pdf, index=eigenvalues)
        return pdf

    def _fit_KDE(self, obs: np.array, kernel: str = "gaussian", x: np.array = None) -> pd.Series:
        """Fit kernel to a series of observations, and derive the prob of observations.
        x is the array of values on which the fit KDE will be evaluated
        :param obs: the series of observations
        :param kernel: kernel hyper-parameter for KernelDensity
        :param x: array of values _fit_KDE will be evaluated against
        :return: an empirical Marcenko-Pastur probability density function.
        """
        if len(obs.shape) == 1:
            obs = obs.reshape(-1, 1)
        kde = KernelDensity(kernel=kernel, bandwidth=self.bandwidth).fit(obs)
        if x is None:
            x = np.unique(obs).reshape(-1, 1)
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        log_prob = kde.score_samples(x)  # log(density)
        pdf = pd.Series(np.exp(log_prob), index=x.flatten())
        return pdf

    def _de_noised_corr(self, eigenvalues: np.array, eigenvectors: np.array, n_facts: int) -> np.array:
        """Shrinks the eigenvalues associated with noise, and returns a de-noised correlation matrix
        :param eigenvalues: array of eigenvalues
        :param eigenvectors: array of eigenvectors
        :param n_facts: number of elements in diagonalized eigenvalues to replace with the mean of eigenvalues
        :return: de-noised correlation matrix.
        """
        # Remove noise from corr by fixing random eigenvalues
        eigenvalues_ = np.diag(eigenvalues).copy()
        eigenvalues_[n_facts:] = eigenvalues_[n_facts:].sum() / float(eigenvalues_.shape[0] - n_facts)
        eigenvalues_ = np.diag(eigenvalues_)
        corr = np.dot(eigenvectors, eigenvalues_).dot(eigenvectors.T)
        corr = cov_to_corr(corr)
        return corr


class DetoneCovarianceTransformer(AbstractCovarianceTransformer):
    """Detoning covariance transformer for market factor removal.

    This transformer removes the largest eigenvalue/eigenvector pairs from the
    covariance matrix, which are typically associated with market-wide factors.
    This has the effect of removing the market's influence on correlations between
    securities, focusing on idiosyncratic risk. Based on methods from "Machine
    Learning for Asset Managers".
    """

    def __init__(self, remove_fraction: float = None, n_remove: int = None) -> None:
        """Initialize Detone covariance transformer.

        Args:
            remove_fraction: Fraction of eigenvalues to remove (0 < remove_fraction < 1)
            n_remove: Number of eigenvalues to remove (backward compatibility)
        """
        if n_remove is not None and remove_fraction is not None:
            raise ValueError("Cannot specify both n_remove and remove_fraction")
        elif n_remove is not None:
            # Backward compatibility: convert n_remove to remove_fraction
            # For a 3x3 matrix, n_remove=1 means remove_fraction=1/3
            self.n_remove = n_remove
            self.remove_fraction = None  # Will be computed dynamically
        elif remove_fraction is not None:
            assert 0 < remove_fraction < 1, "remove_fraction must be between 0 and 1"
            self.remove_fraction = remove_fraction
            self.n_remove = None
        else:
            # Default behavior
            self.remove_fraction = 0.1
            self.n_remove = None

    def transform(self, df_cov: pd.DataFrame, n_observations: Optional[int] = None) -> pd.DataFrame:
        """Apply detoning transformation to remove market factors from covariance matrix.

        Removes the largest eigenvalue/eigenvector pairs that correspond to market-wide
        factors, focusing the covariance matrix on idiosyncratic risk components.

        Args:
            df_cov: Input covariance matrix as pandas DataFrame.
            n_observations: Number of observations used to estimate the covariance
                (unused in this transformer).

        Returns:
            Detoned covariance matrix as pandas DataFrame with preserved asset names.
        """
        # Extract numpy array and asset names
        cov_array, asset_names = _extract_cov_info(df_cov)

        corr = cov_to_corr(cov_array)

        w, v = linalg.eig(corr)

        # sort from highest eigenvalues to lowest
        sort_index = np.argsort(-np.abs(w))  # get sort_index in descending absolute order - i.e. from most significant
        w = w[sort_index]
        v = v[:, sort_index]

        # remove largest eigenvalue component
        n_remove = self.n_remove if self.n_remove is not None else max(int(self.remove_fraction * len(w)), 1)

        # Handle special case: n_remove=0 means return original matrix
        if n_remove == 0:
            return _create_cov_dataframe(cov_array, asset_names)

        v_market = v[:, 0:n_remove]  # largest eigenvectors
        w_market = w[0:n_remove]

        market_comp = np.matmul(
            np.matmul(v_market, w_market).reshape(
                (
                    v.shape[0],
                    n_remove,
                )
            ),
            np.transpose(v_market),
        )

        c2 = corr - market_comp

        # normalize the correlation matrix so the diagonals are 1
        norm_matrix = np.diag(c2.diagonal() ** -0.5)
        c2 = np.matmul(np.matmul(norm_matrix, c2), np.transpose(norm_matrix))

        transformed_cov = _corr_to_cov(c2, np.diag(cov_array) ** 0.5)
        transformed_cov = _ensure_symmetric(transformed_cov)

        # Return as pandas DataFrame with preserved asset names
        return _create_cov_dataframe(transformed_cov, asset_names)
