import time
from datetime import datetime

import numpy as np
import pandas as pd
from common.allocation.allocation_to_allocators import OPTIMIZER_LIST

from allo_optim.optimizer.allocation_metric import estimate_linear_moments


def main():
    # Generate proper test data: historical returns for moment estimation
    np.random.seed(42)
    n_observations = 100
    returns_data = np.random.multivariate_normal(
        mean=[0.1, 0.12, 0.08, 0.09, 0.11],
        cov=[
            [0.04, 0.01, 0.005, 0.002, 0.003],
            [0.01, 0.05, 0.002, 0.001, 0.004],
            [0.005, 0.002, 0.03, 0.001, 0.002],
            [0.002, 0.001, 0.001, 0.02, 0.003],
            [0.003, 0.004, 0.002, 0.003, 0.05],
        ],
        size=n_observations,
    )

    # Calculate expected returns and covariance from historical data
    mu = np.mean(returns_data, axis=0)
    cov = np.cov(returns_data, rowvar=False)

    asset_names = ["AAPL", "GOOGL", "MSFT", "AMZN", "FB"]

    mu = pd.Series(mu, index=asset_names)
    cov = pd.DataFrame(cov, index=mu.index, columns=mu.index)

    df_prices = pd.DataFrame(
        np.cumprod(1 + returns_data, axis=0),
        columns=mu.index.tolist(),
    )
    l_moments = estimate_linear_moments(df_prices)

    print(f"Expected returns: {mu}")
    print(f"Expected returns shape: {mu.shape}")
    print(f"Covariance matrix shape: {cov.shape}")

    duration_per_optimizer = {}

    for optimizer in OPTIMIZER_LIST:
        print(f"\n=== Testing optimizer: {optimizer.name} ===")
        for _ in range(2):
            optimizer.fit(df_prices=df_prices)

            today = datetime.now()
            start_time = time.time()
            weights_basic = optimizer.allocate(
                ds_mu=mu, df_cov=cov, df_prices=df_prices, time=today, l_moments=l_moments
            )
            print(f"Optimal weights: {weights_basic}")

            duration = time.time() - start_time
            duration_per_optimizer[optimizer.name] = duration

    print("\n=== Duration per optimizer ===")
    for name, duration in duration_per_optimizer.items():
        print(f"{name}: {duration:.4f} seconds")


if __name__ == "__main__":
    main()
