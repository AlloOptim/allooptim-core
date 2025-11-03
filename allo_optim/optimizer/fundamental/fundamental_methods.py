"""
Pure Fundamental-Based Portfolio Allocation
Uses 5 key fundamental metrics to determine portfolio weights
No price data, returns, or covariance analysis
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import yfinance as yf
from pydantic import BaseModel, model_validator

from allo_optim.config.default_pydantic_config import DEFAULT_PYDANTIC_CONFIG

logger = logging.getLogger(__name__)

# Constants for fundamental analysis thresholds
DEBT_TO_EQUITY_PERCENTAGE_THRESHOLD = 50  # Above this, likely expressed as percentage
MINIMUM_WEIGHT_DISPLAY_THRESHOLD = 0.001  # 0.1% minimum weight to display


class FundamentalData(BaseModel):
	"""Fundamental data for a single ticker"""

	model_config = DEFAULT_PYDANTIC_CONFIG

	ticker: str
	market_cap: Optional[float] = None
	roe: Optional[float] = None  # Return on Equity
	debt_to_equity: Optional[float] = None
	pb_ratio: Optional[float] = None  # Price to Book ratio
	current_ratio: Optional[float] = None

	@property
	def is_valid(self) -> bool:
		"""Determine if this fundamental data is valid"""

		return any(
			[
				self.market_cap is not None,
				self.roe is not None,
				self.debt_to_equity is not None,
				self.pb_ratio is not None,
				self.current_ratio is not None,
			]
		)


class BalancedFundamentalConfig(BaseModel):
	"""Configuration for fundamental analysis weights"""

	model_config = DEFAULT_PYDANTIC_CONFIG

	# Metric weights (must sum to 1.0)
	market_cap_weight: float = 0.25  # Company size/stability
	roe_weight: float = 0.25  # Return on Equity - profitability
	debt_to_equity_weight: float = 0.20  # Financial leverage/risk
	pb_ratio_weight: float = 0.15  # Price to Book - valuation
	current_ratio_weight: float = 0.15  # Liquidity/short-term health

	# Scoring preferences (higher is better after normalization)
	prefer_large_cap: bool = True  # True = prefer large cap, False = prefer small cap
	prefer_low_debt: bool = True  # True = prefer low debt/equity ratio
	prefer_low_pb: bool = True  # True = prefer low P/B (value investing)

	# Market cap tiers (in USD)
	large_cap_threshold: float = 10e9  # > $10B
	small_cap_threshold: float = 2e9  # < $2B

	# ROE thresholds
	excellent_roe: float = 0.20  # 20%+ is excellent
	good_roe: float = 0.15  # 15%+ is good

	# Debt/Equity thresholds
	low_debt: float = 0.5  # < 0.5 is conservative
	high_debt: float = 2.0  # > 2.0 is aggressive

	# P/B thresholds
	value_pb: float = 3.0  # < 3.0 might be undervalued

	# Current ratio thresholds
	healthy_current_ratio: float = 1.5  # > 1.5 is healthy liquidity

	@model_validator(mode="after")
	def validate_weights(self):
		"""Validate that weights sum to 1.0"""
		total = (
			self.market_cap_weight
			+ self.roe_weight
			+ self.debt_to_equity_weight
			+ self.pb_ratio_weight
			+ self.current_ratio_weight
		)
		if not np.isclose(total, 1.0):
			raise ValueError(f"Metric weights must sum to 1.0, got {total}")

		return self


class QualityGrowthFundamentalConfig(BalancedFundamentalConfig):
	"""Configuration favoring quality growth stocks"""

	market_cap_weight: float = 0.20
	roe_weight: float = 0.40
	debt_to_equity_weight: float = 0.20
	pb_ratio_weight: float = 0.05
	current_ratio_weight: float = 0.15
	prefer_large_cap: bool = True
	prefer_low_debt: bool = True
	prefer_low_pb: bool = False  # Growth stocks often have high P/B


class ValueInvestingFundamentalConfig(BalancedFundamentalConfig):
	"""Configuration favoring value investing"""

	market_cap_weight: float = 0.15
	roe_weight: float = 0.30
	debt_to_equity_weight: float = 0.25
	pb_ratio_weight: float = 0.25
	current_ratio_weight: float = 0.05
	prefer_large_cap: bool = True
	prefer_low_debt: bool = True
	prefer_low_pb: bool = True


class OnlyMarketCapFundamentalConfig(BalancedFundamentalConfig):
	"""Configuration favoring market cap"""

	market_cap_weight: float = 1.0
	roe_weight: float = 0.0
	debt_to_equity_weight: float = 0.0
	pb_ratio_weight: float = 0.0
	current_ratio_weight: float = 0.0


def get_fundamental_data(today: datetime, tickers: list[str], batch_size: int = 1000) -> list[FundamentalData]:
	"""
	Download fundamental data for multiple stocks using batch processing

	Args:
	    tickers: list of ticker symbols to fetch data for
	    batch_size: Maximum number of tickers to process in one batch

	Returns:
	    list of FundamentalData objects, one for each ticker
	"""
	all_results = []

	logger.debug(f"Fetching fundamental data for {len(tickers)} tickers in batches of {batch_size}")

	if today - datetime.now() > timedelta(days=1):
		logger.error("Data fetching is only supported for the current day.")

	# Process tickers in batches
	for i in range(0, len(tickers), batch_size):
		batch_tickers = tickers[i : i + batch_size]
		batch_num = i // batch_size + 1
		total_batches = (len(tickers) + batch_size - 1) // batch_size

		logger.debug(f"Processing batch {batch_num}/{total_batches}: {len(batch_tickers)} tickers")

		try:
			tickers_data = yf.Tickers(
				tickers=batch_tickers,
			)

			# Process each ticker in the batch
			for ticker in batch_tickers:
				try:
					info = tickers_data.tickers[ticker].info

					# Extract fundamental metrics
					market_cap = info.get("marketCap")
					roe = info.get("returnOnEquity")
					debt_to_equity = info.get("debtToEquity")
					pb_ratio = info.get("priceToBook")
					current_ratio = info.get("currentRatio")

					# Handle debt_to_equity format variations (some APIs return as percentage)
					if debt_to_equity is not None and debt_to_equity > DEBT_TO_EQUITY_PERCENTAGE_THRESHOLD:
						debt_to_equity = debt_to_equity / 100.0

					# Create FundamentalData object
					fund_data = FundamentalData(
						ticker=ticker,
						market_cap=market_cap,
						roe=roe,
						debt_to_equity=debt_to_equity,
						pb_ratio=pb_ratio,
						current_ratio=current_ratio,
					)

					all_results.append(fund_data)

					if fund_data.is_valid:
						logger.debug(f"  ✓ {ticker}")
					else:
						logger.warning(f"  ✗ {ticker} (no valid data)")

				except Exception as e:
					logger.error(f"  ✗ {ticker}: {e}")
					# Create empty FundamentalData for failed ticker
					all_results.append(FundamentalData(ticker=ticker))

		except Exception as e:
			logger.error(f"Batch {batch_num} failed: {e}")
			# Create empty FundamentalData for all tickers in failed batch
			for ticker in batch_tickers:
				all_results.append(FundamentalData(ticker=ticker))

	# Summary statistics
	valid_count = sum(1 for data in all_results if data.is_valid)
	logger.debug(f"Fundamental data summary: {valid_count}/{len(tickers)} tickers successful")

	return all_results


def normalize_metric(values: np.ndarray, inverse: bool = False) -> np.ndarray:
	"""
	Normalize values to 0-1 range using min-max normalization

	Args:
	    values: Array of metric values
	    inverse: If True, invert so lower values get higher scores

	Returns:
	    Normalized scores between 0 and 1
	"""
	values = np.array(values, dtype=float)

	# Handle edge cases
	if len(values) == 0 or np.all(np.isnan(values)):
		return np.zeros_like(values)

	# Remove NaN for min/max calculation
	valid_values = values[~np.isnan(values)]
	if len(valid_values) == 0:
		return np.zeros_like(values)

	min_val = np.nanmin(valid_values)
	max_val = np.nanmax(valid_values)

	# If all values are the same, return equal scores
	if np.isclose(min_val, max_val):
		result = np.ones_like(values) * 0.5
		result[np.isnan(values)] = 0
		return result

	# Normalize
	normalized = (values - min_val) / (max_val - min_val)

	# Invert if needed (for metrics where lower is better)
	if inverse:
		normalized = 1.0 - normalized

	# Set NaN values to 0 (worst score)
	normalized[np.isnan(normalized)] = 0

	return normalized


def calculate_fundamental_scores(fundamentals: list[FundamentalData], config: BalancedFundamentalConfig) -> np.ndarray:
	"""
	Calculate composite fundamental scores for each stock

	Args:
	    fundamentals: list of FundamentalData objects
	    config: Configuration for scoring

	Returns:
	    Array of composite scores for each stock
	"""
	n = len(fundamentals)

	# Extract metrics into arrays from FundamentalData objects
	market_caps = np.array([f.market_cap if f.market_cap is not None else np.nan for f in fundamentals])
	roes = np.array([f.roe if f.roe is not None else np.nan for f in fundamentals])
	debt_ratios = np.array([f.debt_to_equity if f.debt_to_equity is not None else np.nan for f in fundamentals])
	pb_ratios = np.array([f.pb_ratio if f.pb_ratio is not None else np.nan for f in fundamentals])
	current_ratios = np.array([f.current_ratio if f.current_ratio is not None else np.nan for f in fundamentals])

	# Normalize each metric to 0-1 scale
	market_cap_scores = normalize_metric(market_caps, inverse=not config.prefer_large_cap)
	roe_scores = normalize_metric(roes, inverse=False)  # Higher ROE is better
	debt_scores = normalize_metric(debt_ratios, inverse=config.prefer_low_debt)
	pb_scores = normalize_metric(pb_ratios, inverse=config.prefer_low_pb)
	current_ratio_scores = normalize_metric(current_ratios, inverse=False)  # Higher is better

	# Calculate weighted composite score
	composite_scores = (
		config.market_cap_weight * market_cap_scores
		+ config.roe_weight * roe_scores
		+ config.debt_to_equity_weight * debt_scores
		+ config.pb_ratio_weight * pb_scores
		+ config.current_ratio_weight * current_ratio_scores
	)

	return composite_scores


def allocate(
	asset_names: list[str],
	today: datetime,
	config: BalancedFundamentalConfig,
) -> np.ndarray:
	"""
	Allocate portfolio weights based purely on fundamental data

	Uses 5 fundamental metrics:
	1. Market Cap - Company size and stability
	2. ROE (Return on Equity) - Profitability efficiency
	3. Debt-to-Equity - Financial leverage and risk
	4. P/B Ratio (Price to Book) - Valuation metric
	5. Current Ratio - Short-term liquidity

	Args:
	    stock_names: list of stock ticker symbols
	    today: Current date (for logging purposes)
	    config: Configuration for fundamental scoring (optional)

	Returns:
	    np.ndarray: Portfolio weights summing to 1.0
	"""

	logger.debug(f"\n{'='*60}")
	logger.debug(f"FUNDAMENTAL PORTFOLIO ALLOCATION - {today.date()}")
	logger.debug(f"{'='*60}")
	logger.debug(f"\nAnalyzing {len(asset_names)} stocks using 5 fundamental metrics:")
	logger.debug(f"  1. Market Cap (weight: {config.market_cap_weight:.1%})")
	logger.debug(f"  2. ROE - Return on Equity (weight: {config.roe_weight:.1%})")
	logger.debug(f"  3. Debt/Equity Ratio (weight: {config.debt_to_equity_weight:.1%})")
	logger.debug(f"  4. P/B - Price to Book (weight: {config.pb_ratio_weight:.1%})")
	logger.debug(f"  5. Current Ratio (weight: {config.current_ratio_weight:.1%})")
	logger.debug(f"\n{'-'*60}")

	# Download fundamental data using batch processing
	logger.debug(f"Fetching fundamental data for {len(asset_names)} assets using batch processing...")
	fundamentals = get_fundamental_data(today, asset_names)

	# Check if we have any valid data
	valid_count = sum(1 for f in fundamentals if f.is_valid)
	if valid_count == 0:
		logger.warning("⚠ No valid fundamental data found. Returning equal weights.")
		return np.ones(len(asset_names)) / len(asset_names)

	logger.debug(f"\n{'-'*60}")
	logger.debug("FUNDAMENTAL DATA SUMMARY:")
	logger.debug(f"{'-'*60}")

	# Print fundamental data
	for fund in fundamentals:
		if fund.is_valid:
			logger.debug(f"\n{fund.ticker}:")
			logger.debug(f"  Market Cap: ${fund.market_cap/1e9:.2f}B" if fund.market_cap else "  Market Cap: N/A")
			logger.debug(f"  ROE: {fund.roe*100:.2f}%" if fund.roe else "  ROE: N/A")
			logger.debug(f"  Debt/Equity: {fund.debt_to_equity:.2f}" if fund.debt_to_equity else "  Debt/Equity: N/A")
			logger.debug(f"  P/B Ratio: {fund.pb_ratio:.2f}" if fund.pb_ratio else "  P/B Ratio: N/A")
			logger.debug(f"  Current Ratio: {fund.current_ratio:.2f}" if fund.current_ratio else "  Current Ratio: N/A")
		else:
			logger.warning(f"\n{fund.ticker}: No data available")

	# Calculate composite scores
	scores = calculate_fundamental_scores(fundamentals, config)

	# Convert scores to weights (score-weighted allocation)
	total_score = np.sum(scores)

	if total_score == 0 or np.isnan(total_score):
		logger.error("\n⚠ All scores are zero. Returning equal weights.")
		weights = np.ones(len(asset_names)) / len(asset_names)
	else:
		weights = scores / total_score

	# Print results
	logger.debug(f"\n{'-'*60}")
	logger.debug("ALLOCATION RESULTS:")
	logger.debug(f"{'-'*60}\n")

	# Sort by weight for display
	sorted_indices = np.argsort(weights)[::-1]

	for idx in sorted_indices:
		ticker = asset_names[idx]
		weight = weights[idx]
		score = scores[idx]

		if weight > MINIMUM_WEIGHT_DISPLAY_THRESHOLD:
			logger.debug(f"{ticker:8s}: {weight:6.2%}  (score: {score:.3f})")

	logger.debug(f"\n{'-'*60}")
	logger.debug(f"Total weight: {weights.sum():.6f}")
	logger.debug(f"Number of positions: {np.sum(weights > MINIMUM_WEIGHT_DISPLAY_THRESHOLD)}")
	logger.debug(f"{'='*60}\n")

	# Final validation
	assert np.all(weights >= 0), "Negative weights detected!"
	assert np.all(weights <= 1), "Weights exceed 1!"
	assert np.isclose(weights.sum(), 1.0), f"Weights sum to {weights.sum()}, not 1!"

	return weights


# Example usage
if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)

	from allo_optim.config.stock_universe import everything_in_alpaca

	# Example stocks
	stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "JNJ", "WMT"]
	# stock_universe = everything_in_alpaca()
	# stocks = [s.symbol for s in stock_universe]

	today = datetime.now() - timedelta(days=10)
	start = time.time()

	# Default configuration (balanced approach)
	logger.info("\n" + "=" * 60)
	logger.info("EXAMPLE 1: Balanced Fundamental Approach")
	logger.info("=" * 60)
	config1 = BalancedFundamentalConfig()
	weights1 = allocate(stocks, today, config1)
	logger.info(f"weights: {weights1}")

	# Value investing configuration (prefer low P/B, low debt, high ROE)
	logger.info("\n\n" + "=" * 60)
	logger.info("EXAMPLE 2: Value Investing Approach")
	logger.info("=" * 60)
	config2 = ValueInvestingFundamentalConfig()
	weights2 = allocate(stocks, today, config2)
	logger.info(f"weights: {weights2}")

	# Growth/momentum configuration (prefer high ROE, less concerned with valuation)
	logger.info("\n\n" + "=" * 60)
	logger.info("EXAMPLE 3: Quality Growth Approach")
	logger.info("=" * 60)
	config3 = QualityGrowthFundamentalConfig()
	weights3 = allocate(stocks, today, config3)
	logger.info(f"weights: {weights3}")

	duration = time.time() - start
	logger.info(f"\nTotal execution time: {duration:.2f} seconds")
