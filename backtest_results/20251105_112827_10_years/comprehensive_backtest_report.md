# Comprehensive Allocation Algorithm Backtest Report

**Generated:** 2025-11-05 14:11:33  
**Period:** 2014-12-31 to 2024-12-31  
**Rebalancing Frequency:** Every 10 trading days  
**Lookback Window:** 90 days  
**Fallback Strategy:** Equal Weights  

## Executive Summary

This report presents a comprehensive backtest of 19 allocation algorithms including:
- 13 individual optimizers from the enhanced allocation framework
- A2A ensemble optimizer (simple average of all individual optimizers)  
- S&P 500 benchmark (100% SPY allocation)

### Key Findings


**Best Sharpe Ratio:** HigherMoment (1.195)
**Best CAGR:** EfficientRisk (67.29%)
**Lowest Max Drawdown:** HigherMoment (33.47%)

## Performance Metrics

### Summary Statistics

| Optimizer | Sharpe Ratio | CAGR | Max Drawdown | Annual Vol | Risk-Adj Return | Total Return |
|-----------|--------------|------|--------------|------------|-----------------|--------------|
| HigherMoment | 1.195 | 27.95% | 33.47% | 20.77% | 6.05% | 1130.09% |
| NCOSharpeOptimizer | 0.929 | 19.52% | 39.87% | 19.01% | 0.65% | 514.35% |
| CappedMomentum | 1.086 | 22.27% | 37.22% | 18.23% | 3.57% | 674.97% |
| Naive | 0.773 | 15.44% | 38.09% | 18.15% | -2.12% | 331.44% |
| CMA_ROBUST_SHARPE | 0.748 | 14.85% | 38.22% | 18.02% | -2.53% | 309.47% |
| CMA_MAX_DRAWDOWN | 0.765 | 15.42% | 38.08% | 18.36% | -2.31% | 330.79% |
| AdjustedReturns_MeanVariance | 1.126 | 64.20% | 52.84% | 56.20% | 9.05% | 15499.39% |
| AdjustedReturns_SemiVariance | 1.116 | 60.89% | 57.28% | 53.59% | 8.23% | 12575.02% |
| MaxSharpe | 1.032 | 18.40% | 35.31% | 15.64% | 2.50% | 458.36% |
| EfficientReturn | 0.845 | 16.52% | 37.05% | 17.59% | -0.73% | 374.42% |
| CMA_CVAR | 0.769 | 15.50% | 38.00% | 18.36% | -2.25% | 333.57% |
| CMA_MEAN_VARIANCE | 0.808 | 16.24% | 38.33% | 18.25% | -1.51% | 363.09% |
| CMA_SORTINO | 0.762 | 14.97% | 36.93% | 17.79% | -2.24% | 313.97% |
| PSO_LMoments | 0.788 | 15.78% | 37.58% | 18.19% | -1.86% | 344.75% |
| EfficientRisk | 1.131 | 67.29% | 57.60% | 58.95% | 9.71% | 18752.61% |
| CMA_L_MOMENTS | 0.783 | 15.80% | 38.07% | 18.35% | -1.98% | 345.41% |
| PSO_MeanVariance | 0.793 | 15.89% | 37.96% | 18.18% | -1.76% | 348.99% |
| SPY_Benchmark | 0.652 | 12.52% | 33.72% | 17.36% | -4.05% | 232.35% |
| A2A_Ensemble | 1.162 | 27.04% | 40.18% | 20.76% | 5.36% | 1043.89% |+654


### Detailed Performance Analysis

#### Returns Distribution Statistics

| Optimizer | Mean Daily Return | Volatility | Skewness | Kurtosis | Min Return | Max Return |
|-----------|-------------------|------------|----------|----------|------------|------------|
| HigherMoment | 26.8155% | 20.7665% | 0.542 | 20.110 | -3293.461% | 3867.964% |
| NCOSharpeOptimizer | 19.6566% | 19.0054% | -0.547 | 20.071 | -3317.350% | 3250.914% |
| CappedMomentum | 21.7981% | 18.2293% | -0.901 | 14.700 | -3224.565% | 2224.678% |
| Naive | 16.0218% | 18.1462% | -0.606 | 19.171 | -3264.841% | 2915.843% |
| CMA_ROBUST_SHARPE | 15.4842% | 18.0175% | -0.566 | 20.070 | -3266.283% | 3026.485% |
| CMA_MAX_DRAWDOWN | 16.0465% | 18.3582% | -0.622 | 19.430 | -3336.554% | 2940.829% |
| AdjustedReturns_MeanVariance | 65.2570% | 56.2024% | 0.648 | 7.125 | -5307.832% | 7565.820% |
| AdjustedReturns_SemiVariance | 61.8203% | 53.5944% | 0.620 | 7.986 | -5307.832% | 7565.820% |
| MaxSharpe | 18.1334% | 15.6367% | -1.065 | 21.710 | -3065.290% | 2315.519% |
| EfficientReturn | 16.8576% | 17.5916% | -0.817 | 17.013 | -3206.680% | 2326.965% |
| CMA_CVAR | 16.1087% | 18.3556% | -0.592 | 19.386 | -3306.387% | 2982.393% |
| CMA_MEAN_VARIANCE | 16.7361% | 18.2477% | -0.583 | 18.592 | -3224.856% | 2983.287% |
| CMA_SORTINO | 15.5511% | 17.7923% | -0.578 | 18.799 | -3194.256% | 2815.396% |
| PSO_LMoments | 16.3290% | 18.1893% | -0.644 | 18.529 | -3262.210% | 2798.095% |
| EfficientRisk | 68.6550% | 58.9481% | 0.702 | 7.255 | -5307.832% | 7565.820% |
| CMA_L_MOMENTS | 16.3719% | 18.3497% | -0.558 | 19.418 | -3325.899% | 3083.594% |
| PSO_MeanVariance | 16.4208% | 18.1807% | -0.647 | 19.001 | -3262.504% | 2846.159% |
| SPY_Benchmark | 13.3160% | 17.3634% | -0.563 | 13.358 | -2757.476% | 2283.197% |
| A2A_Ensemble | 26.1214% | 20.7641% | -0.626 | 12.792 | -3484.335% | 2894.116% |

#### Portfolio Turnover Analysis

| Optimizer | Mean Turnover | Turnover Std | Min Turnover | Max Turnover | Median Turnover |
|-----------|---------------|--------------|--------------|--------------|-----------------|
| HigherMoment | 59.15% | 25.58% | 0.00% | 100.00% | 60.12% |
| NCOSharpeOptimizer | 44.00% | 18.13% | 25.26% | 94.84% | 38.60% |
| CappedMomentum | 26.14% | 9.75% | 11.13% | 63.23% | 24.18% |
| Naive | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| CMA_ROBUST_SHARPE | 14.47% | 1.94% | 11.35% | 35.18% | 14.22% |
| CMA_MAX_DRAWDOWN | 6.84% | 2.04% | 6.45% | 30.37% | 6.50% |
| AdjustedReturns_MeanVariance | 44.18% | 43.57% | 0.00% | 100.00% | 38.59% |
| AdjustedReturns_SemiVariance | 47.42% | 41.73% | 0.00% | 100.00% | 50.13% |
| MaxSharpe | 41.32% | 12.76% | 6.99% | 76.94% | 41.57% |
| EfficientReturn | 11.27% | 9.26% | 0.70% | 53.85% | 8.94% |
| CMA_CVAR | 6.92% | 1.39% | 6.65% | 24.42% | 6.68% |
| CMA_MEAN_VARIANCE | 14.61% | 2.34% | 9.35% | 36.69% | 14.44% |
| CMA_SORTINO | 14.70% | 2.59% | 11.69% | 40.33% | 14.35% |
| PSO_LMoments | 30.52% | 1.99% | 19.82% | 35.44% | 30.56% |
| EfficientRisk | 48.99% | 50.09% | 0.00% | 100.00% | 0.00% |
| CMA_L_MOMENTS | 14.51% | 2.48% | 9.22% | 35.04% | 14.25% |
| PSO_MeanVariance | 30.48% | 3.49% | 0.00% | 35.57% | 30.82% |
| SPY_Benchmark | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| A2A_Ensemble | 18.71% | 8.03% | 6.29% | 33.07% | 17.46% |

#### Portfolio Change Rate Analysis

| Optimizer | Mean Change Rate | Change Rate Std | Min Change Rate | Max Change Rate | Median Change Rate |
|-----------|------------------|-----------------|-----------------|-----------------|-------------------|
| HigherMoment | inf% | nan% | -55.51% | inf% | 31012434504582.10% |
| NCOSharpeOptimizer | inf% | nan% | inf% | inf% | nan% |
| CappedMomentum | inf% | nan% | -31.44% | inf% | nan% |
| Naive | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| CMA_ROBUST_SHARPE | 48970094671.26% | 769578730677.20% | 52.16% | 12094882092901.60% | 1973.64% |
| CMA_MAX_DRAWDOWN | 6974.49% | 2562.85% | 1395.53% | 12683.62% | 4812.32% |
| AdjustedReturns_MeanVariance | inf% | nan% | 0.00% | inf% | nan% |
| AdjustedReturns_SemiVariance | inf% | nan% | 0.00% | inf% | nan% |
| MaxSharpe | 367.04% | 152.38% | 43.33% | 1075.19% | 362.27% |
| EfficientReturn | inf% | nan% | -16.13% | inf% | 1.31% |
| CMA_CVAR | 99957.89% | 90077.83% | 6869.65% | 192354.61% | 128662.12% |
| CMA_MEAN_VARIANCE | 1917526.54% | 28876041.27% | 83.06% | 453823933.65% | 2157.02% |
| CMA_SORTINO | 155476071.54% | 2440443209.09% | 76.30% | 38354762544.79% | 1746.73% |
| PSO_LMoments | 593.86% | 2630.73% | 62.48% | 36369.48% | 204.57% |
| EfficientRisk | inf% | nan% | 0.00% | inf% | 0.00% |
| CMA_L_MOMENTS | 330618.45% | 4553523.29% | 30.92% | 71540285.20% | 2070.43% |
| PSO_MeanVariance | 461.89% | 2245.93% | 0.00% | 34723.56% | 201.00% |
| SPY_Benchmark | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| A2A_Ensemble | 11.28% | 7.66% | 0.59% | 42.47% | 8.78% |

#### Portfolio Diversification Metrics

##### Assets Above Threshold (Mean Count)

| Optimizer | 5% Above Equal Weight | 10% Above Equal Weight | 50% Above Equal Weight | 100% Above Equal Weight |
|-----------|----------------------|------------------------|------------------------|-------------------------|
| HigherMoment | 17.7 (6.3%) | 17.7 (6.2%) | 17.4 (6.1%) | 17.1 (6.0%) |
| NCOSharpeOptimizer | 144.6 (51.1%) | 142.1 (50.2%) | 120.6 (42.6%) | 95.9 (33.9%) |
| CappedMomentum | 181.3 (64.1%) | 178.1 (62.9%) | 149.8 (52.9%) | 108.6 (38.4%) |
| Naive | 283.0 (100.0%) | 283.0 (100.0%) | 283.0 (100.0%) | 0.0 (0.0%) |
| CMA_ROBUST_SHARPE | 262.6 (92.8%) | 254.2 (89.8%) | 201.5 (71.2%) | 138.5 (48.9%) |
| CMA_MAX_DRAWDOWN | 191.0 (67.5%) | 175.2 (61.9%) | 147.9 (52.3%) | 140.9 (49.8%) |
| AdjustedReturns_MeanVariance | 1.5 (0.5%) | 1.5 (0.5%) | 1.5 (0.5%) | 1.5 (0.5%) |
| AdjustedReturns_SemiVariance | 1.7 (0.6%) | 1.7 (0.6%) | 1.7 (0.6%) | 1.7 (0.6%) |
| MaxSharpe | 282.2 (99.7%) | 156.8 (55.4%) | 32.3 (11.4%) | 30.6 (10.8%) |
| EfficientReturn | 283.0 (100.0%) | 282.6 (99.9%) | 266.5 (94.2%) | 99.5 (35.2%) |
| CMA_CVAR | 191.7 (67.7%) | 182.3 (64.4%) | 150.2 (53.1%) | 141.5 (50.0%) |
| CMA_MEAN_VARIANCE | 259.8 (91.8%) | 250.2 (88.4%) | 194.1 (68.6%) | 136.5 (48.2%) |
| CMA_SORTINO | 259.7 (91.8%) | 250.8 (88.6%) | 196.5 (69.4%) | 136.5 (48.2%) |
| PSO_LMoments | 277.3 (98.0%) | 271.4 (95.9%) | 219.7 (77.6%) | 140.1 (49.5%) |
| EfficientRisk | 1.0 (0.4%) | 1.0 (0.4%) | 1.0 (0.4%) | 1.0 (0.4%) |
| CMA_L_MOMENTS | 259.8 (91.8%) | 250.5 (88.5%) | 196.1 (69.3%) | 136.9 (48.4%) |
| PSO_MeanVariance | 277.3 (98.0%) | 271.5 (95.9%) | 219.2 (77.5%) | 140.0 (49.5%) |
| SPY_Benchmark | 0.0 (0.0%) | 0.0 (0.0%) | 0.0 (0.0%) | 0.0 (0.0%) |
| A2A_Ensemble | 283.0 (100.0%) | 283.0 (100.0%) | 234.2 (82.7%) | 52.6 (18.6%) |

##### Top N Assets Weight Concentration

| Optimizer | Top 5 Assets Weight | Top 10 Assets Weight | Top 50 Assets Weight |
|-----------|--------------------|--------------------|---------------------|
| HigherMoment | 75.4% | 93.8% | 98.3% |
| NCOSharpeOptimizer | 11.6% | 20.5% | 65.5% |
| CappedMomentum | 11.5% | 19.2% | 56.4% |
| Naive | 1.8% | 3.5% | 17.7% |
| CMA_ROBUST_SHARPE | 3.8% | 7.5% | 34.7% |
| CMA_MAX_DRAWDOWN | 3.6% | 7.3% | 36.0% |
| AdjustedReturns_MeanVariance | 99.9% | 100.0% | 100.0% |
| AdjustedReturns_SemiVariance | 99.9% | 100.0% | 100.0% |
| MaxSharpe | 17.6% | 35.1% | 91.5% |
| EfficientReturn | 4.9% | 8.4% | 27.8% |
| CMA_CVAR | 3.6% | 7.2% | 35.7% |
| CMA_MEAN_VARIANCE | 3.9% | 7.7% | 35.7% |
| CMA_SORTINO | 3.8% | 7.7% | 35.6% |
| PSO_LMoments | 3.6% | 7.0% | 31.9% |
| EfficientRisk | 100.0% | 100.0% | 100.0% |
| CMA_L_MOMENTS | 3.8% | 7.6% | 35.5% |
| PSO_MeanVariance | 3.6% | 7.0% | 31.9% |
| SPY_Benchmark | 0.0% | 0.0% | 0.0% |
| A2A_Ensemble | 23.7% | 27.4% | 46.0% |

#### Computational Performance

| Optimizer | Avg Computation Time (s) | Max Computation Time (s) | Avg Memory Usage (MB) | Max Memory Usage (MB) |
|-----------|---------------------------|---------------------------|------------------------|------------------------|
| HigherMoment | 0.0000 | 0.0000 | 0.00 | 0.00 |
| NCOSharpeOptimizer | 0.0000 | 0.0000 | 0.00 | 0.00 |
| CappedMomentum | 0.0000 | 0.0000 | 0.00 | 0.00 |
| Naive | 0.0000 | 0.0000 | 0.00 | 0.00 |
| CMA_ROBUST_SHARPE | 0.0000 | 0.0000 | 0.00 | 0.00 |
| CMA_MAX_DRAWDOWN | 0.0000 | 0.0000 | 0.00 | 0.00 |
| AdjustedReturns_MeanVariance | 0.0000 | 0.0000 | 0.00 | 0.00 |
| AdjustedReturns_SemiVariance | 0.0000 | 0.0000 | 0.00 | 0.00 |
| MaxSharpe | 0.0000 | 0.0000 | 0.00 | 0.00 |
| EfficientReturn | 0.0000 | 0.0000 | 0.00 | 0.00 |
| CMA_CVAR | 0.0000 | 0.0000 | 0.00 | 0.00 |
| CMA_MEAN_VARIANCE | 0.0000 | 0.0000 | 0.00 | 0.00 |
| CMA_SORTINO | 0.0000 | 0.0000 | 0.00 | 0.00 |
| PSO_LMoments | 0.0000 | 0.0000 | 0.00 | 0.00 |
| EfficientRisk | 0.0000 | 0.0000 | 0.00 | 0.00 |
| CMA_L_MOMENTS | 0.0000 | 0.0000 | 0.00 | 0.00 |
| PSO_MeanVariance | 0.0000 | 0.0000 | 0.00 | 0.00 |
| SPY_Benchmark | 0.0000 | 0.0000 | 0.00 | 0.00 |
| A2A_Ensemble | 0.0000 | 0.0000 | 0.00 | 0.00 |

## Optimizer Clustering Analysis

The clustering analysis groups optimizers based on their performance characteristics,
portfolio similarities, and return patterns to identify which algorithms behave similarly.


### Performance Clustering

**Method:** hierarchical_performance  
**Number of Clusters:** 4  

**Cluster 3:** HigherMoment, A2A_Ensemble

**Cluster 4:** NCOSharpeOptimizer, CappedMomentum, MaxSharpe

**Cluster 2:** Naive, CMA_ROBUST_SHARPE, CMA_MAX_DRAWDOWN, EfficientReturn, CMA_CVAR, CMA_MEAN_VARIANCE, CMA_SORTINO, PSO_LMoments, CMA_L_MOMENTS, PSO_MeanVariance, SPY_Benchmark

**Cluster 1:** AdjustedReturns_MeanVariance, AdjustedReturns_SemiVariance, EfficientRisk


### Portfolio Correlation Clustering

**Method:** portfolio_correlation  
**Number of Clusters:** 4  

**Cluster 2:** HigherMoment, NCOSharpeOptimizer, CappedMomentum, AdjustedReturns_MeanVariance, AdjustedReturns_SemiVariance, MaxSharpe, EfficientReturn, CMA_MEAN_VARIANCE, CMA_SORTINO, PSO_LMoments, EfficientRisk, CMA_L_MOMENTS, PSO_MeanVariance, A2A_Ensemble

**Cluster 4:** Naive

**Cluster 3:** CMA_ROBUST_SHARPE

**Cluster 1:** CMA_MAX_DRAWDOWN, CMA_CVAR, SPY_Benchmark


### Returns Correlation Clustering

**Method:** returns_correlation_kmeans  
**Number of Clusters:** 4  

**Cluster 2:** HigherMoment

**Cluster 1:** NCOSharpeOptimizer, CappedMomentum, Naive, CMA_ROBUST_SHARPE, CMA_MAX_DRAWDOWN, MaxSharpe, EfficientReturn, CMA_CVAR, CMA_MEAN_VARIANCE, CMA_SORTINO, PSO_LMoments, CMA_L_MOMENTS, PSO_MeanVariance, SPY_Benchmark

**Cluster 0:** AdjustedReturns_MeanVariance, AdjustedReturns_SemiVariance, EfficientRisk

**Cluster 3:** A2A_Ensemble


### Combined Clustering

**Method:** combined_kmeans  
**Number of Clusters:** 5  

**Cluster 2:** HigherMoment

**Cluster 4:** NCOSharpeOptimizer, MaxSharpe

**Cluster 3:** CappedMomentum, A2A_Ensemble

**Cluster 1:** Naive, CMA_ROBUST_SHARPE, CMA_MAX_DRAWDOWN, EfficientReturn, CMA_CVAR, CMA_MEAN_VARIANCE, CMA_SORTINO, PSO_LMoments, CMA_L_MOMENTS, PSO_MeanVariance, SPY_Benchmark

**Cluster 0:** AdjustedReturns_MeanVariance, AdjustedReturns_SemiVariance, EfficientRisk


### Euclidean Distance Analysis

This analysis computes the mean Euclidean distance between optimizer portfolio weights
across all timesteps, revealing which optimizers make the most similar allocation decisions.


#### Most Similar Optimizer Pairs (Shortest Distances)

| Rank | Optimizer A | Optimizer B | Mean Euclidean Distance |
|------|-------------|-------------|-------------------------|
| 1 | CMA_MAX_DRAWDOWN | CMA_CVAR | 0.0111 |
| 2 | Naive | EfficientReturn | 0.0224 |
| 3 | Naive | PSO_LMoments | 0.0327 |
| 4 | Naive | PSO_MeanVariance | 0.0328 |
| 5 | Naive | CMA_ROBUST_SHARPE | 0.0391 |
| 6 | EfficientReturn | PSO_LMoments | 0.0403 |
| 7 | EfficientReturn | PSO_MeanVariance | 0.0404 |
| 8 | Naive | CMA_SORTINO | 0.0408 |
| 9 | Naive | CMA_L_MOMENTS | 0.0409 |
| 10 | Naive | CMA_MEAN_VARIANCE | 0.0413 |


#### Distance-Based Groupings

Using hierarchical clustering on Euclidean distances, optimizers are grouped into
5 clusters:

**Distance Cluster 4:** HigherMoment

**Distance Cluster 3:** NCOSharpeOptimizer, CappedMomentum, Naive, CMA_ROBUST_SHARPE, CMA_MAX_DRAWDOWN, MaxSharpe, EfficientReturn, CMA_CVAR, CMA_MEAN_VARIANCE, CMA_SORTINO, PSO_LMoments, CMA_L_MOMENTS, PSO_MeanVariance, A2A_Ensemble

**Distance Cluster 1:** AdjustedReturns_MeanVariance, AdjustedReturns_SemiVariance

**Distance Cluster 2:** EfficientRisk

**Distance Cluster 5:** SPY_Benchmark


**Key Insights:**
- Optimizers with small Euclidean distances make very similar allocation decisions
- Distance-based clusters reveal functional similarity beyond theoretical groupings
- The closest pairs often represent variations of the same underlying approach
- Large distances indicate fundamentally different allocation strategies


## Theoretical Optimizer Groupings

Based on the underlying optimization approaches, we can group the algorithms theoretically:

### Mean Reversion & Risk Parity Group
- **RiskParityOptimizer:** Equal risk contribution
- **NaiveOptimizer:** Equal weight allocation
- **EfficientRiskOptimizer:** Risk-based allocation

### Modern Portfolio Theory Group  
- **MeanVarianceParticalSwarmOptimizer:** PSO with mean-variance optimization
- **MeanVarianceAdjustedReturnsOptimizer:** Classical mean-variance with adjusted returns
- **MaxSharpeOptimizer:** Maximum Sharpe ratio optimization

### Alternative Risk Models Group
- **LMomentsParticleSwarmOptimizer:** PSO with L-moments
- **LMomentsAdjustedReturnsOptimizer:** L-moments based allocation
- **HRPOptimizer:** Hierarchical risk parity

### Advanced Optimization Group
- **NCOOptimizer:** Nested clustered optimization
- **MomentumOptimizer:** Momentum-based allocation
- **CongressSenateOptimizer:** Congress trading patterns

### Market-Based Group
- **MarketCapOptimizer:** Market capitalization weighted
- **SPY_Benchmark:** S&P 500 benchmark

### Ensemble Group
- **A2A_Ensemble:** Average of all individual optimizers

## Key Insights and Recommendations

### Performance Insights

1. **Benchmark Comparison**: The S&P 500 benchmark achieved a Sharpe ratio of
0.652 vs A2A ensemble of 1.162
2. **Ensemble Effect**: The A2A ensemble outperformed
the S&P 500 benchmark

3. **Concentration Analysis**: Naive is most concentrated
(top 5 assets: 1.8%)
4. **Diversification Leader**: Naive uses most assets above 5% threshold
(avg: 283.0 assets)

5. **Clustering Analysis**: 4 clusters identified
(avg cluster size: 4.8 assets)
6. **Risk-Return Profile**: HigherMoment leads with Sharpe ratio
1.19 (avg return: 0.00%)

### Algorithm Clustering Insights

1. **Performance Clustering**: Identifies optimizers with similar risk-return profiles
2. **Portfolio Correlation**: Groups algorithms that tend to select similar assets
3. **Returns Pattern**: Clusters based on return distribution characteristics

### Recommendations

1. **Diversification Strategy**: Use optimizers from different theoretical and experimental clusters
2. **Ensemble Optimization**: The A2A approach shows promise for risk diversification
3. **Computational Efficiency**: Consider computation time vs performance trade-offs
4. **Market Regime Sensitivity**: Monitor performance across different market conditions

## Technical Details

### Data Quality
- **Universe Size**: Approximately 400 assets from Alpaca-available universe
- **Data Source**: Yahoo Finance via yfinance library
- **Missing Data Handling**: Forward fill with 80% completeness threshold

### Methodology
- **Rebalancing**: Portfolio weights updated every 5 trading days
- **Lookback Window**: 90 days of historical data for each optimization
- **Execution**: Perfect execution assumed (no slippage, transaction costs, or liquidity constraints)
- **Fallback Strategy**: Equal weights used when optimizers fail

### Risk Considerations
- **Survivorship Bias**: Only includes currently available assets
- **Look-Ahead Bias**: Avoided by using only historical data at each rebalancing point
- **Transaction Costs**: Not included in performance calculations
- **Market Impact**: Not considered due to perfect execution assumption

## Appendix

### Configuration Parameters
- **Start Date**: 2014-12-31
- **End Date**: 2024-12-31
- **Rebalancing Frequency**: 10 trading days
- **Lookback Period**: 90 days
- **Fallback Strategy**: Equal Weights
- **Results Directory**: backtest_results\20251105_112827

### Generated Files
- `performance_comparison.png`: Performance metrics bar charts
- `portfolio_evolution.png`: Portfolio value time series  
- `risk_return_scatter.png`: Risk-return scatter plot
- `clustering_dendrogram.png`: Hierarchical clustering visualization
- `backtest_results.csv`: Detailed results in CSV format
- `optimizer_distances.csv`: Pairwise Euclidean distances between optimizers

---
*This report was generated automatically by the comprehensive backtest framework.*
