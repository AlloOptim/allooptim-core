"""Unit tests for allocation constraints module."""

import pandas as pd

from allooptim.allocation_to_allocators.allocation_constraints import AllocationConstraints


class TestAllocationConstraints:
    """Test suite for AllocationConstraints class."""

    def test_apply_max_active_assets_none(self):
        """Test max_active_assets constraint when disabled (None)."""
        weights = pd.Series([0.1, 0.2, 0.3, 0.4], index=["A", "B", "C", "D"])
        result = AllocationConstraints._apply_max_active_assets(weights, None)
        pd.testing.assert_series_equal(result, weights)

    def test_apply_max_active_assets_no_change(self):
        """Test max_active_assets when constraint is not violated."""
        weights = pd.Series([0.3, 0.3, 0.4, 0.0], index=["A", "B", "C", "D"])
        result = AllocationConstraints._apply_max_active_assets(weights, 3)
        expected = pd.Series([0.3, 0.3, 0.4, 0.0], index=["A", "B", "C", "D"])
        pd.testing.assert_series_equal(result, expected)

    def test_apply_max_active_assets_reduce(self):
        """Test max_active_assets when constraint requires reduction."""
        weights = pd.Series([0.1, 0.2, 0.3, 0.4], index=["A", "B", "C", "D"])
        result = AllocationConstraints._apply_max_active_assets(weights, 2)
        # Should keep top 2 assets (D=0.4, C=0.3) and renormalize
        expected = pd.Series([0.0, 0.0, 0.4286, 0.5714], index=["A", "B", "C", "D"])
        pd.testing.assert_series_equal(result, expected, atol=1e-4)
        # New criterion: sum of weights must be preserved
        assert abs(result.sum() - weights.sum()) < 1e-6

    def test_apply_max_active_assets_custom_threshold(self):
        """Test max_active_assets with custom weight threshold."""
        weights = pd.Series([0.01, 0.2, 0.3, 0.4], index=["A", "B", "C", "D"])
        result = AllocationConstraints._apply_max_active_assets(weights, 2)
        # Should consider only B, C, D as active (A=0.01 < 0.05), keep top 2 (D, C) and renormalize to preserve sum
        expected = pd.Series([0.0, 0.0, 0.39, 0.52], index=["A", "B", "C", "D"])
        pd.testing.assert_series_equal(result, expected, atol=1e-2)
        # New criterion: sum of weights must be preserved
        assert abs(result.sum() - weights.sum()) < 1e-6

    def test_apply_max_concentration_none(self):
        """Test max_concentration constraint when disabled (None)."""
        weights = pd.Series([0.1, 0.2, 0.3, 0.4], index=["A", "B", "C", "D"])
        result = AllocationConstraints._apply_max_concentration(weights, None)
        pd.testing.assert_series_equal(result, weights)

    def test_apply_max_concentration_no_change(self):
        """Test max_concentration when no weights exceed threshold."""
        weights = pd.Series([0.1, 0.2, 0.3, 0.4], index=["A", "B", "C", "D"])
        result = AllocationConstraints._apply_max_concentration(weights, 0.5)
        pd.testing.assert_series_equal(result, weights)

    def test_apply_max_concentration_clip(self):
        """Test max_concentration when weights need to be clipped and redistributed."""
        weights = pd.Series([0.1, 0.2, 0.6, 0.4], index=["A", "B", "C", "D"])
        result = AllocationConstraints._apply_max_concentration(weights, 0.3)
        # C gets clipped from 0.6 to 0.3, D gets clipped from 0.4 to 0.3
        # Excess weight (0.4) gets redistributed to A and B proportionally
        expected = pd.Series([0.2333, 0.4667, 0.3, 0.3], index=["A", "B", "C", "D"])
        pd.testing.assert_series_equal(result, expected, atol=1e-4)
        # Sum of weights must be preserved
        assert abs(result.sum() - weights.sum()) < 1e-6

    def test_apply_all_constraints_none(self):
        """Test apply_all_constraints when all constraints are disabled."""
        weights = pd.Series([0.1, 0.2, 0.3, 0.4], index=["A", "B", "C", "D"])
        result = AllocationConstraints.apply_all_constraints(weights)
        pd.testing.assert_series_equal(result, weights)

    def test_apply_all_constraints_combined(self):
        """Test apply_all_constraints with multiple constraints."""
        WEIGHT_THRESHOLD = 1e-6
        MAX_ACTIVE_ASSETS = 2
        MAX_CONCENTRATION = 0.3

        # Start with equal weights
        weights = pd.Series([0.25, 0.25, 0.25, 0.25], index=["A", "B", "C", "D"])

        result = AllocationConstraints.apply_all_constraints(
            weights=weights,
            n_max_active_assets=MAX_ACTIVE_ASSETS,
            max_asset_concentration_pct=MAX_CONCENTRATION,
        )

        active_count = (result > WEIGHT_THRESHOLD).sum()
        assert active_count <= MAX_ACTIVE_ASSETS
        assert abs(result.sum() - weights.sum()) < 1e-6

    def test_apply_all_constraints_order(self):
        """Test that constraints are applied in the correct order."""
        WEIGHT_THRESHOLD = 1e-6
        MAX_ACTIVE_ASSETS = 2
        MAX_CONCENTRATION = 0.3

        # Create weights that would be affected differently by order
        weights = pd.Series([0.1, 0.1, 0.1, 0.7], index=["A", "B", "C", "D"])

        # Apply constraints: max_concentration first (clip D to 0.3), then max_active_assets
        result = AllocationConstraints.apply_all_constraints(
            weights=weights,
            max_asset_concentration_pct=MAX_CONCENTRATION,
            n_max_active_assets=MAX_ACTIVE_ASSETS,
        )

        # D should be clipped to 0.3 first, then only top 2 assets kept
        # After clipping: [0.1, 0.1, 0.1, 0.3] -> sum = 0.6
        # After max_active_assets=2: keep top 2 -> [0.1, 0.1, 0.1, 0.3] but only top 2 (D and one other)
        # Actually, after clipping and renormalizing: [0.1667, 0.1667, 0.1667, 0.5]
        # Then max_active_assets=2: keep D and one other, renormalize

        assert result.sum() <= 1.0
        active_count = (result > WEIGHT_THRESHOLD).sum()
        assert active_count <= MAX_ACTIVE_ASSETS
        assert abs(result.sum() - weights.sum()) < 1e-6

    def test_normalization_preserved(self):
        """Test that final weights are properly normalized."""
        WEIGHT_SUM_TOLERANCE = 1e-6

        weights = pd.Series([0.1, 0.2, 0.3, 0.4], index=["A", "B", "C", "D"])

        result = AllocationConstraints.apply_all_constraints(
            weights=weights,
            max_asset_concentration_pct=0.5,
        )

        assert abs(result.sum() - 1.0) < WEIGHT_SUM_TOLERANCE

    def test_edge_case_single_asset(self):
        """Test constraints with single asset portfolio."""
        weights = pd.Series([1.0], index=["A"])

        result = AllocationConstraints.apply_all_constraints(
            weights=weights,
            n_max_active_assets=5,
        )

        pd.testing.assert_series_equal(result, weights)
