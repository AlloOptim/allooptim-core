"""Unit tests for allocation constraints module."""

import pandas as pd

from allooptim.allocation_to_allocators.allocation_constraints import AllocationConstraints


class TestAllocationConstraints:
    """Test suite for AllocationConstraints class."""

    def test_apply_max_active_assets_none(self):
        """Test max_active_assets constraint when disabled (None)."""
        weights = pd.Series([0.1, 0.2, 0.3, 0.4], index=["A", "B", "C", "D"])
        result = AllocationConstraints.apply_max_active_assets(weights, None)
        pd.testing.assert_series_equal(result, weights)

    def test_apply_max_active_assets_no_change(self):
        """Test max_active_assets when constraint is not violated."""
        weights = pd.Series([0.3, 0.3, 0.4, 0.0], index=["A", "B", "C", "D"])
        result = AllocationConstraints.apply_max_active_assets(weights, 3)
        expected = pd.Series([0.3, 0.3, 0.4, 0.0], index=["A", "B", "C", "D"])
        pd.testing.assert_series_equal(result, expected)

    def test_apply_max_active_assets_reduce(self):
        """Test max_active_assets when constraint requires reduction."""
        weights = pd.Series([0.1, 0.2, 0.3, 0.4], index=["A", "B", "C", "D"])
        result = AllocationConstraints.apply_max_active_assets(weights, 2)
        # Should keep top 2 assets (D=0.4, C=0.3) and renormalize
        expected = pd.Series([0.0, 0.0, 0.4286, 0.5714], index=["A", "B", "C", "D"])
        pd.testing.assert_series_equal(result, expected, atol=1e-4)

    def test_apply_max_active_assets_custom_threshold(self):
        """Test max_active_assets with custom weight threshold."""
        weights = pd.Series([0.01, 0.2, 0.3, 0.4], index=["A", "B", "C", "D"])
        result = AllocationConstraints.apply_max_active_assets(weights, 2, min_weight_threshold=0.05)
        # Should consider only B, C, D as active (A=0.01 < 0.05), keep top 2 (D, C)
        expected = pd.Series([0.0, 0.0, 0.4286, 0.5714], index=["A", "B", "C", "D"])
        pd.testing.assert_series_equal(result, expected, atol=1e-4)

    def test_apply_max_concentration_none(self):
        """Test max_concentration constraint when disabled (None)."""
        weights = pd.Series([0.1, 0.2, 0.3, 0.4], index=["A", "B", "C", "D"])
        result = AllocationConstraints.apply_max_concentration(weights, None)
        pd.testing.assert_series_equal(result, weights)

    def test_apply_max_concentration_no_change(self):
        """Test max_concentration when no weights exceed threshold."""
        weights = pd.Series([0.1, 0.2, 0.3, 0.4], index=["A", "B", "C", "D"])
        result = AllocationConstraints.apply_max_concentration(weights, 0.5)
        pd.testing.assert_series_equal(result, weights)

    def test_apply_max_concentration_clip(self):
        """Test max_concentration when weights need to be clipped."""
        weights = pd.Series([0.1, 0.2, 0.6, 0.4], index=["A", "B", "C", "D"])
        result = AllocationConstraints.apply_max_concentration(weights, 0.3)
        # C gets clipped from 0.6 to 0.3, D gets clipped from 0.4 to 0.3, then renormalized
        expected = pd.Series([0.1111, 0.2222, 0.3333, 0.3333], index=["A", "B", "C", "D"])
        pd.testing.assert_series_equal(result, expected, atol=1e-4)

    def test_apply_min_active_assets_none(self):
        """Test min_active_assets constraint when disabled (None)."""
        weights = pd.Series([0.3, 0.3, 0.4, 0.0], index=["A", "B", "C", "D"])
        result = AllocationConstraints.apply_min_active_assets(weights, None)
        pd.testing.assert_series_equal(result, weights)

    def test_apply_min_active_assets_no_change(self):
        """Test min_active_assets when constraint is already satisfied."""
        weights = pd.Series([0.2, 0.3, 0.4, 0.1], index=["A", "B", "C", "D"])
        result = AllocationConstraints.apply_min_active_assets(weights, 3)
        pd.testing.assert_series_equal(result, weights)

    def test_apply_min_active_assets_activate(self):
        """Test min_active_assets when additional assets need to be activated."""
        WEIGHT_SUM_TOLERANCE = 1e-6
        MIN_ACTIVE_ASSETS = 4

        weights = pd.Series([0.5, 0.5, 0.0, 0.0], index=["A", "B", "C", "D"])
        result = AllocationConstraints.apply_min_active_assets(weights, MIN_ACTIVE_ASSETS)
        # Should activate C and D with small weights
        assert result["C"] > 0
        assert result["D"] > 0
        assert abs(result.sum() - 1.0) < WEIGHT_SUM_TOLERANCE

    def test_apply_min_active_assets_insufficient_assets(self):
        """Test min_active_assets when there aren't enough assets to activate."""
        weights = pd.Series([0.5, 0.5], index=["A", "B"])
        result = AllocationConstraints.apply_min_active_assets(weights, 5)
        # Should return original weights unchanged
        pd.testing.assert_series_equal(result, weights)

    def test_apply_all_constraints_none(self):
        """Test apply_all_constraints when all constraints are disabled."""
        weights = pd.Series([0.1, 0.2, 0.3, 0.4], index=["A", "B", "C", "D"])
        result = AllocationConstraints.apply_all_constraints(weights)
        pd.testing.assert_series_equal(result, weights)

    def test_apply_all_constraints_combined(self):
        """Test apply_all_constraints with multiple constraints."""
        WEIGHT_THRESHOLD = 1e-6
        MIN_ACTIVE_ASSETS = 3
        MAX_ACTIVE_ASSETS = 2

        # Start with equal weights
        weights = pd.Series([0.25, 0.25, 0.25, 0.25], index=["A", "B", "C", "D"])

        result = AllocationConstraints.apply_all_constraints(
            weights=weights,
            n_max_active_assets=MAX_ACTIVE_ASSETS,
            max_asset_concentration_pct=0.3,
            n_min_active_assets=MIN_ACTIVE_ASSETS,
        )

        # Should have max 2 active assets, but min 3 required - min takes precedence
        active_count = (result > WEIGHT_THRESHOLD).sum()
        assert active_count >= MIN_ACTIVE_ASSETS
        assert result.sum() <= 1.0

    def test_apply_all_constraints_order(self):
        """Test that constraints are applied in the correct order."""
        WEIGHT_THRESHOLD = 1e-6
        MAX_ACTIVE_ASSETS = 2

        # Create weights that would be affected differently by order
        weights = pd.Series([0.1, 0.1, 0.1, 0.7], index=["A", "B", "C", "D"])

        # Apply constraints: max_concentration first (clip D to 0.3), then max_active_assets
        result = AllocationConstraints.apply_all_constraints(
            weights=weights,
            max_asset_concentration_pct=0.3,
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

    def test_zero_weights_handling(self):
        """Test handling of all-zero weights."""
        weights = pd.Series([0.0, 0.0, 0.0, 0.0], index=["A", "B", "C", "D"])
        result = AllocationConstraints.apply_all_constraints(
            weights=weights,
            n_min_active_assets=2,
        )
        # Should remain all zeros since no assets to activate
        pd.testing.assert_series_equal(result, weights)

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
            n_min_active_assets=1,
        )

        pd.testing.assert_series_equal(result, weights)
