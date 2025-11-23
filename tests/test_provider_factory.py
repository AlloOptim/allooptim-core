"""Tests for fundamental data provider factory."""

import os
from unittest.mock import Mock, patch

import pytest

from allooptim.data.fundamental_data import FundamentalData
from allooptim.data.fundamental_providers import (
    FundamentalDataStore,
    SimFinProvider,
    YahooFinanceProvider,
)
from allooptim.data.provider_factory import (
    FundamentalDataProviderFactory,
    UnifiedFundamentalProvider,
)


class TestFundamentalDataProviderFactory:
    """Test the provider factory."""

    def test_factory_simfin_priority(self):
        """Verify SimFin used when API key available."""
        with patch.dict(os.environ, {"SIMFIN_API_KEY": "test_key"}):
            with patch("allooptim.data.provider_factory.SimFinProvider") as mock_simfin:
                with patch("allooptim.data.provider_factory.YahooFinanceProvider") as mock_yahoo:
                    mock_simfin.return_value = Mock(spec=SimFinProvider)
                    mock_yahoo.return_value = Mock(spec=YahooFinanceProvider)

                    provider = FundamentalDataProviderFactory.create_provider()

                    # Should have 2 providers: SimFin first, then Yahoo
                    assert len(provider.providers) == 2
                    mock_simfin.assert_called_once_with(api_key="test_key")
                    mock_yahoo.assert_called_once()
                    assert provider.providers[0] is mock_simfin.return_value
                    assert provider.providers[1] is mock_yahoo.return_value

    def test_factory_yahoo_fallback(self):
        """Verify Yahoo used when no SimFin key."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("allooptim.data.provider_factory.SimFinProvider") as mock_simfin:
                with patch("allooptim.data.provider_factory.YahooFinanceProvider") as mock_yahoo:
                    mock_yahoo.return_value = Mock(spec=YahooFinanceProvider)

                    provider = FundamentalDataProviderFactory.create_provider()

                    # Should have only Yahoo provider
                    assert len(provider.providers) == 1
                    mock_simfin.assert_not_called()
                    mock_yahoo.assert_called_once()
                    assert provider.providers[0] is mock_yahoo.return_value

    def test_factory_caching_enabled(self):
        """Verify caching initialization."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("allooptim.data.provider_factory.YahooFinanceProvider") as mock_yahoo:
                mock_yahoo.return_value = Mock(spec=YahooFinanceProvider)

                provider = FundamentalDataProviderFactory.create_provider(enable_caching=True)

                # Should have cache initialized
                assert provider.cache is not None
                assert isinstance(provider.cache, FundamentalDataStore)

    def test_factory_caching_disabled(self):
        """Verify no caching when disabled."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("allooptim.data.provider_factory.YahooFinanceProvider") as mock_yahoo:
                mock_yahoo.return_value = Mock(spec=YahooFinanceProvider)

                provider = FundamentalDataProviderFactory.create_provider(enable_caching=False)

                # Should not have cache
                assert provider.cache is None

    def test_factory_simfin_import_error_fallback(self):
        """Verify fallback to Yahoo when SimFin import fails."""
        with patch.dict(os.environ, {"SIMFIN_API_KEY": "test_key"}):
            with patch("allooptim.data.provider_factory.SimFinProvider", side_effect=ImportError("No simfin")):
                with patch("allooptim.data.provider_factory.YahooFinanceProvider") as mock_yahoo:
                    mock_yahoo.return_value = Mock(spec=YahooFinanceProvider)

                    provider = FundamentalDataProviderFactory.create_provider()

                    # Should have only Yahoo provider
                    assert len(provider.providers) == 1
                    mock_yahoo.assert_called_once()
                    assert provider.providers[0] is mock_yahoo.return_value

    def test_factory_explicit_api_key(self):
        """Verify explicit API key overrides environment."""
        with patch.dict(os.environ, {"SIMFIN_API_KEY": "env_key"}):
            with patch("allooptim.data.provider_factory.SimFinProvider") as mock_simfin:
                with patch("allooptim.data.provider_factory.YahooFinanceProvider") as mock_yahoo:
                    mock_simfin.return_value = Mock(spec=SimFinProvider)
                    mock_yahoo.return_value = Mock(spec=YahooFinanceProvider)

                    provider = FundamentalDataProviderFactory.create_provider(
                        simfin_api_key="explicit_key"
                    )

                    # Should use explicit key
                    mock_simfin.assert_called_once_with(api_key="explicit_key")

    def test_factory_prefer_simfin_false(self):
        """Verify Yahoo only when SimFin not preferred."""
        with patch.dict(os.environ, {"SIMFIN_API_KEY": "test_key"}):
            with patch("allooptim.data.provider_factory.SimFinProvider") as mock_simfin:
                with patch("allooptim.data.provider_factory.YahooFinanceProvider") as mock_yahoo:
                    mock_yahoo.return_value = Mock(spec=YahooFinanceProvider)

                    provider = FundamentalDataProviderFactory.create_provider(prefer_simfin=False)

                    # Should have only Yahoo provider
                    assert len(provider.providers) == 1
                    mock_simfin.assert_not_called()
                    mock_yahoo.assert_called_once()


class TestUnifiedFundamentalProvider:
    """Test the unified provider."""

    def test_unified_provider_fallback(self):
        """Test provider fallback on failure."""
        failing_provider = Mock(spec=YahooFinanceProvider)
        failing_provider.get_fundamental_data.side_effect = Exception("API Error")
        failing_provider.supports_historical_data.return_value = True

        working_provider = Mock(spec=YahooFinanceProvider)
        working_provider.get_fundamental_data.return_value = [
            FundamentalData(ticker="AAPL", market_cap=3e12)
        ]
        working_provider.supports_historical_data.return_value = False

        provider = UnifiedFundamentalProvider([failing_provider, working_provider])

        result = provider.get_fundamental_data(["AAPL"])

        assert len(result) == 1
        assert result[0].ticker == "AAPL"
        failing_provider.get_fundamental_data.assert_called_once_with(["AAPL"], None)
        working_provider.get_fundamental_data.assert_called_once_with(["AAPL"], None)

    def test_unified_provider_cache_hit(self):
        """Test cache retrieval."""
        mock_provider = Mock(spec=YahooFinanceProvider)
        provider = UnifiedFundamentalProvider([mock_provider], enable_caching=True)

        from datetime import datetime
        date = datetime(2023, 1, 1)

        # Store data
        data = [FundamentalData(ticker="AAPL", market_cap=3e12)]
        provider.cache.store_data(data, date)

        # Retrieve (should not call provider)
        result = provider.get_fundamental_data(["AAPL"], date)

        assert len(result) == 1
        assert result[0].ticker == "AAPL"
        mock_provider.get_fundamental_data.assert_not_called()

    def test_unified_provider_cache_miss(self):
        """Test cache miss calls provider."""
        mock_provider = Mock(spec=YahooFinanceProvider)
        mock_provider.get_fundamental_data.return_value = [
            FundamentalData(ticker="AAPL", market_cap=3e12)
        ]
        mock_provider.supports_historical_data.return_value = True

        provider = UnifiedFundamentalProvider([mock_provider], enable_caching=True)

        from datetime import datetime
        date = datetime(2023, 1, 1)

        # Retrieve (cache empty, should call provider)
        result = provider.get_fundamental_data(["AAPL"], date)

        assert len(result) == 1
        mock_provider.get_fundamental_data.assert_called_once_with(["AAPL"], date)

    def test_unified_provider_historical_skip(self):
        """Test skipping non-historical providers."""
        yahoo_provider = Mock(spec=YahooFinanceProvider)
        yahoo_provider.supports_historical_data.return_value = False

        provider = UnifiedFundamentalProvider([yahoo_provider])

        from datetime import datetime
        date = datetime(2020, 1, 1)

        # Yahoo doesn't support historical, should return empty data
        result = provider.get_fundamental_data(["AAPL"], date)

        # Should return empty data since provider skipped
        assert len(result) == 1
        assert result[0].ticker == "AAPL"
        assert not result[0].is_valid
        yahoo_provider.get_fundamental_data.assert_not_called()

    def test_unified_provider_all_fail(self):
        """Test all providers fail returns empty data."""
        failing_provider = Mock(spec=YahooFinanceProvider)
        failing_provider.get_fundamental_data.side_effect = Exception("API Error")
        failing_provider.supports_historical_data.return_value = True

        provider = UnifiedFundamentalProvider([failing_provider])

        result = provider.get_fundamental_data(["AAPL"])

        assert len(result) == 1
        assert result[0].ticker == "AAPL"
        assert not result[0].is_valid

    def test_unified_provider_preload_without_cache(self):
        """Test preload warning when caching disabled."""
        mock_provider = Mock(spec=YahooFinanceProvider)
        provider = UnifiedFundamentalProvider([mock_provider], enable_caching=False)

        from datetime import datetime
        start = datetime(2020, 1, 1)
        end = datetime(2023, 1, 1)

        # Should not crash, just warning
        provider.preload_data(["AAPL"], start, end)

        # Provider should not be called
        mock_provider.get_fundamental_data.assert_not_called()

    def test_unified_provider_supports_historical(self):
        """Test historical support check."""
        historical_provider = Mock(spec=YahooFinanceProvider)
        historical_provider.supports_historical_data.return_value = True

        non_historical_provider = Mock(spec=YahooFinanceProvider)
        non_historical_provider.supports_historical_data.return_value = False

        # Test with historical support
        provider1 = UnifiedFundamentalProvider([historical_provider])
        assert provider1.supports_historical_data()

        # Test without historical support
        provider2 = UnifiedFundamentalProvider([non_historical_provider])
        assert not provider2.supports_historical_data()

        # Test mixed (any true = true)
        provider3 = UnifiedFundamentalProvider([non_historical_provider, historical_provider])
        assert provider3.supports_historical_data()