"""Test CashConfig dataclass and AllowCashOption enum."""
from allooptim.config.cash_config import CashConfig, AllowCashOption


def test_cash_config_defaults():
    """Test CashConfig default values."""
    config = CashConfig()
    assert config.allow_cash_option == AllowCashOption.OPTIMIZER_DECIDES
    assert config.max_leverage is None


def test_cash_config_global_allow():
    """Test CashConfig with GLOBAL_ALLOW_CASH option."""
    config = CashConfig(
        allow_cash_option=AllowCashOption.GLOBAL_ALLOW_CASH,
        max_leverage=1.5
    )
    assert config.allow_cash_option == AllowCashOption.GLOBAL_ALLOW_CASH
    assert config.max_leverage == 1.5


def test_cash_config_global_forbid():
    """Test CashConfig with GLOBAL_FORBID_CASH option."""
    config = CashConfig(allow_cash_option=AllowCashOption.GLOBAL_FORBID_CASH)
    assert config.allow_cash_option == AllowCashOption.GLOBAL_FORBID_CASH


def test_cash_config_optimizer_decides():
    """Test CashConfig with OPTIMIZER_DECIDES option."""
    config = CashConfig(allow_cash_option=AllowCashOption.OPTIMIZER_DECIDES)
    assert config.allow_cash_option == AllowCashOption.OPTIMIZER_DECIDES


def test_allow_cash_option_enum_values():
    """Test AllowCashOption enum values."""
    assert AllowCashOption.GLOBAL_ALLOW_CASH.value == "global_allow_cash"
    assert AllowCashOption.OPTIMIZER_DECIDES.value == "optimizer_decides"
    assert AllowCashOption.GLOBAL_FORBID_CASH.value == "global_forbid_cash"