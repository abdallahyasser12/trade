from config import AccountMode
from crypto_config import CryptoEngineConfig


def test_normal_live_crypto_config_is_rejected_outside_quant_runtime():
    errors = CryptoEngineConfig(account_mode=AccountMode.LIVE).validate()
    assert any("Quant-controlled lifecycle runtime" in error for error in errors)


def test_paper_config_remains_normally_valid():
    assert CryptoEngineConfig(account_mode=AccountMode.PAPER).validate() == []
