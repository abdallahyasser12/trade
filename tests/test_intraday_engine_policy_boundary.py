import pytest

from intraday_crypto_engine import _strict_intraday_context
from quant.intraday_lifecycle import intraday_promotion_requirements
from quant.lifecycle import LifecycleManager
from quant.store import QuantStore


def test_default_direct_intraday_context_uses_strict_high_velocity_requirements(tmp_path):
    store = QuantStore(str(tmp_path / "quant.db"))
    try:
        resolved_store, lifecycle = _strict_intraday_context(store, None)
        assert resolved_store is store
        assert lifecycle.requirements == intraday_promotion_requirements()
    finally:
        store.close()


def test_generic_weaker_lifecycle_is_rejected_at_engine_boundary(tmp_path):
    store = QuantStore(str(tmp_path / "quant.db"))
    try:
        weaker = LifecycleManager(store)
        with pytest.raises(ValueError, match="weaker than the production policy"):
            _strict_intraday_context(store, weaker)
    finally:
        store.close()


def test_lifecycle_and_quant_store_must_be_same_instance(tmp_path):
    first = QuantStore(str(tmp_path / "first.db"))
    second = QuantStore(str(tmp_path / "second.db"))
    try:
        lifecycle = LifecycleManager(
            first,
            requirements=intraday_promotion_requirements(),
        )
        with pytest.raises(ValueError, match="same store instance"):
            _strict_intraday_context(second, lifecycle)
    finally:
        first.close()
        second.close()
