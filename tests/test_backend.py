import pytest

from harbichess.core.backend import EncodedPosition


def test_encoded_position_validates_shape() -> None:
    position = EncodedPosition(values=(0.0,) * 128, shape=(2, 8, 8), schema_version=1)
    assert position.shape == (2, 8, 8)


def test_encoded_position_rejects_mismatched_values() -> None:
    with pytest.raises(ValueError, match="shape requires 64 values"):
        EncodedPosition(values=(0.0,), shape=(8, 8), schema_version=1)

