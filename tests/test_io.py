import numpy as np
import pytest
import tifffile

from phasora.core.io import load_tiff


def test_load_channel_last_tiff(tmp_path) -> None:
    image = np.random.default_rng(42).random((10, 12, 8))
    path = tmp_path / "channel_last.tiff"

    tifffile.imwrite(path, image)

    loaded = load_tiff(path)

    assert loaded.shape == (10, 12, 8)
    assert loaded.dtype == np.float64
    np.testing.assert_allclose(loaded, image)


def test_load_channel_first_tiff(tmp_path) -> None:
    image = np.random.default_rng(42).random((8, 10, 12))
    path = tmp_path / "channel_first.tiff"

    tifffile.imwrite(path, image)

    loaded = load_tiff(path)

    assert loaded.shape == (10, 12, 8)
    assert loaded.dtype == np.float64


def test_missing_file_raises_error(tmp_path) -> None:
    path = tmp_path / "missing.tiff"

    with pytest.raises(FileNotFoundError):
        load_tiff(path)


def test_invalid_extension_raises_error(tmp_path) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"not an image")

    with pytest.raises(ValueError, match="tif"):
        load_tiff(path)