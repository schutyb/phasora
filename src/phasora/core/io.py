from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from numpy.typing import NDArray


SUPPORTED_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".lsm",
    ".czi",
}

TIFF_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".lsm",
}


def load_image(
    file_path: str | Path,
) -> NDArray[np.float64]:
    """
    Load a supported microscopy image and return it as YXC.

    The returned array has:
        axis 0: Y
        axis 1: X
        axis 2: channels, temporal bins, or spectral bins
    """
    path = Path(file_path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Image path is not a file: {path}")

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported image format '{extension}'. Supported formats: {supported}")

    if extension in TIFF_EXTENSIONS:
        image = _load_tiff_like(path)
        return _prepare_image(image)

    if extension == ".czi":
        # _load_czi already returns YXC.
        image = _load_czi(path)
        return _validate_channel_last_image(image)

    raise ValueError(f"Unsupported image format: {extension}")


def load_tiff(
    file_path: str | Path,
) -> NDArray[np.float64]:
    """
    Load a TIFF, BigTIFF, or Zeiss LSM image.

    This function is retained for backward compatibility.
    """
    path = Path(file_path).expanduser()

    if path.suffix.lower() not in TIFF_EXTENSIONS:
        raise ValueError(
            "load_tiff only supports .tif, .tiff, and .lsm files. "
            "Use load_image for other supported formats."
        )

    return load_image(path)


def _load_tiff_like(
    path: Path,
) -> NDArray[np.generic]:
    """
    Read TIFF-compatible formats, including Zeiss LSM.
    """
    try:
        return np.asarray(tifffile.imread(path))
    except (tifffile.TiffFileError, OSError) as error:
        raise ValueError(f"Could not read TIFF/LSM image: {path.name}") from error


def _load_czi(
    path: Path,
) -> NDArray[np.float64]:
    """
    Read the first scene, time point, and Z plane of a CZI file.

    The returned array is ordered as YXC.
    """
    try:
        from aicspylibczi import CziFile
    except ImportError as error:
        raise ImportError(
            "CZI support requires aicspylibczi. Install it with: python -m pip install aicspylibczi"
        ) from error

    try:
        czi = CziFile(path)
    except (RuntimeError, ValueError, OSError) as error:
        raise ValueError(f"Could not open CZI file: {path.name}") from error

    dimension_selection: dict[str, int] = {}

    # Select the first available block, scene, time point,
    # and Z plane for the current basic reader.
    for dimension in ("B", "S", "T", "Z"):
        if dimension in czi.dims:
            dimension_selection[dimension] = 0

    try:
        data, dimension_info = czi.read_image(**dimension_selection)
    except (RuntimeError, ValueError, IndexError) as error:
        raise ValueError(f"Could not read CZI image: {path.name}") from error

    array = np.asarray(data)

    dimension_names = [dimension_name for dimension_name, _size in dimension_info]

    if array.ndim != len(dimension_names):
        raise ValueError(
            "CZI dimension metadata does not match the returned "
            f"array. Dimensions: {dimension_names}; "
            f"shape: {array.shape}."
        )

    # Remove any remaining non-CYX singleton dimensions while
    # keeping dimension names aligned with array axes.
    for axis in reversed(range(array.ndim)):
        dimension_name = dimension_names[axis]

        if dimension_name in {"C", "Y", "X"}:
            continue

        if array.shape[axis] != 1:
            raise ValueError(
                "The selected CZI image contains an unsupported "
                f"dimension '{dimension_name}' with size "
                f"{array.shape[axis]}. Current Phasora support "
                "requires a single scene, time point, Z plane, "
                "and non-mosaic image."
            )

        array = np.take(
            array,
            indices=0,
            axis=axis,
        )
        dimension_names.pop(axis)

    missing_dimensions = {"C", "Y", "X"} - set(dimension_names)

    if missing_dimensions:
        missing = ", ".join(sorted(missing_dimensions))
        raise ValueError(
            "The selected CZI plane does not contain all required "
            f"dimensions. Missing: {missing}. "
            f"Received dimensions: {dimension_names}; "
            f"shape: {array.shape}."
        )

    if len(dimension_names) != 3:
        raise ValueError(
            "Expected exactly C, Y, and X dimensions after "
            f"selection, but received {dimension_names} "
            f"with shape {array.shape}."
        )

    channel_axis = dimension_names.index("C")
    y_axis = dimension_names.index("Y")
    x_axis = dimension_names.index("X")

    array = np.transpose(
        array,
        axes=(y_axis, x_axis, channel_axis),
    )

    return _validate_channel_last_image(array)


def _prepare_image(
    image: NDArray[np.generic],
) -> NDArray[np.float64]:
    """
    Convert a generic three-dimensional image to YXC order.

    For TIFF and LSM files without explicit axis handling, the
    smallest dimension is treated as the channel/bin axis.
    """
    array = np.asarray(image)
    array = np.squeeze(array)

    if array.ndim != 3:
        raise ValueError(
            "Expected a three-dimensional multichannel image after "
            "removing singleton dimensions, but received shape "
            f"{array.shape}."
        )

    array = _move_channel_axis_last(array)

    return _validate_channel_last_image(array)


def _validate_channel_last_image(
    image: NDArray[np.generic],
) -> NDArray[np.float64]:
    """
    Validate and convert a YXC image to float64.
    """
    array = np.asarray(image)

    if array.ndim != 3:
        raise ValueError(
            f"Expected a three-dimensional YXC image, but received shape {array.shape}."
        )

    if array.shape[-1] < 2:
        raise ValueError(
            f"Expected at least two channels or bins, but received shape {array.shape}."
        )

    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"Image has invalid spatial dimensions: {array.shape}.")

    return np.asarray(
        array,
        dtype=np.float64,
    )


def _move_channel_axis_last(
    image: NDArray[np.generic],
) -> NDArray[np.generic]:
    """
    Move the most likely channel/bin axis to the final position.

    The current TIFF/LSM workflow assumes that the channel or bin
    dimension is smaller than both spatial dimensions.
    """
    if image.ndim != 3:
        raise ValueError(
            f"_move_channel_axis_last requires a 3D image, but received shape {image.shape}."
        )

    shape = image.shape
    minimum_size = min(shape)

    candidate_axes = [axis for axis, size in enumerate(shape) if size == minimum_size]

    if len(candidate_axes) != 1:
        raise ValueError(
            "Could not determine the channel/bin axis uniquely from "
            f"image shape {shape}. The smallest dimension is not "
            "unique."
        )

    channel_axis = candidate_axes[0]

    return np.moveaxis(
        image,
        channel_axis,
        -1,
    )
