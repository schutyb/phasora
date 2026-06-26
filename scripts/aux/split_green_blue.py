from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile


EXPECTED_INPUT_CHANNELS = 31
OUTPUT_CHANNELS = 16


def detect_channel_axis(image: np.ndarray) -> int:
    """
    Detect the channel axis by finding the unique dimension of length 31.
    """
    matching_axes = [
        axis for axis, size in enumerate(image.shape) if size == EXPECTED_INPUT_CHANNELS
    ]

    if len(matching_axes) != 1:
        raise ValueError(
            "Could not determine the channel axis automatically. "
            f"Expected exactly one axis of length {EXPECTED_INPUT_CHANNELS}, "
            f"but received shape {image.shape}."
        )

    return matching_axes[0]


def split_green_blue(
    image: np.ndarray,
    channel_axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a 31-bin stack into corrected 16-bin green and blue stacks.

    Green:
        Input bins 0-15. The final bin is replaced by a copy of bin 14.

    Blue:
        Input bins 16-30, giving 15 bins. A sixteenth bin is appended
        as a copy of the final available blue bin.
    """
    image = np.asarray(image)

    if image.ndim < 2:
        raise ValueError(f"Expected a multidimensional TIFF stack, received shape {image.shape}.")

    channel_axis = np.core.numeric.normalize_axis_index(
        channel_axis,
        image.ndim,
    )

    if image.shape[channel_axis] != EXPECTED_INPUT_CHANNELS:
        raise ValueError(
            f"Expected {EXPECTED_INPUT_CHANNELS} channels along axis "
            f"{channel_axis}, but received shape {image.shape}."
        )

    # Work internally with channels on the last axis.
    channel_last = np.moveaxis(image, channel_axis, -1)

    green = channel_last[..., :16].copy()

    # The recorded final green bin is invalid/missing.
    # Replace it with the preceding valid bin.
    green[..., -1] = green[..., -2]

    blue_recorded = channel_last[..., 16:].copy()

    if blue_recorded.shape[-1] != 15:
        raise ValueError(
            "The blue portion was expected to contain 15 recorded bins, "
            f"but contains {blue_recorded.shape[-1]}."
        )

    # Append the missing sixteenth blue bin by copying the preceding bin.
    blue = np.concatenate(
        [
            blue_recorded,
            blue_recorded[..., -1:],
        ],
        axis=-1,
    )

    if green.shape[-1] != OUTPUT_CHANNELS:
        raise RuntimeError(f"Green output has unexpected shape {green.shape}.")

    if blue.shape[-1] != OUTPUT_CHANNELS:
        raise RuntimeError(f"Blue output has unexpected shape {blue.shape}.")

    # Restore the original axis order.
    green = np.moveaxis(green, -1, channel_axis)
    blue = np.moveaxis(blue, -1, channel_axis)

    return green, blue


def build_output_paths(
    input_path: Path,
    output_directory: Path,
) -> tuple[Path, Path]:
    stem = input_path.stem

    green_path = output_directory / f"{stem}_green_16bins.tiff"
    blue_path = output_directory / f"{stem}_blue_16bins.tiff"

    return green_path, blue_path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Split a 31-bin TIFF into corrected green and blue 16-bin TIFF stacks.")
    )

    parser.add_argument(
        "input_tiff",
        type=Path,
        help="Path to the original 31-bin TIFF.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=("Directory for the output TIFFs. Defaults to the input file's directory."),
    )

    parser.add_argument(
        "--channel-axis",
        type=int,
        default=None,
        help=("Channel-axis index. If omitted, the script detects the unique axis of length 31."),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    input_path = args.input_tiff.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input TIFF does not exist: {input_path}")

    if input_path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("The input file must have a .tif or .tiff extension.")

    output_directory = (
        args.output_dir.expanduser().resolve() if args.output_dir is not None else input_path.parent
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    image = tifffile.imread(input_path)

    channel_axis = (
        args.channel_axis if args.channel_axis is not None else detect_channel_axis(image)
    )

    green, blue = split_green_blue(
        image=image,
        channel_axis=channel_axis,
    )

    green_path, blue_path = build_output_paths(
        input_path=input_path,
        output_directory=output_directory,
    )

    tifffile.imwrite(
        green_path,
        green,
        photometric="minisblack",
    )
    tifffile.imwrite(
        blue_path,
        blue,
        photometric="minisblack",
    )

    print(f"Input shape:  {image.shape}")
    print(f"Channel axis: {channel_axis}")
    print(f"Green shape:  {green.shape}")
    print(f"Blue shape:   {blue.shape}")
    print(f"Green saved:  {green_path}")
    print(f"Blue saved:   {blue_path}")


if __name__ == "__main__":
    main()
