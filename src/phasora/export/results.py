from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from numpy.typing import NDArray


@dataclass(frozen=True)
class SelectionResult:
    index: int
    center_g: float
    center_s: float
    radius: float
    mask: NDArray[np.bool_]


def export_selections(
    output_directory: str | Path,
    selections: list[SelectionResult],
) -> None:
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    if not selections:
        raise ValueError("At least one selection is required")

    reference_shape = selections[0].mask.shape

    label_image = np.zeros(reference_shape, dtype=np.uint16)

    csv_path = output_path / "selections.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "selection",
                "center_g",
                "center_s",
                "radius",
                "selected_pixels",
            ]
        )

        for selection in selections:
            if selection.mask.shape != reference_shape:
                raise ValueError("All selection masks must have the same shape")

            mask_uint8 = selection.mask.astype(np.uint8)

            tifffile.imwrite(
                output_path / f"selection_{selection.index:02d}_mask.tiff",
                mask_uint8,
            )

            label_image[selection.mask] = selection.index

            writer.writerow(
                [
                    selection.index,
                    selection.center_g,
                    selection.center_s,
                    selection.radius,
                    int(np.count_nonzero(selection.mask)),
                ]
            )

    tifffile.imwrite(
        output_path / "selection_labels.tiff",
        label_image,
    )
