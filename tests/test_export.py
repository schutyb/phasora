import csv

import numpy as np
import tifffile

from phasora.export.results import SelectionResult, export_selections


def test_export_selections(tmp_path) -> None:
    first_mask = np.array(
        [
            [True, False],
            [False, False],
        ]
    )

    second_mask = np.array(
        [
            [False, True],
            [False, False],
        ]
    )

    selections = [
        SelectionResult(
            index=1,
            center_g=0.1,
            center_s=0.2,
            radius=0.05,
            mask=first_mask,
        ),
        SelectionResult(
            index=2,
            center_g=0.3,
            center_s=0.4,
            radius=0.08,
            mask=second_mask,
        ),
    ]

    export_selections(tmp_path, selections)

    assert (tmp_path / "selection_01_mask.tiff").exists()
    assert (tmp_path / "selection_02_mask.tiff").exists()
    assert (tmp_path / "selection_labels.tiff").exists()
    assert (tmp_path / "selections.csv").exists()

    labels = tifffile.imread(tmp_path / "selection_labels.tiff")

    expected_labels = np.array(
        [
            [1, 2],
            [0, 0],
        ],
        dtype=np.uint16,
    )

    np.testing.assert_array_equal(labels, expected_labels)


def test_export_csv_contains_selection_data(tmp_path) -> None:
    mask = np.array(
        [
            [True, True],
            [False, False],
        ]
    )

    selection = SelectionResult(
        index=1,
        center_g=0.15,
        center_s=0.25,
        radius=0.1,
        mask=mask,
    )

    export_selections(tmp_path, [selection])

    with (tmp_path / "selections.csv").open(
        newline="",
        encoding="utf-8",
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert len(rows) == 1
    assert rows[0]["selection"] == "1"
    assert rows[0]["selected_pixels"] == "2"