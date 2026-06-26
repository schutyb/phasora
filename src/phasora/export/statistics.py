from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CursorStatistics:
    cursor_index: int
    center_g: float
    center_s: float
    radius: float
    selected_pixels: int
    selected_percentage: float
    mean_g: float
    std_g: float
    mean_s: float
    std_s: float
    mean_dc: float
    std_dc: float
    min_dc: float
    max_dc: float

    @classmethod
    def from_arrays(
        cls,
        *,
        cursor_index: int,
        center_g: float,
        center_s: float,
        radius: float,
        selected_pixels: int,
        selected_percentage: float,
        g: np.ndarray,
        s: np.ndarray,
        dc: np.ndarray,
    ) -> "CursorStatistics":
        g_values = _finite_values(g)
        s_values = _finite_values(s)
        dc_values = _finite_values(dc)

        return cls(
            cursor_index=int(cursor_index),
            center_g=float(center_g),
            center_s=float(center_s),
            radius=float(radius),
            selected_pixels=int(selected_pixels),
            selected_percentage=float(selected_percentage),
            mean_g=_mean_or_nan(g_values),
            std_g=_std_or_nan(g_values),
            mean_s=_mean_or_nan(s_values),
            std_s=_std_or_nan(s_values),
            mean_dc=_mean_or_nan(dc_values),
            std_dc=_std_or_nan(dc_values),
            min_dc=_min_or_nan(dc_values),
            max_dc=_max_or_nan(dc_values),
        )


def export_cursor_statistics(
    *,
    output_path: str | Path,
    statistics: Iterable[CursorStatistics],
) -> None:
    path = Path(output_path)
    rows = list(statistics)

    if path.suffix.lower() != ".csv":
        raise ValueError("Cursor statistics must be exported as CSV.")

    if not rows:
        raise ValueError("No cursor statistics were provided.")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(asdict(rows[0]).keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow({key: _format_value(value) for key, value in asdict(row).items()})


def _finite_values(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _mean_or_nan(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def _std_or_nan(values: np.ndarray) -> float:
    return float(np.std(values)) if values.size else float("nan")


def _min_or_nan(values: np.ndarray) -> float:
    return float(np.min(values)) if values.size else float("nan")


def _max_or_nan(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else float("nan")


def _format_value(value: object) -> object:
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        return f"{value:.10g}"

    return value
