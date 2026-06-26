from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


SESSION_VERSION = 1


@dataclass(frozen=True)
class CursorState:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class ViewRangeState:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass(frozen=True)
class PhasoraSession:
    image_path: str
    mode: str
    histogram_bins: int
    colormap: str
    filter_name: str
    filter_size: int
    filter_repetitions: int
    minimum_threshold: float | None
    maximum_threshold: float | None
    cursors: list[CursorState] = field(default_factory=list)
    view_range: ViewRangeState | None = None
    version: int = SESSION_VERSION

    def save(self, path: str | Path) -> None:
        output_path = Path(path)

        if output_path.suffix.lower() != ".json":
            raise ValueError("Phasora sessions must use the .json extension.")

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as json_file:
            json.dump(
                asdict(self),
                json_file,
                indent=2,
                sort_keys=True,
            )

    @classmethod
    def load(cls, path: str | Path) -> "PhasoraSession":
        input_path = Path(path)

        with input_path.open(
            "r",
            encoding="utf-8",
        ) as json_file:
            data = json.load(json_file)

        if not isinstance(data, dict):
            raise ValueError("Invalid Phasora session file.")

        version = int(data.get("version", 0))
        if version != SESSION_VERSION:
            raise ValueError(
                f"Unsupported session version {version}. Expected version {SESSION_VERSION}."
            )

        image_path = data.get("image_path")
        if not isinstance(image_path, str) or not image_path:
            raise ValueError("Session does not contain a valid image path.")

        cursor_data = data.get("cursors", [])
        if not isinstance(cursor_data, list):
            raise ValueError("Session cursors must be a list.")

        cursors = [
            CursorState(
                x=float(cursor["x"]),
                y=float(cursor["y"]),
                width=float(cursor["width"]),
                height=float(cursor["height"]),
            )
            for cursor in cursor_data
        ]

        view_range_data: Any = data.get("view_range")
        view_range = None

        if view_range_data is not None:
            if not isinstance(view_range_data, dict):
                raise ValueError("Session view range must be an object.")

            view_range = ViewRangeState(
                x_min=float(view_range_data["x_min"]),
                x_max=float(view_range_data["x_max"]),
                y_min=float(view_range_data["y_min"]),
                y_max=float(view_range_data["y_max"]),
            )

        return cls(
            image_path=image_path,
            mode=str(data["mode"]),
            histogram_bins=int(data["histogram_bins"]),
            colormap=str(data["colormap"]),
            filter_name=str(data["filter_name"]),
            filter_size=int(data["filter_size"]),
            filter_repetitions=int(data["filter_repetitions"]),
            minimum_threshold=(
                None if data.get("minimum_threshold") is None else float(data["minimum_threshold"])
            ),
            maximum_threshold=(
                None if data.get("maximum_threshold") is None else float(data["maximum_threshold"])
            ),
            cursors=cursors,
            view_range=view_range,
            version=version,
        )
