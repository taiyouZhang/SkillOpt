"""Storyboard data loader — items already embed script_text and ground_truth_csv."""
from __future__ import annotations

import json
from pathlib import Path

from skillopt.datasets.base import SplitDataLoader


class StoryboardDataLoader(SplitDataLoader):
    """Load storyboard items from pre-split directories.

    Each items.json contains objects with: id, task_type, script_text, ground_truth_csv.
    """

    def load_split_items(self, split_path: str) -> list[dict]:
        path = Path(split_path)
        json_files = sorted(path.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No .json file found in {split_path}")
        with json_files[0].open(encoding="utf-8") as f:
            items = json.load(f)
        if not isinstance(items, list):
            raise ValueError(f"Expected JSON array in {json_files[0]}")
        return items
