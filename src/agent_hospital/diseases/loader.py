"""Pluggable dataset loading.

`DatasetLoader` handles the file mechanics (read a `.json` array or a `.jsonl`
file, assign case ids). Each dataset only implements `parse_record`, mapping its
own JSON schema into a canonical case type. To support a new JSON dataset,
subclass `DatasetLoader` and implement `parse_record` — nothing else changes.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

T = TypeVar("T")


class DatasetLoader(ABC, Generic[T]):
    """Load cases from a JSON/JSONL file into a canonical case type `T`."""

    @abstractmethod
    def parse_record(self, record: dict, case_id: str) -> T:
        """Map one raw JSON record into a canonical case."""

    def load(self, path: str | Path) -> list[T]:
        path = Path(path)
        return [
            self.parse_record(record, case_id=f"{path.stem}-{i:04d}")
            for i, record in enumerate(self._read(path))
        ]

    @staticmethod
    def _read(path: Path) -> list[dict]:
        """Read records from `.jsonl` (one object per line) or `.json` (array or object)."""
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
