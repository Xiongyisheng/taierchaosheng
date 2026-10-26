"""Check the extracted dataset against the original patient-level split."""

from __future__ import annotations

import csv
from pathlib import Path, PureWindowsPath


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "fetal_ultrasound"


def read_split(name: str) -> list[dict[str, str]]:
    with (ROOT / "split" / f"{name}.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    splits = {name: read_split(name) for name in ("train", "val", "test")}
    all_rows = [row for rows in splits.values() for row in rows]
    paths = [row["filepath"] for row in all_rows]
    missing = 0
    for original in paths:
        source = PureWindowsPath(original)
        if not (DATA_ROOT / source.parent.name / source.name).is_file():
            missing += 1

    overlaps = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        patients_left = {row["patient_id"] for row in splits[left]}
        patients_right = {row["patient_id"] for row in splits[right]}
        overlaps[f"{left}-{right}"] = len(patients_left & patients_right)

    print("Rows:", {name: len(rows) for name, rows in splits.items()})
    print("Unique image paths:", len(set(paths)), "/", len(paths))
    print("Missing images:", missing)
    print("Patients shared between splits:", overlaps)

    good = missing == 0 and len(set(paths)) == len(paths) and not any(overlaps.values())
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
