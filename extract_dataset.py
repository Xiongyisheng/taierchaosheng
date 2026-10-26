"""Safely extract and verify the archived 10-class image dataset."""

from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "dataset" / "fetal_ultrasound.tar"
DESTINATION = ROOT / "fetal_ultrasound"
EXPECTED_IMAGES = 6137


def digest_file(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.digest()


def main() -> int:
    with tarfile.open(ARCHIVE, "r:") as archive:
        entries = []
        seen = set()
        for member in archive.getmembers():
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe archive path")
            if not (member.isdir() or member.isfile()):
                raise ValueError("Unexpected archive member type")
            if member.isfile():
                folded = str(relative).casefold()
                if folded in seen:
                    raise ValueError("Duplicate image path on Windows")
                seen.add(folded)
                entries.append((member, relative))

        if len(entries) != EXPECTED_IMAGES:
            raise ValueError(f"Expected {EXPECTED_IMAGES} images, found {len(entries)}")

        DESTINATION.mkdir(exist_ok=True)
        total_bytes = 0
        for member, relative in entries:
            target = DESTINATION.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("Could not read archived image")
            with source, target.open("wb") as output:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    output.write(block)
                    digest.update(block)
                    total_bytes += len(block)
            if digest_file(target) != digest.digest():
                raise ValueError("Extracted image failed checksum")

    actual = sum(path.is_file() for path in DESTINATION.rglob("*"))
    if actual != EXPECTED_IMAGES:
        raise ValueError(f"Extracted file count {actual} != {EXPECTED_IMAGES}")
    print(f"Extracted and verified {actual} images ({total_bytes} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
