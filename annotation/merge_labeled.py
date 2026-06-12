import argparse
import json
import sys
from pathlib import Path
from typing import Any


def normalize_input_dir(path: Path) -> Path:
    labeled_child = path / "labeled"
    if labeled_child.exists() and labeled_child.is_dir():
        return labeled_child
    return path


def record_key(record: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(record["source_file"]),
        int(record["source_row"]),
        str(record["comment_id"]),
    )


def output_relative_path(record: dict[str, Any], fallback: Path) -> Path:
    source_file = record.get("source_file")
    if isinstance(source_file, str) and source_file:
        return Path(source_file).with_suffix(".labels.jsonl")
    return fallback


def load_records(input_dirs: list[Path]) -> tuple[dict[Path, dict[tuple[str, int, str], dict[str, Any]]], dict[str, int]]:
    grouped: dict[Path, dict[tuple[str, int, str], dict[str, Any]]] = {}
    stats = {
        "files_read": 0,
        "records_read": 0,
        "records_kept": 0,
        "duplicates": 0,
        "conflicts": 0,
        "invalid": 0,
    }

    for raw_input_dir in input_dirs:
        input_dir = normalize_input_dir(raw_input_dir)
        if not input_dir.exists():
            print(f"Skipping missing input folder: {input_dir}", file=sys.stderr)
            continue

        for labels_file in sorted(input_dir.rglob("*.labels.jsonl")):
            stats["files_read"] += 1
            fallback = labels_file.relative_to(input_dir)

            with labels_file.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    line = line.strip()
                    if not line:
                        continue

                    stats["records_read"] += 1
                    try:
                        record = json.loads(line)
                        key = record_key(record)
                        relative_output = output_relative_path(record, fallback)
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                        stats["invalid"] += 1
                        print(
                            f"Skipping invalid record in {labels_file}:{line_number}: {error}",
                            file=sys.stderr,
                        )
                        continue

                    records_for_file = grouped.setdefault(relative_output, {})
                    existing = records_for_file.get(key)
                    if existing is not None:
                        stats["duplicates"] += 1
                        if existing.get("label") != record.get("label"):
                            stats["conflicts"] += 1
                        continue

                    records_for_file[key] = record
                    stats["records_kept"] += 1

    return grouped, stats


def write_merged(
    grouped: dict[Path, dict[tuple[str, int, str], dict[str, Any]]],
    output_dir: Path,
) -> int:
    files_written = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    for relative_output, records_by_key in sorted(grouped.items(), key=lambda item: item[0].as_posix()):
        output_path = output_dir / relative_output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        records = sorted(records_by_key.values(), key=lambda record: int(record["source_row"]))

        with output_path.open("w", encoding="utf-8", newline="\n") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

        files_written += 1

    return files_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge labeled JSONL outputs from multiple machines and dedupe by source_file/source_row/comment_id."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Folders containing *.labels.jsonl files, or project folders containing a labeled/ subfolder.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("merged_labeled"),
        help="Output folder for merged *.labels.jsonl files. Default: merged_labeled",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped, stats = load_records(args.inputs)
    files_written = write_merged(grouped, args.output)

    print(f"Merged label folders into: {args.output}")
    print(f"files_read={stats['files_read']}, files_written={files_written}")
    print(
        f"records_read={stats['records_read']}, records_kept={stats['records_kept']}, "
        f"duplicates={stats['duplicates']}, conflicts={stats['conflicts']}, invalid={stats['invalid']}"
    )


if __name__ == "__main__":
    main()
