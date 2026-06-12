import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


ROOT = Path(__file__).resolve().parent
LABELED_DIR = ROOT / "labeled"
CRAWLED_DIR = ROOT / "youtube_comments_v2"
CUTOFF = datetime(2026, 5, 3, 7, 30, tzinfo=timezone.utc)
DEFAULT_SAMPLE_SIZE = 190
DEFAULT_SEED = 20260503
OUTPUT_COLUMNS = [
    "source_file",
    "source_row",
    "comment_id",
    "post_id",
    "comment_text",
    "title_youtube",
    "source_query",
    "labeled_at",
    "temperature_group",
]


TemperatureGroup = Literal["1.0", "0.0"]


@dataclass(frozen=True)
class SampleRecord:
    source_file: str
    source_row: int
    comment_id: str
    post_id: str
    comment_text: str
    title_youtube: str
    source_query: str
    temperature_group: TemperatureGroup
    labeled_at: str = ""

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.source_file, self.source_row, self.comment_id)


def parse_labeled_at(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def record_temperature_group(labeled_at: str) -> TemperatureGroup:
    if parse_labeled_at(labeled_at) < CUTOFF:
        return "1.0"
    return "0.0"


def read_labeled_records(target_group: TemperatureGroup) -> list[SampleRecord]:
    records: dict[tuple[str, int, str], SampleRecord] = {}

    if not LABELED_DIR.exists():
        raise FileNotFoundError(f"Labeled folder not found: {LABELED_DIR}")

    for labels_file in sorted(LABELED_DIR.rglob("*.labels.jsonl")):
        with labels_file.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    raw = json.loads(line)
                    group = record_temperature_group(str(raw["labeled_at"]))
                    if group != target_group:
                        continue

                    record = SampleRecord(
                        source_file=str(raw["source_file"]),
                        source_row=int(raw["source_row"]),
                        comment_id=str(raw["comment_id"]),
                        post_id=str(raw["post_id"]),
                        comment_text="",
                        title_youtube="",
                        source_query="",
                        temperature_group=group,
                        labeled_at=str(raw["labeled_at"]),
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    print(
                        f"Skipping invalid labeled record {labels_file}:{line_number}: {error}",
                        file=sys.stderr,
                    )
                    continue

                records.setdefault(record.key, record)

    return list(records.values())


def read_existing_sample_keys(path: Path) -> set[tuple[str, int, str]]:
    keys: set[tuple[str, int, str]] = set()
    if not path.exists():
        return keys
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            key = (row.get("source_file", ""), int(row.get("source_row", 0)), row.get("comment_id", ""))
            keys.add(key)
    return keys


def read_remaining_crawled_records(excluded_keys: set[tuple[str, int, str]]) -> list[SampleRecord]:
    records: list[SampleRecord] = []

    if not CRAWLED_DIR.exists():
        raise FileNotFoundError(f"Crawled folder not found: {CRAWLED_DIR}")

    for csv_path in sorted(CRAWLED_DIR.rglob("*.csv")):
        source_file = csv_path.relative_to(CRAWLED_DIR).as_posix()

        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for source_row, row in enumerate(reader, start=1):
                comment_id = row.get("comment_id", "")
                key = (source_file, source_row, comment_id)
                if key in excluded_keys:
                    continue

                records.append(
                    SampleRecord(
                        source_file=source_file,
                        source_row=source_row,
                        comment_id=comment_id,
                        post_id=row.get("post_id", ""),
                        comment_text=row.get("comment_text", ""),
                        title_youtube=row.get("title_youtube", ""),
                        source_query=row.get("source_query", ""),
                        temperature_group="0.0",
                    )
                )

    return records


def sample_records(
    records: list[SampleRecord],
    sample_size: int,
    seed: int,
) -> list[SampleRecord]:
    if len(records) < sample_size:
        raise ValueError(f"Not enough records to sample {sample_size}; only found {len(records)}.")

    rng = random.Random(seed)
    return rng.sample(records, sample_size)


def load_original_rows(records: list[SampleRecord]) -> dict[tuple[str, int, str], dict[str, str]]:
    needed_by_file: dict[str, set[int]] = {}
    for record in records:
        needed_by_file.setdefault(record.source_file, set()).add(record.source_row)

    loaded: dict[tuple[str, int, str], dict[str, str]] = {}
    for source_file, needed_rows in needed_by_file.items():
        csv_path = CRAWLED_DIR / source_file
        if not csv_path.exists():
            raise FileNotFoundError(f"Original CSV not found for sample join: {csv_path}")

        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for source_row, row in enumerate(reader, start=1):
                if source_row not in needed_rows:
                    continue
                key = (source_file, source_row, row.get("comment_id", ""))
                loaded[key] = row

    return loaded


def build_output_rows(records: list[SampleRecord]) -> list[dict[str, Any]]:
    original_rows = load_original_rows(records)
    output_rows: list[dict[str, Any]] = []

    for record in records:
        original = original_rows.get(record.key)
        if original is None:
            raise KeyError(
                f"Could not join labeled record back to CSV: "
                f"{record.source_file} row {record.source_row} comment_id={record.comment_id}"
            )

        output_rows.append(
            {
                "source_file": record.source_file,
                "source_row": record.source_row,
                "comment_id": record.comment_id,
                "post_id": record.post_id,
                "comment_text": record.comment_text or original.get("comment_text", ""),
                "title_youtube": record.title_youtube or original.get("title_youtube", ""),
                "source_query": record.source_query or original.get("source_query", ""),
                "labeled_at": record.labeled_at,
                "temperature_group": record.temperature_group,
            }
        )

    return output_rows


def write_sample(output_path: Path, rows: list[dict[str, Any]], append: bool = False) -> None:
    existing_rows: list[dict[str, Any]] = []
    if append and output_path.exists():
        with output_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            existing_rows = list(reader)

    all_rows = existing_rows + rows

    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)


def parse_args(default_output: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample crawled comments for manual labeling.")
    parser.add_argument("--n", type=int, default=DEFAULT_SAMPLE_SIZE, help="Number of rows to sample.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducible sampling.")
    parser.add_argument("--output", type=Path, default=default_output, help="Output CSV path.")
    parser.add_argument("--append-to", type=Path, default=None, help="Append to existing CSV file (merges new + existing rows).")
    return parser.parse_args()


def run_sampling(target_group: TemperatureGroup, default_output: str) -> None:
    args = parse_args(ROOT / default_output)

    existing_keys: set[tuple[str, int, str]] = set()
    for existing_file in [ROOT / "manual_samples_temperature_0.csv", ROOT / "manual_samples_temperature_1.csv"]:
        existing_keys |= read_existing_sample_keys(existing_file)
    if args.append_to:
        existing_keys |= read_existing_sample_keys(args.append_to)

    if target_group == "1.0":
        labeled_keys = {record.key for record in read_labeled_records("1.0")}
        all_excluded = existing_keys | labeled_keys
        records = read_remaining_crawled_records(all_excluded)
        pool_description = "crawled_dataset_excluding_temperature_1_and_existing_samples"
    else:
        temperature_1_keys = {record.key for record in read_labeled_records("1.0")}
        all_excluded = existing_keys | temperature_1_keys
        records = read_remaining_crawled_records(all_excluded)
        pool_description = "crawled_dataset_excluding_temperature_1_and_existing_samples"

    sampled = sample_records(records, args.n, args.seed)
    rows = build_output_rows(sampled)

    output_path = args.append_to if args.append_to else args.output
    write_sample(output_path, rows, append=bool(args.append_to))

    print(
        f"temperature_group={target_group}, pool={len(records)}, pool_source={pool_description}, "
        f"sampled={len(rows)}, output={output_path}"
    )
