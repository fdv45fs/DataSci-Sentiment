import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "youtube_comments_v2"
OUTPUT_DIR = ROOT / "labeled"
ERROR_DIR = ROOT / "label_errors"
PROMPT_FILE = ROOT / "sentiment_prompt.txt"
SCHEMA_VERSION = "sentiment-v1"
PART_COUNT = 6
MAX_API_ATTEMPTS = 3
VALID_LABELS = {"positive", "neutral", "negative"}
REQUIRED_COLUMNS = {"comment_id", "post_id", "comment_text", "title_youtube", "source_query"}
BATCH_NUMBER_RE = re.compile(r"batch_(\d+)")


@dataclass(frozen=True)
class PendingComment:
    source_file: str
    source_row: int
    comment_id: str
    post_id: str
    comment_text: str
    title_youtube: str
    source_query: str

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.source_file, self.source_row, self.comment_id)

    def api_payload(self) -> dict[str, str]:
        return {
            "comment_text": self.comment_text,
            "title_youtube": self.title_youtube,
            "source_query": self.source_query,
        }


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_prompt() -> str:
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_FILE}")
    return PROMPT_FILE.read_text(encoding="utf-8")


def batch_sort_key(path: Path) -> tuple[int, str]:
    match = BATCH_NUMBER_RE.search(path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (sys.maxsize, path.name)


def discover_batches() -> list[Path]:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")
    batches = sorted(INPUT_DIR.rglob("*.csv"), key=batch_sort_key)
    if not batches:
        raise FileNotFoundError(f"No CSV batches found in: {INPUT_DIR}")
    return batches


def split_batches(batches: list[Path], part_count: int) -> list[list[Path]]:
    total = len(batches)
    return [
        batches[(total * index) // part_count : (total * (index + 1)) // part_count]
        for index in range(part_count)
    ]


def describe_part(part_batches: list[Path]) -> str:
    if not part_batches:
        return "0 batch(es)"

    first = part_batches[0].name
    last = part_batches[-1].name
    first_number = batch_sort_key(part_batches[0])[0]
    last_number = batch_sort_key(part_batches[-1])[0]
    if first_number != sys.maxsize and last_number != sys.maxsize:
        return f"{len(part_batches)} batch(es), batch {first_number} -> {last_number}"
    return f"{len(part_batches)} batch(es), {first} -> {last}"


def choose_batch_part(batches: list[Path]) -> tuple[int, list[Path]]:
    parts = split_batches(batches, PART_COUNT)
    env_part = os.getenv("LABEL_PART")

    if env_part:
        try:
            selected = int(env_part)
        except ValueError as error:
            raise ValueError(f"LABEL_PART must be a number from 1 to {PART_COUNT}.") from error
        if 1 <= selected <= PART_COUNT:
            return selected, parts[selected - 1]
        raise ValueError(f"LABEL_PART must be a number from 1 to {PART_COUNT}.")

    print("Choose which data part this machine should process:")
    for index, part_batches in enumerate(parts, start=1):
        print(f"  {index}. {describe_part(part_batches)}")

    while True:
        choice = input(f"Enter part number (1-{PART_COUNT}): ").strip()
        try:
            selected = int(choice)
        except ValueError:
            print(f"Please enter a number from 1 to {PART_COUNT}.")
            continue
        if 1 <= selected <= PART_COUNT:
            return selected, parts[selected - 1]
        print(f"Please enter a number from 1 to {PART_COUNT}.")


def output_path_for(batch_path: Path) -> Path:
    relative_path = batch_path.relative_to(INPUT_DIR)
    return OUTPUT_DIR / relative_path.with_suffix(".labels.jsonl")


def error_path_for(batch_path: Path) -> Path:
    relative_path = batch_path.relative_to(INPUT_DIR)
    return ERROR_DIR / relative_path.with_suffix(".errors.jsonl")


def read_existing_keys(output_path: Path) -> set[tuple[str, int, str]]:
    keys: set[tuple[str, int, str]] = set()
    if not output_path.exists():
        return keys

    with output_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                keys.add(
                    (
                        str(record["source_file"]),
                        int(record["source_row"]),
                        str(record["comment_id"]),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Cannot resume because {output_path} has an invalid JSONL "
                    f"record at line {line_number}: {error}"
                ) from error
    return keys


def read_batch(batch_path: Path, existing_keys: set[tuple[str, int, str]]) -> tuple[int, list[PendingComment]]:
    pending: list[PendingComment] = []
    source_file = batch_path.relative_to(INPUT_DIR).as_posix()

    with batch_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {batch_path}")

        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"CSV is missing required columns ({missing}): {batch_path}")

        total_rows = 0
        for source_row, row in enumerate(reader, start=1):
            total_rows += 1
            comment = PendingComment(
                source_file=source_file,
                source_row=source_row,
                comment_id=row["comment_id"],
                post_id=row["post_id"],
                comment_text=row["comment_text"],
                title_youtube=row["title_youtube"],
                source_query=row["source_query"],
            )
            if comment.key not in existing_keys:
                pending.append(comment)

    return total_rows, pending


def strip_code_fence(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def extract_json_array(content: str) -> list[Any]:
    text = strip_code_fence(content)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise
        data = json.loads(text[start : end + 1])

    if not isinstance(data, list):
        raise ValueError("Model response must be a JSON array.")
    return data


def parse_single_label(content: str) -> dict[str, Any]:
    text = strip_code_fence(content)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        normalized = text.strip().strip('"').strip("'").lower()
        if normalized in VALID_LABELS:
            return {"label": normalized}

        for label in VALID_LABELS:
            if normalized == label:
                return {"label": label}

        raise ValueError(f"Model response is not a valid label: {content!r}")

    if isinstance(data, str):
        label = data.lower()
        if label in VALID_LABELS:
            return {"label": label}
        raise ValueError(f"Model response has invalid label {data!r}.")

    if isinstance(data, list):
        return validate_labels(data, 1)[0]

    if isinstance(data, dict):
        return validate_labels([data], 1)[0]

    raise ValueError(f"Model response is not a valid label: {content!r}")


def validate_labels(labels: list[Any], expected_count: int) -> list[dict[str, Any]]:
    if len(labels) != expected_count:
        raise ValueError(f"Expected {expected_count} labels, got {len(labels)}.")

    validated: list[dict[str, Any]] = []
    for index, item in enumerate(labels, start=1):
        if isinstance(item, str):
            item = {"label": item}
        elif not isinstance(item, dict):
            raise ValueError(f"Label item {index} must be a string label or JSON object.")

        label = item.get("label")
        if label not in VALID_LABELS:
            raise ValueError(
                f"Label item {index} has invalid label {label!r}; "
                f"expected one of {sorted(VALID_LABELS)}."
            )

        compact_item = {"label": label}
        if "confidence" in item:
            compact_item["confidence"] = item["confidence"]
        if "reason" in item:
            compact_item["reason"] = item["reason"]
        validated.append(compact_item)

    return validated


def classify_comment(
    client: OpenAI,
    model: str,
    prompt: str,
    comment: PendingComment,
) -> dict[str, Any]:
    payload = comment.api_payload()
    user_message = (
        "Classify this comment. Return exactly one label word only.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message},
        ],
        stream=False,
        extra_body={"thinking": {"type": "disabled"}},
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Model returned an empty response.")

    return parse_single_label(content)


def classify_comment_with_retries(
    client: OpenAI,
    model: str,
    prompt: str,
    comment: PendingComment,
) -> dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(1, MAX_API_ATTEMPTS + 1):
        try:
            return classify_comment(client, model, prompt, comment)
        except KeyboardInterrupt:
            raise
        except Exception as error:
            last_error = error
            print(f"    attempt {attempt}/{MAX_API_ATTEMPTS} failed: {error}", file=sys.stderr)
            if attempt < MAX_API_ATTEMPTS:
                time.sleep(2 ** (attempt - 1))

    if last_error is None:
        raise RuntimeError("API call failed without an error.")
    raise last_error


def append_labels(
    output_path: Path,
    comments: list[PendingComment],
    labels: list[dict[str, Any]],
    model: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_at = now_utc()

    with output_path.open("a", encoding="utf-8", newline="\n") as file:
        for comment, label_data in zip(comments, labels, strict=True):
            record = {
                "schema_version": SCHEMA_VERSION,
                "source_file": comment.source_file,
                "source_row": comment.source_row,
                "comment_id": comment.comment_id,
                "post_id": comment.post_id,
                "label": label_data["label"],
                "model": model,
                "labeled_at": labeled_at,
            }
            if "confidence" in label_data:
                record["confidence"] = label_data["confidence"]
            if "reason" in label_data:
                record["reason"] = label_data["reason"]

            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def append_error(
    error_path: Path,
    comment: PendingComment,
    error: Exception,
    model: str,
    stage: str,
) -> None:
    error_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": SCHEMA_VERSION,
        "source_file": comment.source_file,
        "source_row": comment.source_row,
        "comment_id": comment.comment_id,
        "post_id": comment.post_id,
        "model": model,
        "stage": stage,
        "error": str(error),
        "errored_at": now_utc(),
    }

    with error_path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    load_dotenv()

    prompt = load_prompt()
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )

    batches = discover_batches()
    selected_part, batches = choose_batch_part(batches)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Selected part {selected_part}/{PART_COUNT}: {describe_part(batches)}")
    print(f"Saving labels to {OUTPUT_DIR}")
    print(f"Saving recoverable errors to {ERROR_DIR}")
    print("Processing 1 comment per API call")

    total_rows = 0
    total_existing = 0
    total_new = 0
    total_errors = 0

    try:
        for batch_index, batch_path in enumerate(batches, start=1):
            output_path = output_path_for(batch_path)
            error_path = error_path_for(batch_path)
            try:
                existing_keys = read_existing_keys(output_path)
                row_count, pending_comments = read_batch(batch_path, existing_keys)
            except Exception as error:
                print(f"Skipping {batch_path}: {error}", file=sys.stderr)
                continue

            already_labeled = row_count - len(pending_comments)

            total_rows += row_count
            total_existing += already_labeled

            relative_batch = batch_path.relative_to(INPUT_DIR).as_posix()
            print(
                f"[{batch_index}/{len(batches)}] {relative_batch}: "
                f"rows={row_count}, already_labeled={already_labeled}, "
                f"remaining={len(pending_comments)}"
            )

            if not pending_comments:
                continue

            saved_for_file = 0
            for comment_index, comment in enumerate(pending_comments, start=1):
                print(
                    f"  comment {comment_index}/{len(pending_comments)}: "
                    f"source_row={comment.source_row}..."
                )
                try:
                    label = classify_comment_with_retries(client, model, prompt, comment)
                    append_labels(output_path, [comment], [label], model)
                    saved_for_file += 1
                    total_new += 1
                except KeyboardInterrupt:
                    raise
                except Exception as error:
                    append_error(error_path, comment, error, model, "single_comment")
                    total_errors += 1
                    print(
                        f"  comment {comment_index}/{len(pending_comments)} failed and was logged: "
                        f"{error}",
                        file=sys.stderr,
                    )
                    continue

                print(
                    f"  comment {comment_index}/{len(pending_comments)} saved; "
                    f"file_progress={already_labeled + saved_for_file}/{row_count}"
                )

            print(f"  saved {saved_for_file} new label(s) -> {output_path.relative_to(ROOT)}")
    except KeyboardInterrupt:
        print("\nCanceled by user. Previously saved labels remain in the labeled folder.")
        raise SystemExit(130)

    print(
        "Done. "
        f"total_rows={total_rows}, already_labeled={total_existing}, "
        f"newly_labeled={total_new}, errors_logged={total_errors}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
