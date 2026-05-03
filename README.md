# DataSci

Python project managed with `uv` for calling the DeepSeek API through the OpenAI-compatible SDK.

## Setup

```powershell
uv sync
Copy-Item .env.example .env
```

Edit `.env` and set `DEEPSEEK_API_KEY`.

## Run

```powershell
uv run python main.py
```

When the script starts, choose one of 6 parts from the menu. Each machine should use a different part.

For unattended runs, set `LABEL_PART` instead of using the menu:

```powershell
$env:LABEL_PART = "1"
uv run python main.py
```

The script uses natural batch ordering (`batch_1`, `batch_2`, ..., `batch_100`) and resumes from existing files in `labeled/`.

## Merge Labels

After running multiple machines, copy each machine's project folder or `labeled/` folder back to one machine, then merge:

```powershell
uv run python merge_labeled.py .\machine1\labeled .\machine2\labeled .\machine3\labeled --output merged_labeled
```

Duplicate records are deduped by `source_file`, `source_row`, and `comment_id`.
