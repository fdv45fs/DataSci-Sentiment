import json
import os
import sys

import httpx
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    api_key = os.environ["DEEPSEEK_API_KEY"]
    base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    url = f"{base}/user/balance"

    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        print(
            f"Request failed: HTTP {e.response.status_code}\n{e.response.text}",
            file=sys.stderr,
        )
        raise SystemExit(1) from e
    except httpx.RequestError as e:
        print(f"Request error: {e}", file=sys.stderr)
        raise SystemExit(1) from e

    data = response.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
