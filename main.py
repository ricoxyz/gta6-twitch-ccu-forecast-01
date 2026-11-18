"""Entrypoint for gta6-twitch-ccu-forecast-01 (Python 3.13, standard library only).

Reads every <id>.in case file in $MAKEFUN_INPUT_DIR (default /input) in deterministic name
order and writes the calibrated 23-hour x 5-quantile forecast to $MAKEFUN_OUTPUT_DIR/<id>.out
(default /output).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Grader executes `python3 -P -s /bot/main.py`; restore script directory to sys.path first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import forecast_case

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        payload = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
