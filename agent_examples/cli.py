from __future__ import annotations

import argparse
import json
from typing import Any


def print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def chapter_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--json", action="store_true", help="保留给扩展使用，demo 默认输出 JSON")
    return parser

