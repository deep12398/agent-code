from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_code.seven_layer import FRAMEWORK_MAPPING, diagnose, sample_customer_service_breakdown


def main() -> None:
    print(
        json.dumps(
            {
                "diagnosis": diagnose("prompt changed and wrong answer"),
                "framework_mapping": FRAMEWORK_MAPPING,
                "case": sample_customer_service_breakdown(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
