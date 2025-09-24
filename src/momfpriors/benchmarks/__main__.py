from __future__ import annotations

import argparse
import logging

from momfpriors.benchmarks import BENCHMARKS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main(
    list: str
):
    match list:
        case "all":
            logger.info(BENCHMARKS.keys())
        case _:
            logger.error(f"Invalid list type: {list}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list",
        required=True,
        choices=["all"],
        help="List all Benchmarks or filter by type"
        "\nall: list all Benchmarks"
    )
    args = parser.parse_args()
    main(args.list)
