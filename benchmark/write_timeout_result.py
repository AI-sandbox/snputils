import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


def write_timeout_result(
    *,
    output: Path,
    benchmark_format: str,
    reader: str,
    path: str,
    sum_strands: bool,
    timeout_seconds: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "datetime": datetime.now(timezone.utc).isoformat(),
        "machine_info": {
            "node": platform.node(),
            "processor": platform.processor(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "system": platform.system(),
        },
        "benchmarks": [
            {
                "group": f"{benchmark_format.upper()}-readers",
                "name": f"test_{benchmark_format}_readers[timeout-{reader}]",
                "fullname": f"benchmark/read_{benchmark_format}.py::timeout[{reader}]",
                "params": {
                    "name": reader,
                    "reader": "timeout",
                },
                "param": f"timeout-{reader}",
                "stats": {
                    "min": timeout_seconds,
                    "max": timeout_seconds,
                    "mean": timeout_seconds,
                    "stddev": 0.0,
                    "rounds": 1,
                    "median": timeout_seconds,
                    "iqr": 0.0,
                    "q1": timeout_seconds,
                    "q3": timeout_seconds,
                    "ops": 1.0 / timeout_seconds if timeout_seconds else 0.0,
                    "iterations": 1,
                    "data": [timeout_seconds],
                },
                "extra_info": {
                    "path": path,
                    "sum_strands": sum_strands,
                    "timeout": True,
                    "timeout_seconds": timeout_seconds,
                    "max_memory_mb": None,
                    "max_memory_mb_mean": None,
                    "max_memory_mb_stddev": None,
                    "max_memory_mb_data": [],
                },
            }
        ],
    }
    with output.open("w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a censored benchmark JSON for a timed-out reader.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", dest="benchmark_format", required=True)
    parser.add_argument("--reader", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--sum-strands", default="true", choices=("true", "false"))
    parser.add_argument("--timeout-seconds", type=float, required=True)
    args = parser.parse_args()

    write_timeout_result(
        output=args.output,
        benchmark_format=args.benchmark_format,
        reader=args.reader,
        path=args.path,
        sum_strands=args.sum_strands == "true",
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    main()
