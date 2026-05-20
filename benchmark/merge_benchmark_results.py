import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def collapse_replicates(benchmarks: list[dict]) -> list[dict]:
    grouped = {}
    order = []
    for bench in benchmarks:
        name = bench.get("params", {}).get("name") or bench.get("name")
        if name not in grouped:
            grouped[name] = []
            order.append(name)
        grouped[name].append(bench)

    collapsed = []
    for name in order:
        group = grouped[name]
        if len(group) == 1:
            collapsed.append(group[0])
            continue

        bench = deepcopy(group[0])
        memory_values = []
        for replicate in group:
            extra_info = replicate.get("extra_info", {})
            value = extra_info.get("max_memory_mb_mean")
            if value is None:
                value = extra_info.get("max_memory_mb")
            if value is not None:
                memory_values.append(float(value))

        if memory_values:
            extra_info = bench.setdefault("extra_info", {})
            extra_info["max_memory_mb"] = max(memory_values)
            extra_info["max_memory_mb_mean"] = mean(memory_values)
            extra_info["max_memory_mb_stddev"] = stdev(memory_values) if len(memory_values) > 1 else 0.0
            extra_info["max_memory_mb_data"] = memory_values

        collapsed.append(bench)
    return collapsed


def merge_results(manifest: Path, output_dir: Path, chrom: str) -> list[Path]:
    rows = read_manifest(manifest)
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for fmt in dict.fromkeys(row["format"] for row in rows):
        format_rows = [row for row in rows if row["format"] == fmt]
        merged = None
        benchmarks = []
        missing = []
        for row in format_rows:
            json_path = Path(row["json"])
            if not json_path.exists():
                missing.append(str(json_path))
                continue
            with json_path.open() as handle:
                data = json.load(handle)
            if merged is None:
                merged = deepcopy(data)
            benchmarks.extend(data.get("benchmarks", []))

        if missing:
            joined = "\n  ".join(missing)
            raise FileNotFoundError(f"Missing benchmark JSON files for {fmt}:\n  {joined}")
        if merged is None:
            raise ValueError(f"No benchmark JSON files were found for {fmt}")

        merged["benchmarks"] = collapse_replicates(benchmarks)
        out_path = output_dir / f"{fmt}_{chrom}.json"
        with out_path.open("w") as handle:
            json.dump(merged, handle, indent=2)
            handle.write("\n")
        output_paths.append(out_path)

    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-reader pytest-benchmark JSON files by format.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chrom", default="chr22")
    args = parser.parse_args()

    for path in merge_results(args.manifest, args.output_dir, args.chrom):
        print(path)


if __name__ == "__main__":
    main()
