import argparse
import json
from math import log10
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


DEFAULT_FORMATS = ("bed", "pgen", "vcf")
DEFAULT_TIME_NAMES = (
    "snputils",
    "pgenlib",
    "pysnptools",
    "pandas-plink",
    "sgkit",
    "plinkio",
    "cyvcf2",
    "scikit-allel",
    "pysam",
    "hail",
    "pyvcf3",
)
DEFAULT_MEMORY_NAMES = (
    "snputils",
    "pgenlib",
    "pysnptools",
    "pandas-plink",
    "sgkit",
    "plinkio",
    "cyvcf2",
    "scikit-allel",
)


def _load_values(results_dir: Path, fmt: str, metric: str) -> dict[str, tuple[float, float]]:
    path = results_dir / f"{fmt}_chr22.json"
    with path.open() as handle:
        data = json.load(handle)

    values = {}
    for bench in data.get("benchmarks", []):
        name = bench.get("params", {}).get("name")
        if not name:
            continue
        if metric == "time":
            value = bench.get("stats", {}).get("mean")
        else:
            value = bench.get("extra_info", {}).get("max_memory_mb")
        if value is not None:
            std = bench.get("stats", {}).get("stddev") or 0.0
            values[name] = (float(value), float(std))
    return values


def _format_time(value: float) -> str:
    if value >= 3600:
        return ">60m"
    if value >= 60:
        minutes = int(value // 60)
        seconds = int(value % 60)
        return f"{minutes}m{seconds}s"
    return f"{value:.1f}s"


def _format_memory(value: float) -> str:
    if value >= 1024:
        return f"{value / 1024:.1f}GiB"
    return f"{value:.0f}MiB"


def _format_value(value: float, metric: str) -> str:
    return _format_time(value) if metric == "time" else _format_memory(value)


def _plot_value(value: float, metric: str) -> float:
    return value / 1024 if metric == "memory" else value


def _axis_limit(values: Iterable[float]) -> tuple[float, float]:
    valid = [value for value in values if value > 0]
    if not valid:
        return 1.0, 1.0
    min_value = min(valid)
    max_value = max(valid)
    crop_limit = min_value * 20
    if max_value <= crop_limit:
        return max_value * 1.25, min_value
    return crop_limit, min_value


def _draw_crop_arrows(ax, xpos: float, ypos: float, n_marks: int, *, color: str, fontsize: int) -> None:
    if n_marks == 1:
        offsets = (0.0,)
    else:
        offsets = (-0.04, 0.04)
    for offset in offsets:
        ax.text(xpos + offset, ypos, "\u2191", ha="center", va="bottom", color=color, fontsize=fontsize)


def _draw_grouped_bars(
    ax,
    names,
    true_values,
    false_values,
    metric: str,
    title: str,
    y_cap: float | None = None,
    axis_top: float | None = None,
) -> None:
    x = np.arange(len(names))
    width = 0.36
    colors = {"true": "#4C72B0", "false": "#DD8452"}
    marker_color = "#C44E52"
    value_rotation = 90
    value_fontsize = 14
    all_values = [
        _plot_value(values.get(name)[0], metric)
        for values in (true_values, false_values)
        for name in names
        if values.get(name) is not None
    ]
    y_max, min_value = _axis_limit(all_values)
    if y_cap is not None:
        y_max = min(y_max, _plot_value(y_cap, metric))
    if axis_top is not None:
        y_top = _plot_value(axis_top, metric)
        y_max = min(y_max, y_top)
    else:
        y_top = y_max * 1.18

    for idx in range(len(names)):
        if idx % 2 == 0:
            ax.axvspan(idx - 0.5, idx + 0.5, color="#f2f2f2", alpha=0.45, linewidth=0, zorder=0)

    for offset, values, label, color in (
        (-width / 2, true_values, "sum_strands=True", colors["true"]),
        (width / 2, false_values, "sum_strands=False", colors["false"]),
    ):
        for idx, name in enumerate(names):
            entry = values.get(name)
            xpos = x[idx] + offset
            if entry is None:
                ax.plot(xpos, y_top * 0.06, "x", color=marker_color, markersize=9, mew=2.5)
                continue

            raw_value, raw_std = entry
            value = _plot_value(raw_value, metric)
            std = _plot_value(raw_std, metric)
            is_cropped = value > y_max
            plotted_value = y_top if is_cropped else value
            yerr = None if is_cropped or std == 0 else std
            ax.bar(
                xpos,
                plotted_value,
                width,
                color=color,
                label=label if idx == 0 else None,
                yerr=yerr,
                ecolor="black",
                capsize=5 if yerr is not None else 0,
                error_kw={"elinewidth": 1.8, "capthick": 1.8},
            )
            text = _format_value(raw_value, metric)
            if is_cropped:
                if min_value > 0:
                    n_marks = max(1, min(2, int(log10(value)) - int(log10(min_value))))
                else:
                    n_marks = 1
                _draw_crop_arrows(ax, xpos, y_top * 0.80, n_marks, color="white", fontsize=15)
                ax.text(
                    xpos,
                    y_top * 0.42,
                    text,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=value_fontsize,
                    rotation=value_rotation,
                )
            else:
                ax.text(
                    xpos,
                    plotted_value + (std if yerr is not None else 0) + y_top * 0.02,
                    text,
                    ha="center",
                    va="bottom",
                    fontsize=value_fontsize,
                    rotation=value_rotation,
                    clip_on=False,
                )

    ax.set_title(title)
    ax.set_ylim(0, y_top)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.grid(True, linestyle="--", alpha=0.45)
    ax.set_axisbelow(True)


def plot_time_memory(
    time_true_dir: Path,
    time_false_dir: Path,
    memory_true_dir: Path,
    memory_false_dir: Path,
    output: Path,
    pdf_output: Path | None,
    time_names: list[str],
    memory_names: list[str],
) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 15,
        "axes.titlesize": 19,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "figure.dpi": 170,
    })

    fig, axs = plt.subplots(
        len(DEFAULT_FORMATS),
        2,
        figsize=(17.75, 9.65),
        sharex="col",
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )

    for row, fmt in enumerate(DEFAULT_FORMATS):
        title = fmt.upper()
        time_true = _load_values(time_true_dir, fmt, "time")
        time_false = _load_values(time_false_dir, fmt, "time")
        memory_true = _load_values(memory_true_dir, fmt, "memory")
        memory_false = _load_values(memory_false_dir, fmt, "memory")

        memory_y_cap = 32_000 if fmt == "bed" else None
        _draw_grouped_bars(
            axs[row, 0],
            time_names,
            time_true,
            time_false,
            "time",
            f"{title} time",
            axis_top=600 if fmt == "vcf" else None,
        )
        _draw_grouped_bars(
            axs[row, 1],
            memory_names,
            memory_true,
            memory_false,
            "memory",
            f"{title} peak memory",
            y_cap=memory_y_cap,
        )
        axs[row, 0].set_ylabel("Time (seconds)")
        axs[row, 1].set_ylabel("Peak memory (GiB)")

    for ax, names in ((axs[-1, 0], time_names), (axs[-1, 1], memory_names)):
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", rotation_mode="anchor")
        ax.get_xticklabels()[0].set_fontweight("bold")

    fig.tight_layout(rect=(0.02, 0, 1, 0.99))
    handles, _ = axs[0, 0].get_legend_handles_labels()
    top_left = axs[0, 0].get_position()
    legend_x = top_left.x0
    fig.text(
        legend_x,
        top_left.y1 + 0.022,
        "sum_strands",
        ha="left",
        va="center",
        fontsize=16,
    )
    fig.legend(
        handles,
        ("True", "False"),
        loc="center left",
        bbox_to_anchor=(legend_x + 0.08, top_left.y1 + 0.022),
        ncol=2,
        frameon=False,
        prop={"size": 16},
        handlelength=1.0,
        handletextpad=0.3,
        columnspacing=0.45,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    if pdf_output is None and output.suffix.lower() != ".pdf":
        pdf_output = output.with_suffix(".pdf")
    if pdf_output is not None and pdf_output != output:
        pdf_output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot time and peak-memory benchmarks side by side.")
    parser.add_argument("--time-true-dir", type=Path, required=True)
    parser.add_argument("--time-false-dir", type=Path, required=True)
    parser.add_argument("--memory-true-dir", type=Path, required=True)
    parser.add_argument("--memory-false-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=None,
        help="Optional PDF output path. Defaults to a .pdf file next to --output.",
    )
    parser.add_argument("--names", nargs="+", default=None, help="Use the same method order for time and memory.")
    parser.add_argument("--time-names", nargs="+", default=None, help="Method order for the time panels.")
    parser.add_argument("--memory-names", nargs="+", default=None, help="Method order for the memory panels.")
    args = parser.parse_args()
    time_names = args.time_names or args.names or list(DEFAULT_TIME_NAMES)
    memory_names = args.memory_names or args.names or list(DEFAULT_MEMORY_NAMES)

    plot_time_memory(
        time_true_dir=args.time_true_dir,
        time_false_dir=args.time_false_dir,
        memory_true_dir=args.memory_true_dir,
        memory_false_dir=args.memory_false_dir,
        output=args.output,
        pdf_output=args.pdf_output,
        time_names=time_names,
        memory_names=memory_names,
    )


if __name__ == "__main__":
    main()
