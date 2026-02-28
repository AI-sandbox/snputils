import builtins
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

from snputils.tools.main import main


def _write_tiny_vcf(path: Path) -> None:
    path.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=1>\n"
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0\ts1\n"
        "1\t100\trs1\tA\tG\t.\tPASS\t.\tGT\t0|1\t1|1\n"
        "1\t200\trs2\tC\tT\t.\tPASS\t.\tGT\t0|0\t0|1\n",
        encoding="utf-8",
    )


def test_main_pca_sklearn_smoke_without_torch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    matplotlib.use("Agg", force=True)
    monkeypatch.setenv("MPLBACKEND", "Agg")

    vcf_path = tmp_path / "tiny.vcf"
    fig_path = tmp_path / "pca.png"
    npy_path = tmp_path / "components.npy"
    _write_tiny_vcf(vcf_path)

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError("No module named 'torch'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "snputils",
            "pca",
            "--vcf_file",
            str(vcf_path),
            "--fig_path",
            str(fig_path),
            "--npy_path",
            str(npy_path),
            "--backend",
            "sklearn",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    assert fig_path.exists()
    assert npy_path.exists()

    components = np.load(npy_path)
    assert components.shape == (2, 2)


def test_main_without_command_prints_help_and_exits_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["snputils"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Usage:" in captured.err
    assert "Available commands:" in captured.err
    assert "pca" in captured.err
    assert "admixture_mapping" in captured.err


def test_main_help_flag_prints_help_and_exits_0(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["snputils", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Usage:" in captured.out
    assert "Available commands:" in captured.out
