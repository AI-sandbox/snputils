import builtins
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pytest

from snputils.ancestry.io.local.read.__test__.fixtures import make_synthetic_dataset, write_msp
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


def _write_binary_phe(path: Path, sample_ids, y_binary: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("#FID IID PHENO\n")
        for sid, yi in zip(sample_ids, y_binary):
            status = 2 if int(yi) == 1 else 1
            handle.write(f"{sid} {sid} {status}\n")


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
            "--snp-path",
            str(vcf_path),
            "--fig-path",
            str(fig_path),
            "--npy-path",
            str(npy_path),
            "--backend",
            "sklearn",
            "--n-components",
            "2",
        ],
    )

    assert main() == 0
    assert fig_path.exists()
    assert npy_path.exists()

    components = np.load(npy_path)
    assert components.shape == (2, 2)


def test_main_pca_sklearn_smoke_with_pgen_auto_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matplotlib.use("Agg", force=True)
    monkeypatch.setenv("MPLBACKEND", "Agg")

    fig_path = tmp_path / "pca_pgen.png"
    npy_path = tmp_path / "components_pgen.npy"
    pgen_path = Path(__file__).resolve().parents[3] / "data" / "pgen" / "subset.pgen"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "snputils",
            "pca",
            "--snp-path",
            str(pgen_path),
            "--fig-path",
            str(fig_path),
            "--npy-path",
            str(npy_path),
            "--backend",
            "sklearn",
            "--sum-strands",
        ],
    )

    assert main() == 0
    assert fig_path.exists()
    assert npy_path.exists()

    components = np.load(npy_path)
    assert components.ndim == 2
    assert components.shape[1] == 2


def test_main_admixture_map_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sample_ids, y, lai, chromosomes, starts, ends, ancestry_map = make_synthetic_dataset(
        n_samples=24,
        n_windows=12,
        seed=77,
    )
    phe_path = tmp_path / "toy.phe"
    msp_path = tmp_path / "toy.msp"
    out_dir = tmp_path / "out"
    _write_binary_phe(phe_path, sample_ids, y)
    write_msp(msp_path, sample_ids, lai, chromosomes, starts, ends, ancestry_map)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "snputils",
            "admixture-map",
            "--phe-id",
            "toy",
            "--phe-path",
            str(phe_path),
            "--msp-path",
            str(msp_path),
            "--results-path",
            str(out_dir),
            "--batch-size",
            "8",
            "--keep-hla",
        ],
    )

    assert main() == 0
    output_file = out_dir / "toy_admixmap.tsv.gz"
    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_main_without_command_prints_help_and_exits_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["snputils"])

    assert main() == 1
    captured = capsys.readouterr()
    assert "usage:" in captured.err
    assert "pca" in captured.err
    assert "admixture-map" in captured.err


def test_main_help_flag_prints_help_and_exits_0(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["snputils", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out
    assert "admixture-map" in captured.out
