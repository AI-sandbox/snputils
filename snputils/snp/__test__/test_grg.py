import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pygrgl as pyg
import pytest

from snputils.snp.genobj.grgobj import GRGObject
from snputils.snp.io import GRGWriter
from snputils.snp.io.read.auto import SNPReader
from snputils.snp.io.read.functional import read_grg
from snputils.snp.io.read.grg import GRGReader
from snputils.snp.io.read.vcf import VCFReader


def _build_toy_grg() -> pyg.MutableGRG:
    """
    Create a non-trivial GRG:
    - 3 diploid individuals (6 sample nodes)
    - two internal branches under a root
    - 5 mutations placed across root/branches/samples
    """
    grg = pyg.MutableGRG(6, 2, True)
    root = grg.make_node()
    left = grg.make_node()
    right = grg.make_node()

    grg.connect(root, left)
    grg.connect(root, right)
    for sample in [0, 1, 2]:
        grg.connect(left, sample)
    for sample in [3, 4, 5]:
        grg.connect(right, sample)

    grg.add_mutation(pyg.Mutation(100.0, "G", "A", 0.0), root)   # frequency 1.0
    grg.add_mutation(pyg.Mutation(110.0, "T", "C", 0.0), left)   # frequency 0.5
    grg.add_mutation(pyg.Mutation(120.0, "C", "G", 0.0), right)  # frequency 0.5
    grg.add_mutation(pyg.Mutation(130.0, "A", "T", 0.0), 0)      # frequency 1/6
    grg.add_mutation(pyg.Mutation(140.0, "G", "T", 0.0), 5)      # frequency 1/6
    return grg


def _build_grg_file(path: Path) -> Path:
    GRGWriter(_build_toy_grg(), str(path)).write()
    return path


def test_grg_object_properties_and_basic_counts(tmp_path):
    grg = _build_toy_grg()
    obj = GRGObject(calldata_gt=grg, filename=str(tmp_path / "toy.grg"), mutable=True)

    assert obj.calldata_gt is grg
    assert obj.filename.endswith("toy.grg")
    assert obj.mutable is True
    assert obj.n_samples() == 3
    assert obj.n_samples(ploidy=1) == 6
    assert obj.n_snps() == 5

    replacement = pyg.MutableGRG(2, 2, True)
    obj.calldata_gt = replacement
    obj.filename = str(tmp_path / "updated.grg")
    assert obj.calldata_gt is replacement
    assert obj.filename.endswith("updated.grg")


def test_grg_object_allele_freq_for_toy_grg():
    obj = GRGObject(calldata_gt=_build_toy_grg())
    np.testing.assert_allclose(
        obj.allele_freq(),
        np.array([1.0, 0.5, 0.5, 1.0 / 6.0, 1.0 / 6.0]),
    )


def test_grg_object_to_grg_writes_loadable_file(tmp_path):
    obj = GRGObject(calldata_gt=_build_toy_grg())
    out = tmp_path / "saved.grg"
    obj.to_grg(str(out))

    loaded = pyg.load_immutable_grg(str(out), True)
    assert loaded.num_mutations == 5
    assert loaded.num_samples == 6


def test_grg_object_cli_wrappers_use_expected_arguments(monkeypatch):
    obj = GRGObject(calldata_gt=_build_toy_grg(), filename="/tmp/default.grg")
    calls = []

    def fake_run(cmd, stdout=None, **_kwargs):
        calls.append(cmd)
        if stdout is not None:
            stdout.write(b"a\tb\n1\t2\n")

    monkeypatch.setattr("snputils.snp.genobj.grgobj.subprocess.run", fake_run)

    freq = obj.allele_freq_from_file(None)
    assert calls[0] == ["grg", "process", "freq", "/tmp/default.grg"]
    assert isinstance(freq, pd.DataFrame)
    assert list(freq.columns) == ["a", "b"]

    gwas = obj.gwas("/tmp/pheno.tsv", "/tmp/input.grg")
    assert calls[1] == ["grg", "process", "gwas", "/tmp/input.grg", "--phenotype", "/tmp/pheno.tsv"]
    assert isinstance(gwas, pd.DataFrame)
    assert list(gwas.columns) == ["a", "b"]


def test_grg_object_merge_delegates_to_underlying_graph():
    class DummyGRG:
        def __init__(self):
            self.called = None

        def merge(self, files, combine_nodes):
            self.called = (files, combine_nodes)

    dummy = DummyGRG()
    obj = GRGObject(calldata_gt=dummy)

    obj.merge(True, "/tmp/a.grg", "/tmp/b.grg")
    assert dummy.called == (["/tmp/a.grg", "/tmp/b.grg"], True)

    with pytest.raises(TypeError):
        obj.merge(False, 7)


def test_grg_reader_loads_immutable_and_mutable(tmp_path):
    path = _build_grg_file(tmp_path / "toy.grg")

    immutable = GRGReader(path).read(mutable=False)
    mutable = GRGReader(path).read(mutable=True)

    assert isinstance(immutable, GRGObject)
    assert isinstance(immutable.calldata_gt, pyg.GRG)
    assert immutable.filename == str(path.resolve())
    assert immutable.mutable is False
    assert immutable.calldata_gt.num_mutations == 5
    assert immutable.calldata_gt.num_samples == 6

    assert isinstance(mutable.calldata_gt, pyg.MutableGRG)
    assert mutable.filename == str(path.resolve())
    assert mutable.mutable is True
    assert mutable.calldata_gt.num_mutations == 5


def test_grg_reader_uses_trees_loader_for_trees_input(monkeypatch, tmp_path):
    trees_file = tmp_path / "toy.trees"
    trees_file.write_text("")
    expected = _build_toy_grg()
    called = {}

    def fake_grg_from_trees(path, binary_mutations):
        called["path"] = path
        called["binary_mutations"] = binary_mutations
        return expected

    monkeypatch.setattr("snputils.snp.io.read.grg.pyg.grg_from_trees", fake_grg_from_trees)

    obj = GRGReader(trees_file).read(binary_mutations=True)
    assert obj.calldata_gt is expected
    assert obj.mutable is True
    assert called["path"] == str(trees_file.resolve())
    assert called["binary_mutations"] is True


def test_grg_writer_full_and_subset_paths(monkeypatch, tmp_path):
    path = tmp_path / "writer_full.grg"
    grg = _build_toy_grg()

    GRGWriter(grg, str(path)).write(allow_simplify=False)
    loaded = pyg.load_immutable_grg(str(path), True)
    assert loaded.num_mutations == 5

    subset_calls = {}

    def fake_save_subset(grgobj, filename, direction, seed_list, bp_range):
        subset_calls["grgobj"] = grgobj
        subset_calls["filename"] = filename
        subset_calls["direction"] = direction
        subset_calls["seed_list"] = seed_list
        subset_calls["bp_range"] = bp_range

    monkeypatch.setattr("snputils.snp.io.write.grg.pyg.save_subset", fake_save_subset)

    GRGWriter(grg, str(tmp_path / "writer_subset.grg")).write(
        subset=True,
        direction=pyg.TraversalDirection.UP,
        seed_list=[0, 1],
    )
    assert subset_calls["grgobj"] is grg
    assert subset_calls["direction"] == pyg.TraversalDirection.UP
    assert subset_calls["seed_list"] == [0, 1]
    assert subset_calls["bp_range"] == (0, 0)

    with pytest.raises(ValueError):
        GRGWriter(grg, str(tmp_path / "missing_direction.grg")).write(subset=True, seed_list=[0])
    with pytest.raises(ValueError):
        GRGWriter(grg, str(tmp_path / "missing_seed.grg")).write(
            subset=True,
            direction=pyg.TraversalDirection.UP,
        )


def test_vcf_to_grg_builds_expected_construct_command(monkeypatch):
    calls = []

    def fake_to_igd(self, igd_file=None, logfile_out=None, logfile_err=None):
        self._igd_path = Path(igd_file or "/tmp/input.igd")

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))

    monkeypatch.setattr(VCFReader, "to_igd", fake_to_igd)
    monkeypatch.setattr("snputils.snp.io.read.vcf.subprocess.run", fake_run)

    reader = VCFReader("/tmp/input.vcf.gz")
    reader.debug = False
    reader.to_grg(
        range="100-200",
        parts=4,
        jobs=3,
        trees=2,
        binmuts=True,
        no_file_cleanup=True,
        maf_flip=True,
        shape_lf_filter=0.2,
        population_ids="/tmp/pops.tsv:POP",
        bs_triplet=7,
        out_file="/tmp/out.grg",
        verbose=True,
        no_merge=True,
    )

    cmd, kwargs = calls[-1]
    assert cmd[:2] == ["grg", "construct"]
    assert "-r" in cmd and "100-200" in cmd
    assert "-p" in cmd and "4" in cmd
    assert "-j" in cmd and "3" in cmd
    assert "-t" in cmd and "2" in cmd
    assert "-b" in cmd
    assert "-c" in cmd
    assert "--maf-flip" in cmd
    assert "--shape-lf-filter" in cmd and "0.2" in cmd
    assert "--population-ids" in cmd and "/tmp/pops.tsv:POP" in cmd
    assert "--bs_triplet" in cmd and "7" in cmd
    assert "--out-file" in cmd and "/tmp/out.grg" in cmd
    assert "-v" in cmd
    assert "--no-merge" in cmd
    assert cmd[-1] == "/tmp/input.igd"
    assert kwargs["check"] is True


def test_grg_writer_is_publicly_exported_and_grg_not_in_auto_reader():
    import snputils as su
    from snputils.snp.io import GRGWriter as io_grg_writer
    from snputils.snp.io.write import GRGWriter as write_grg_writer

    assert su.GRGWriter is io_grg_writer
    assert io_grg_writer is write_grg_writer

    with pytest.raises(ValueError):
        SNPReader("toy.grg")


def test_read_grg_functional_entrypoint(tmp_path):
    path = _build_grg_file(tmp_path / "functional.grg")

    obj = read_grg(path, mutable=False)
    assert isinstance(obj, GRGObject)
    assert obj.calldata_gt.num_mutations == 5


@pytest.mark.integration
def test_vcf_to_grg_cli_integration(tmp_path):
    if shutil.which("grg") is None:
        pytest.skip("grg CLI not available")

    pysam = pytest.importorskip("pysam", reason="pysam is required to create tabix-indexed VCFs")

    raw_vcf = tmp_path / "tiny.vcf"
    raw_vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=1>\n"
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0\ts1\n"
        "1\t100\trs1\tA\tG\t.\tPASS\t.\tGT\t0|1\t1|1\n"
    )

    gz_vcf = tmp_path / "tiny.vcf.gz"
    pysam.tabix_compress(str(raw_vcf), str(gz_vcf), force=True)
    pysam.tabix_index(str(gz_vcf), preset="vcf", force=True)

    out_grg = tmp_path / "tiny.grg"
    reader = VCFReader(gz_vcf)
    reader.debug = False

    try:
        reader.to_grg(parts=1, jobs=1, trees=1, out_file=str(out_grg))
    except subprocess.CalledProcessError as exc:
        pytest.skip(f"grg CLI conversion failed in this environment: {exc}")

    assert out_grg.exists()
    loaded = pyg.load_immutable_grg(str(out_grg), True)
    assert loaded.num_mutations >= 1
