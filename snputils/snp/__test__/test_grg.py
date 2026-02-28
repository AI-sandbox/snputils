from pathlib import Path

import numpy as np
import pandas as pd
import pygrgl as pyg
import pytest

from snputils.snp.genobj.grgobj import GRGObject
from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io import GRGWriter
from snputils.snp.io.read.pgen import PGENReader
from snputils.snp.io.read.auto import SNPReader
from snputils.snp.io.read.functional import read_grg
from snputils.snp.io.read.grg import GRGReader
from snputils.snp.io.read.vcf import VCFReader
from snputils.snp.io.write.pgen import PGENWriter


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


def test_grg_object_to_snpobject_preserves_genotypes_and_metadata():
    obj = GRGObject(calldata_gt=_build_toy_grg())
    snp = obj.to_snpobject(chrom="22", sum_strands=False)

    assert isinstance(snp, SNPObject)
    assert snp.calldata_gt.shape == (5, 3, 2)
    assert snp.samples.tolist() == ["sample_0", "sample_1", "sample_2"]
    assert snp.variants_chrom.tolist() == ["22"] * 5
    assert snp.variants_pos.tolist() == [100, 110, 120, 130, 140]
    assert snp.variants_id.tolist() == [
        "22:100",
        "22:110",
        "22:120",
        "22:130",
        "22:140",
    ]

    # Mutation(100, G, A) means ALT=G, REF=A for this API.
    assert snp.variants_ref.tolist() == ["A", "C", "G", "T", "T"]
    assert snp.variants_alt.tolist() == ["G", "T", "C", "A", "G"]

    expected = np.array(
        [
            [[1, 1], [1, 1], [1, 1]],  # root mutation
            [[1, 1], [1, 0], [0, 0]],  # left branch
            [[0, 0], [0, 1], [1, 1]],  # right branch
            [[1, 0], [0, 0], [0, 0]],  # sample 0
            [[0, 0], [0, 0], [0, 1]],  # sample 5
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(snp.calldata_gt, expected)


def test_grg_object_to_snpobject_sum_strands_matches_phased_sum():
    obj = GRGObject(calldata_gt=_build_toy_grg())
    phased = obj.to_snpobject(chrom="22", sum_strands=False)
    summed = obj.to_snpobject(chrom="22", sum_strands=True)

    assert summed.calldata_gt.shape == (5, 3)
    np.testing.assert_array_equal(summed.calldata_gt, phased.calldata_gt.sum(axis=2))


def test_grg_object_to_snpobject_allows_pgen_roundtrip(tmp_path):
    obj = GRGObject(calldata_gt=_build_toy_grg())
    snp = obj.to_snpobject(chrom="22", sum_strands=False)

    out = tmp_path / "from_grg"
    PGENWriter(snp, str(out)).write(vzs=False, rename_missing_values=False)

    loaded = PGENReader(out).read(sum_strands=False)
    assert loaded.calldata_gt.shape == (5, 3, 2)
    np.testing.assert_array_equal(loaded.calldata_gt, snp.calldata_gt)
    np.testing.assert_array_equal(loaded.variants_pos, snp.variants_pos)


def test_grg_object_cli_wrappers_use_expected_arguments(monkeypatch):
    obj = GRGObject(calldata_gt=_build_toy_grg(), filename="/tmp/default.grg")
    calls = []

    def fake_run(cmd, stdout=None, **_kwargs):
        calls.append(cmd)
        if cmd[:3] == ["grg", "process", "freq"] and stdout is not None:
            stdout.write(b"a\tb\n1\t2\n")
        if cmd[:2] == ["grapp", "assoc"]:
            out_idx = cmd.index("-o") + 1
            Path(cmd[out_idx]).write_text("a\tb\n1\t2\n")

    monkeypatch.setattr("snputils.snp.genobj.grgobj.subprocess.run", fake_run)

    freq = obj.allele_freq_from_file(None)
    assert calls[0] == ["grg", "process", "freq", "/tmp/default.grg"]
    assert isinstance(freq, pd.DataFrame)
    assert list(freq.columns) == ["a", "b"]

    gwas = obj.gwas("/tmp/pheno.tsv", "/tmp/input.grg")
    assert calls[1][0] == "grapp"
    assert calls[1][1] == "assoc"
    assert "-p" in calls[1] and "/tmp/pheno.tsv" in calls[1]
    assert "-o" in calls[1]
    assert calls[1][-1] == "/tmp/input.grg"
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

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))

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
        population_ids="/tmp/pops.tsv:SAMPLE:POP",
        mutation_batch_size=128,
        out_file="/tmp/out.grg",
        verbose=True,
        no_merge=True,
        force=True,
    )

    cmd, kwargs = calls[-1]
    assert cmd[:2] == ["grg", "construct"]
    assert "-r" in cmd and "100-200" in cmd
    assert "-p" in cmd and "4" in cmd
    assert "-j" in cmd and "3" in cmd
    assert "-t" in cmd and "2" in cmd
    assert "--binary-muts" in cmd
    assert "--no-file-cleanup" in cmd
    assert "--maf-flip" in cmd
    assert "--population-ids" in cmd and "/tmp/pops.tsv:SAMPLE:POP" in cmd
    assert "--mutation-batch-size" in cmd and "128" in cmd
    assert "--out-file" in cmd and "/tmp/out.grg" in cmd
    assert "--verbose" in cmd
    assert "--no-merge" in cmd
    assert "--force" in cmd
    assert cmd[-1].endswith("/tmp/input.vcf.gz")
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


def test_vcf_to_grg_cli_integration(tmp_path):
    vcf = tmp_path / "tiny.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=1>\n"
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0\ts1\n"
        "1\t100\trs1\tA\tG\t.\tPASS\t.\tGT\t0|1\t1|1\n"
    )

    out_grg = tmp_path / "tiny.grg"
    reader = VCFReader(vcf)
    reader.debug = False
    reader.to_grg(parts=1, jobs=1, trees=1, force=True, out_file=str(out_grg))

    assert out_grg.exists()
    loaded = pyg.load_immutable_grg(str(out_grg), True)
    assert loaded.num_samples == 4
