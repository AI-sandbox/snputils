import importlib
from pathlib import Path

import numpy as np
import pytest

from snputils.snp.genobj.snpobj import SNPObject


def _streaming_chunk(
    variant_ids=("v1", "v2", "v3"),
    samples=("s1", "s2"),
    chrom="22",
) -> SNPObject:
    n_variants = len(variant_ids)
    n_samples = len(samples)
    return SNPObject(
        calldata_gt=np.zeros((n_variants, n_samples, 2), dtype=np.int8),
        samples=np.array(samples, dtype=object),
        variants_ref=np.resize(np.array(["A", "C", "G"], dtype=object), n_variants),
        variants_alt=np.resize(np.array(["G", "T", "A"], dtype=object), n_variants),
        variants_chrom=np.repeat(chrom, n_variants).astype(object),
        variants_id=np.array(variant_ids, dtype=object),
        variants_pos=np.arange(10, 10 + n_variants),
    )


def _filter_chunk(chunk: SNPObject, *, samples=None, variant_ids=None) -> SNPObject:
    if samples is not None:
        chunk = chunk.filter_samples(samples=samples)
    if variant_ids is not None:
        wanted = np.asarray(variant_ids, dtype=object)
        chunk = chunk.filter_variants(mask=np.isin(chunk.variants_id, wanted))
    return chunk


def test_load_dataset_streaming_honors_variant_ids_with_max_variants(tmp_path, monkeypatch):
    load_dataset_module = importlib.import_module("snputils.datasets.load_dataset")

    class FakeSNPReader:
        def __init__(self, source):
            self.source = source

        def iter_read(self, *, sample_ids=None, variant_ids=None, **kwargs):
            chunk = _streaming_chunk()
            yield _filter_chunk(chunk, samples=sample_ids, variant_ids=variant_ids)

    monkeypatch.setattr(load_dataset_module, "SNPReader", FakeSNPReader)

    snpobj = load_dataset_module.load_dataset(
        "1kgp",
        genotype_sources=[tmp_path / "toy.vcf"],
        download_genotypes=False,
        variants_ids=["v3"],
        max_variants=1,
        verbose=False,
    )

    assert snpobj.variants_id.tolist() == ["v3"]


def test_load_dataset_streaming_finds_requested_variant_after_empty_source(tmp_path, monkeypatch):
    load_dataset_module = importlib.import_module("snputils.datasets.load_dataset")
    chunks = {
        "chr1.vcf": _streaming_chunk(("v1", "v2")),
        "chr2.vcf": _streaming_chunk(("v3", "v4")),
    }

    class FakeSNPReader:
        def __init__(self, source):
            self.source = Path(source)

        def iter_read(self, *, sample_ids=None, variant_ids=None, **kwargs):
            chunk = chunks[self.source.name]
            yield _filter_chunk(chunk, samples=sample_ids, variant_ids=variant_ids)

    monkeypatch.setattr(load_dataset_module, "SNPReader", FakeSNPReader)

    snpobj = load_dataset_module.load_dataset(
        "1kgp",
        genotype_sources=[tmp_path / "chr1.vcf", tmp_path / "chr2.vcf"],
        download_genotypes=False,
        variants_ids=["v3"],
        max_variants=1,
        verbose=False,
    )

    assert snpobj.variants_id.tolist() == ["v3"]


def test_load_dataset_streaming_forwards_sample_ids_to_reader(tmp_path, monkeypatch):
    load_dataset_module = importlib.import_module("snputils.datasets.load_dataset")

    class FakeSNPReader:
        calls = []

        def __init__(self, source):
            self.source = source

        def iter_read(self, *, sample_ids=None, **kwargs):
            FakeSNPReader.calls.append(sample_ids)
            yield _filter_chunk(_streaming_chunk(), samples=sample_ids)

    monkeypatch.setattr(load_dataset_module, "SNPReader", FakeSNPReader)

    snpobj = load_dataset_module.load_dataset(
        "1kgp",
        genotype_sources=[tmp_path / "toy.vcf"],
        download_genotypes=False,
        sample_ids=["s2"],
        max_variants=1,
        verbose=False,
    )

    assert [call.tolist() for call in FakeSNPReader.calls] == [["s2"]]
    assert snpobj.samples.tolist() == ["s2"]


def test_load_dataset_populations_add_sample_metadata(tmp_path, monkeypatch):
    load_dataset_module = importlib.import_module("snputils.datasets.load_dataset")
    metadata_path = tmp_path / "panel.tsv"
    metadata_path.write_text(
        "sample\tpopulation\tsex\n"
        "s1\tCEU\tF\n"
        "s2\tYRI\tM\n"
        "s3\tYRI\tF\n"
    )

    class FakeSNPReader:
        def __init__(self, source):
            self.source = source

        def iter_read(self, *, sample_ids=None, **kwargs):
            yield _filter_chunk(_streaming_chunk(samples=("s1", "s2", "s3")), samples=sample_ids)

    monkeypatch.setattr(load_dataset_module, "SNPReader", FakeSNPReader)

    snpobj = load_dataset_module.load_dataset(
        "1kgp",
        genotype_sources=[tmp_path / "toy.vcf"],
        download_genotypes=False,
        populations=["YRI"],
        samples_per_population=2,
        metadata_path=metadata_path,
        max_variants=1,
        verbose=False,
    )

    assert snpobj.samples.tolist() == ["s2", "s3"]
    assert snpobj.sample_fid.tolist() == ["YRI", "YRI"]
    assert snpobj.sample_sex.tolist() == ["M", "F"]


def test_load_dataset_streaming_raises_when_variant_ids_do_not_match(tmp_path, monkeypatch):
    load_dataset_module = importlib.import_module("snputils.datasets.load_dataset")

    class FakeSNPReader:
        def __init__(self, source):
            self.source = source

        def iter_read(self, *, sample_ids=None, variant_ids=None, **kwargs):
            yield _filter_chunk(_streaming_chunk(), samples=sample_ids, variant_ids=variant_ids)

    monkeypatch.setattr(load_dataset_module, "SNPReader", FakeSNPReader)

    with pytest.raises(RuntimeError, match="No eligible variants matched variants_ids"):
        load_dataset_module.load_dataset(
            "1kgp",
            genotype_sources=[tmp_path / "toy.vcf"],
            download_genotypes=False,
            variants_ids=["missing"],
            max_variants=1,
            verbose=False,
        )


def test_load_dataset_non_streaming_plink_path_uses_extract_and_keep(tmp_path, monkeypatch):
    load_dataset_module = importlib.import_module("snputils.datasets.load_dataset")
    captured = {}
    commands = []

    def fake_execute_plink_cmd(cmd, cwd=None):
        commands.append(cmd)
        if "--extract" in cmd:
            captured["extract"] = Path(cmd[cmd.index("--extract") + 1]).read_text().splitlines()
        if "--keep" in cmd:
            captured["keep"] = Path(cmd[cmd.index("--keep") + 1]).read_text().splitlines()
        out_prefix = cmd[cmd.index("--out") + 1]
        for ext in ("pgen", "psam", "pvar"):
            Path(cwd, f"{out_prefix}.{ext}").write_text("")

    class FakePGENReader:
        paths = []

        def __init__(self, path):
            self.path = Path(path)
            FakePGENReader.paths.append(self.path)

        def read(self, **kwargs):
            return _streaming_chunk(("v2",), samples=("s2",))

    monkeypatch.setattr(load_dataset_module, "execute_plink_cmd", fake_execute_plink_cmd)
    monkeypatch.setattr(load_dataset_module, "PGENReader", FakePGENReader)

    snpobj = load_dataset_module.load_dataset(
        "1kgp",
        genotype_sources=[tmp_path / "chr22.vcf"],
        output_dir=tmp_path,
        download_genotypes=False,
        variants_ids=["v2"],
        sample_ids=["s2"],
        verbose=False,
    )

    assert "--extract" in commands[0]
    assert "--keep" in commands[0]
    assert captured == {"extract": ["v2"], "keep": ["s2"]}
    assert FakePGENReader.paths == [tmp_path / "1kgp"]
    assert snpobj.variants_id.tolist() == ["v2"]
