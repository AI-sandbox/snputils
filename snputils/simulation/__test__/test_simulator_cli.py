import argparse
import pickle
import sys
import types

import numpy as np
import pytest

import snputils.simulation.simulator_cli as simulator_cli
from snputils.simulation.simulator_cli import (
    _batch_genotypes_to_snp_layout,
    _infer_output_format,
    _output_genotype_path,
    _write_batch_truth,
    _write_genotypes_like_input,
)
from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.read.bcf import BCFReader


def _source_snpobject() -> SNPObject:
    return SNPObject(
        genotypes=np.array(
            [
                [[0, 1], [1, 1]],
                [[1, 0], [0, 0]],
            ],
            dtype=np.int8,
        ),
        samples=np.array(["founder1", "founder2"], dtype=object),
        variants_ref=np.array(["A", "C"], dtype=object),
        variants_alt=np.array(["G", "T"], dtype=object),
        variants_chrom=np.array(["1", "1"], dtype=object),
        variants_id=np.array(["rs1", "rs2"], dtype=object),
        variants_pos=np.array([10, 20]),
    )


def test_simulator_output_paths_include_bcf_pickle_and_npz(tmp_path):
    prefix = tmp_path / "simulated"

    assert _output_genotype_path(prefix, "bcf") == tmp_path / "simulated.bcf"
    assert _output_genotype_path(prefix, "pkl") == tmp_path / "simulated.pkl"
    assert _output_genotype_path(prefix, "npz") == tmp_path / "simulated.npz"
    assert _infer_output_format("founders.bcf", "same") == "bcf"


def test_simulator_writes_phased_bcf_and_snpobject_pickle(tmp_path):
    source = _source_snpobject()
    simulated = np.array(
        [
            [[0, 1], [1, 0]],
            [[1, 1], [0, 0]],
        ],
        dtype=np.int8,
    )
    samples = np.array(["SIM1", "SIM2"], dtype=object)

    bcf_path = _write_genotypes_like_input(
        source,
        simulated,
        samples,
        tmp_path / "cohort",
        "bcf",
    )
    observed_bcf = BCFReader(bcf_path).read(genotype_mode="phased")
    np.testing.assert_array_equal(observed_bcf.genotypes, simulated)
    np.testing.assert_array_equal(observed_bcf.samples, samples)

    pickle_path = _write_genotypes_like_input(
        source,
        simulated,
        samples,
        tmp_path / "cohort_object",
        "pkl",
    )
    with pickle_path.open("rb") as handle:
        observed_pickle = pickle.load(handle)
    np.testing.assert_array_equal(observed_pickle.genotypes, simulated)
    np.testing.assert_array_equal(observed_pickle.samples, samples)


def test_batch_output_layout_and_truth_sidecar(tmp_path):
    batch = np.array(
        [
            [[0, 1, 0], [1, 1, 0]],
            [[1, 0, 1], [0, 0, 1]],
        ],
        dtype=np.float32,
    )

    observed = _batch_genotypes_to_snp_layout(batch)
    expected = np.transpose(batch, (2, 0, 1)).astype(np.int8)
    np.testing.assert_array_equal(observed, expected)

    truth_path = tmp_path / "batch_0001.labels.npz"
    _write_batch_truth(
        truth_path,
        labels_d=np.array([[[0, 1]], [[1, 0]]]),
        labels_c=None,
        cp=np.array([[[False, True]], [[True, False]]]),
        population_names=np.array(["AFR", "EUR"]),
        output_ploidy=2,
    )

    with np.load(truth_path) as truth:
        np.testing.assert_array_equal(truth["population_names"], ["AFR", "EUR"])
        assert int(truth["output_ploidy"]) == 2
        assert truth["labels_c"].size == 0


def test_batch_genotype_formats_require_diploid_layout():
    with pytest.raises(ValueError, match="Diploid batch output"):
        _batch_genotypes_to_snp_layout(np.zeros((2, 3), dtype=np.int8))


def test_batch_pickle_output_keeps_npz_truth_sidecar(tmp_path, monkeypatch):
    source = _source_snpobject()
    metadata_path = tmp_path / "founders.tsv"
    metadata_path.write_text("Sample\tPopulation\nfounder1\tAFR\nfounder2\tEUR\n")

    class FakeReader:
        def read(self, genotype_mode):
            assert genotype_mode == "phased"
            return source

    class FakeSimulator:
        population_names = np.array(["AFR", "EUR"])

        def __init__(self, **kwargs):
            assert kwargs["snp_data"] is source

        def simulate(self, *, batch_size, **kwargs):
            assert batch_size == 4
            return (
                np.array(
                    [
                        [0, 1],
                        [1, 1],
                        [1, 0],
                        [0, 0],
                    ],
                    dtype=np.float32,
                ),
                None,
                None,
                None,
            )

    fake_simulator_module = types.ModuleType("snputils.simulation.simulator")
    fake_simulator_module.OnlineSimulator = FakeSimulator
    monkeypatch.setitem(sys.modules, "snputils.simulation.simulator", fake_simulator_module)
    monkeypatch.setattr(simulator_cli, "SNPReader", lambda path: FakeReader())

    args = argparse.Namespace(
        ancestry_proportions=None,
        batch_size=2,
        device="cpu",
        diploid_output=True,
        expand_haplotypes=False,
        fixed_generations=False,
        genetic_map=None,
        metadata=str(metadata_path),
        n_batches=1,
        n_individuals=None,
        num_generations=10,
        output_dir=str(tmp_path / "output"),
        output_format="pkl",
        output_prefix=None,
        sample_prefix="SIM",
        snp="founders.pgen",
        store_latlon_as_nvec=False,
        verbose=False,
        window_size=1000,
    )

    assert simulator_cli.run_simulator_command(args) == 0

    pickle_path = tmp_path / "output" / "batch_0001.pkl"
    truth_path = tmp_path / "output" / "batch_0001.labels.npz"
    assert pickle_path.exists()
    assert truth_path.exists()
    with pickle_path.open("rb") as handle:
        observed = pickle.load(handle)
    assert observed.genotypes.shape == (2, 2, 2)
    np.testing.assert_array_equal(observed.samples, ["SIMB0001_000001", "SIMB0001_000002"])
