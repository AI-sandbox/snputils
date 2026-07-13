import importlib

import numpy as np
import pandas as pd

from snputils.processing.mdpca import mdPCA


def test_fit_transform_uses_call_time_arguments(monkeypatch):
    module = importlib.import_module("snputils.processing.mdpca")
    old_snp = object()
    old_lai = type("OldLAI", (), {"ancestry_map": {"0": "Old"}})()
    new_snp = object()
    new_lai = type("NewLAI", (), {"ancestry_map": {"1": "Target"}})()
    labels = pd.DataFrame({"indID": ["sample"], "label": ["population"]})
    observed = {}

    def fake_process_genotypes(snpobj, laiobj, ancestry, average_haplotypes, *args):
        observed.update(
            snpobj=snpobj,
            laiobj=laiobj,
            ancestry=ancestry,
            average_haplotypes=average_haplotypes,
        )
        return {1: np.zeros((2, 1))}, ["v1", "v2"], np.array(["sample_A"]), None

    def fake_process_labels(
        labels_file,
        mask,
        variants_id,
        haplotypes,
        average_haplotypes,
        ancestry,
        *args,
    ):
        observed.update(
            labels=labels_file,
            labels_average_haplotypes=average_haplotypes,
            labels_ancestry=ancestry,
        )
        return mask, haplotypes, np.array(["population"]), np.ones(1)

    monkeypatch.setattr(module, "process_genotypes", fake_process_genotypes)
    monkeypatch.setattr(module, "process_labels_weights", fake_process_labels)

    model = mdPCA(snpobj=old_snp, laiobj=old_lai, is_masked=True, ancestry=0)
    monkeypatch.setattr(model, "_run_cov_matrix", lambda matrix, weights: np.zeros((1, 1)))

    model.fit_transform(
        snpobj=new_snp,
        laiobj=new_lai,
        labels=labels,
        ancestry="Target",
        average_haplotypes=True,
    )

    assert observed == {
        "snpobj": new_snp,
        "laiobj": new_lai,
        "ancestry": 1,
        "average_haplotypes": True,
        "labels": labels,
        "labels_average_haplotypes": True,
        "labels_ancestry": 1,
    }
    assert model.snpobj is new_snp
    assert model.laiobj is new_lai
    assert model.labels_file is labels
    assert model.ancestry == 1
    assert model.average_haplotypes is True
