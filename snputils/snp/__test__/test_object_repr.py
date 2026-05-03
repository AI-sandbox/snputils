import numpy as np
import pandas as pd
import pytest

from snputils.ancestry.genobj.local import LocalAncestryObject
from snputils.ancestry.genobj.wide import GlobalAncestryObject
from snputils.ibd.genobj.ibdobj import IBDObject
from snputils.phenotype.genobj.multi_phenobj import MultiPhenotypeObject
from snputils.phenotype.genobj.phenobj import PhenotypeObject
from snputils.snp.genobj.snpobj import SNPObject


def test_snpobject_repr_and_shape_are_informative():
    snp = SNPObject(
        calldata_gt=np.zeros((3, 2, 2), dtype=np.int8),
        samples=np.array(["S0", "S1"]),
        variants_chrom=np.array(["1", "1", "2"]),
        variants_pos=np.array([10, 20, 30]),
    )

    assert snp.shape == (3, 2, 2)
    assert str(snp) == repr(snp)
    assert "SNPObject" in repr(snp)
    assert "n_snps=3" in repr(snp)
    assert "n_samples=2" in repr(snp)
    assert "calldata_gt_shape=(3, 2, 2)" in repr(snp)


def test_snpobject_metadata_only_shape_uses_available_dimensions():
    snp = SNPObject(
        samples=np.array(["S0", "S1"]),
        variants_chrom=np.array(["1", "1", "2"]),
        variants_pos=np.array([10, 20, 30]),
    )

    assert snp.shape == (3, 2)
    assert "shape=(3, 2)" in repr(snp)


def test_local_ancestry_repr_and_shape_are_informative():
    laiobj = LocalAncestryObject(
        haplotypes=["S0.0", "S0.1", "S1.0", "S1.1"],
        lai=np.array([[0, 1, 1, 0], [1, 1, 0, 0]], dtype=np.int8),
        samples=["S0", "S1"],
        chromosomes=np.array(["1", "1"]),
        physical_pos=np.array([[1, 10], [11, 20]]),
    )

    assert laiobj.shape == (2, 4)
    assert "LocalAncestryObject" in repr(laiobj)
    assert "n_windows=2" in repr(laiobj)
    assert "n_haplotypes=4" in repr(laiobj)


def test_global_ancestry_repr_and_shape_are_informative():
    gai = GlobalAncestryObject(
        Q=np.array([[0.25, 0.75], [0.8, 0.2]]),
        P=np.array([[0.1, 0.9], [0.6, 0.4], [0.3, 0.7]]),
        samples=["S0", "S1"],
        snps=["rs1", "rs2", "rs3"],
    )

    assert gai.shape == (2, 2)
    assert "GlobalAncestryObject" in repr(gai)
    assert "n_snps=3" in repr(gai)
    assert "Q_shape=(2, 2)" in repr(gai)
    assert "P_shape=(3, 2)" in repr(gai)


def test_phenotype_repr_and_shape_are_informative():
    phen = PhenotypeObject(
        samples=["S0", "S1", "S2"],
        values=[0, 1, 1],
        phenotype_name="case_control",
    )

    assert phen.shape == (3,)
    assert "PhenotypeObject" in repr(phen)
    assert "trait_type='binary'" in repr(phen)
    assert "n_cases=2" in repr(phen)
    assert "n_controls=1" in repr(phen)


def test_multi_phenotype_repr_and_shape_are_informative():
    phen = MultiPhenotypeObject(
        pd.DataFrame(
            {
                "IID": ["S0", "S1"],
                "height": [170.0, 180.0],
                "case_control": [0, 1],
            }
        )
    )

    assert phen.shape == (2, 3)
    assert phen.n_phenotypes == 2
    assert "MultiPhenotypeObject" in repr(phen)
    assert "n_samples=2" in repr(phen)
    assert "n_phenotypes=2" in repr(phen)


def test_ibd_repr_and_shape_are_informative():
    ibd = IBDObject(
        sample_id_1=np.array(["S0", "S1"]),
        haplotype_id_1=np.array([1, 2]),
        sample_id_2=np.array(["S1", "S2"]),
        haplotype_id_2=np.array([2, 1]),
        chrom=np.array(["1", "2"]),
        start=np.array([100, 200]),
        end=np.array([150, 250]),
        length_cm=np.array([1.0, 2.0]),
    )

    assert ibd.shape == (2,)
    assert ibd.n_samples == 3
    assert "IBDObject" in repr(ibd)
    assert "n_segments=2" in repr(ibd)
    assert "n_chromosomes=2" in repr(ibd)


def test_grg_repr_and_shape_are_informative_without_loaded_graph():
    pytest.importorskip("pygrgl")
    from snputils.snp.genobj.grgobj import GRGObject

    grg = GRGObject(filename="example.grg", mutable=False)

    assert grg.shape == (None, None)
    assert "GRGObject" in repr(grg)
    assert "filename='example.grg'" in repr(grg)
    assert "loaded=False" in repr(grg)
