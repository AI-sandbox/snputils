import numpy as np
import pandas as pd

from snputils.stats import f2, f3, f4, d_stat, f4_ratio


def _toy_data():
    # Build simple AFs where populations A, B, C, D have clear relationships
    # n_snps=6
    # Pop A and B close; C different; D outgroup-like
    afs = np.array([
        [0.1, 0.1, 0.9, 0.9],
        [0.2, 0.2, 0.8, 0.8],
        [0.3, 0.3, 0.7, 0.7],
        [0.4, 0.4, 0.6, 0.6],
        [0.5, 0.5, 0.5, 0.5],
        [0.6, 0.55, 0.4, 0.4],
    ])
    counts = np.full_like(afs, 20)
    pops = ["A", "B", "C", "D"]
    return afs, counts, pops


def test_f2_basic():
    afs, counts, pops = _toy_data()
    res = f2((afs, counts, pops), block_size=2)
    # Symmetry on diagonal pairs
    ab = res[(res.pop1 == "A") & (res.pop2 == "B")].iloc[0]
    assert np.isfinite(ab.est)
    assert ab.n_blocks == 3
    assert ab.n_snps == 6


def test_f3_basic():
    afs, counts, pops = _toy_data()
    res = f3((afs, counts, pops), target=["A"], ref1=["B"], ref2=["C"], block_size=3)
    assert res.shape[0] == 1
    row = res.iloc[0]
    assert row.target == "A"
    assert np.isfinite(row.est)


def test_f4_and_d_basic():
    afs, counts, pops = _toy_data()
    quads = dict(a=["A"], b=["B"], c=["C"], d=["D"])
    res4 = f4((afs, counts, pops), **quads, block_size=2)
    resd = d_stat((afs, counts, pops), **quads, block_size=2)
    assert res4.shape[0] == 1
    assert resd.shape[0] == 1
    assert np.isfinite(res4.est.iloc[0])
    assert np.isfinite(resd.est.iloc[0])


def test_f4_ratio_basic():
    afs, counts, pops = _toy_data()
    num = [("A", "B", "C", "D")]
    den = [("A", "C", "B", "D")]
    res = f4_ratio((afs, counts, pops), num=num, den=den, block_size=2)
    assert res.shape[0] == 1
    assert np.isfinite(res.est.iloc[0])


