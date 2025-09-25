import numpy as np
import pytest

from snputils.ibd.genobj.ibdobj import IBDObject
from snputils.ancestry.genobj.local import LocalAncestryObject


def _make_lai_object():
    # Three windows on chr1: [500-1500], [1501-2500], [2501-3500]
    physical_pos = np.array([[500, 1500], [1501, 2500], [2501, 3500]], dtype=int)
    chromosomes = np.array(["1", "1", "1"], dtype=object)
    # Simple cM map: 0.0-0.1, 0.1-0.4, 0.4-0.6
    centimorgan_pos = np.array([[0.0, 0.1], [0.1, 0.4], [0.4, 0.6]], dtype=float)
    haplotypes = ["A.0", "A.1", "B.0", "B.1"]
    # Initialize with zeros (ancestry 0)
    lai = np.zeros((3, 4), dtype=int)
    return physical_pos, chromosomes, centimorgan_pos, haplotypes, lai


def test_restrict_to_ancestry_hap_known_and_cm():
    physical_pos, chromosomes, centimorgan_pos, haplotypes, lai = _make_lai_object()

    # Set ancestry=1 for A.0 and B.1 in the middle window only
    lai[1, :] = [1, 0, 0, 1]

    laiobj = LocalAncestryObject(
        haplotypes=haplotypes,
        lai=lai,
        samples=["A", "B"],
        ancestry_map=None,
        window_sizes=np.array([1, 1, 1]),
        centimorgan_pos=centimorgan_pos,
        chromosomes=chromosomes,
        physical_pos=physical_pos,
    )

    # IBD segment covers all three windows; haplotypes are known: A hap1 (.0), B hap2 (.1)
    ibd = IBDObject(
        sample_id_1=np.array(["A"], dtype=object),
        haplotype_id_1=np.array([1], dtype=int),
        sample_id_2=np.array(["B"], dtype=object),
        haplotype_id_2=np.array([2], dtype=int),
        chrom=np.array(["1"], dtype=object),
        start=np.array([1000], dtype=int),
        end=np.array([3000], dtype=int),
        length_cm=np.array([5.0], dtype=float),
        segment_type=np.array(["IBD1"], dtype=object),
    )

    # Restrict to ancestry 1; expect one trimmed subsegment [1501, 2500] with cM ~= 0.3
    out = ibd.restrict_to_ancestry(laiobj=laiobj, ancestry=1)
    assert out is not None
    assert out.n_segments == 1
    assert out.start.tolist() == [1501]
    assert out.end.tolist() == [2500]
    assert out.chrom.tolist() == ["1"]
    assert out.sample_id_1.tolist() == ["A"]
    assert out.sample_id_2.tolist() == ["B"]
    # cM from middle window (0.1 -> 0.4)
    assert np.isclose(out.length_cm[0], 0.3, rtol=0, atol=1e-9)

    # If requiring both haps to match, nothing should remain because A.1 and B.0 are 0 in that window
    out2 = ibd.restrict_to_ancestry(laiobj=laiobj, ancestry=1, require_both_haplotypes=True)
    assert out2 is not None
    assert out2.n_segments == 0


def test_restrict_to_ancestry_unknown_haps_any_vs_both():
    physical_pos, chromosomes, centimorgan_pos, haplotypes, lai = _make_lai_object()
    # In the middle window, A.0 = 1, B.1 = 1; other haps 0
    lai[1, :] = [1, 0, 0, 1]

    laiobj = LocalAncestryObject(
        haplotypes=haplotypes,
        lai=lai,
        samples=["A", "B"],
        ancestry_map=None,
        window_sizes=np.array([1, 1, 1]),
        centimorgan_pos=centimorgan_pos,
        chromosomes=chromosomes,
        physical_pos=physical_pos,
    )

    # IBD segment with unknown hap IDs (ancIBD-like)
    ibd = IBDObject(
        sample_id_1=np.array(["A"], dtype=object),
        haplotype_id_1=np.array([-1], dtype=int),
        sample_id_2=np.array(["B"], dtype=object),
        haplotype_id_2=np.array([-1], dtype=int),
        chrom=np.array(["1"], dtype=object),
        start=np.array([1000], dtype=int),
        end=np.array([3000], dtype=int),
        length_cm=np.array([5.0], dtype=float),
        segment_type=np.array(["IBD1"], dtype=object),
    )

    out_any = ibd.restrict_to_ancestry(laiobj=laiobj, ancestry=1, require_both_haplotypes=False)
    assert out_any.n_segments == 1
    assert out_any.start.tolist() == [1501]
    assert out_any.end.tolist() == [2500]

    out_both = ibd.restrict_to_ancestry(laiobj=laiobj, ancestry=1, require_both_haplotypes=True)
    # A.1 and B.0 are 0, so requiring both per individual should drop it
    assert out_both.n_segments == 0


def test_restrict_to_ancestry_cm_fallback_scaling():
    # Same windows but without cM map; length should scale from original by bp fraction
    physical_pos = np.array([[500, 1500], [1501, 2500], [2501, 3500]], dtype=int)
    chromosomes = np.array(["1", "1", "1"], dtype=object)
    haplotypes = ["A.0", "A.1", "B.0", "B.1"]
    lai = np.zeros((3, 4), dtype=int)
    lai[1, :] = [1, 0, 0, 1]

    laiobj = LocalAncestryObject(
        haplotypes=haplotypes,
        lai=lai,
        samples=["A", "B"],
        ancestry_map=None,
        window_sizes=np.array([1, 1, 1]),
        centimorgan_pos=None,
        chromosomes=chromosomes,
        physical_pos=physical_pos,
    )

    # IBD covers 1000-3000 (length=2001 bp). The kept window is ~1000 bp.
    # If original cM is 6.0, expect ~3.0 cM after restricting.
    ibd = IBDObject(
        sample_id_1=np.array(["A"], dtype=object),
        haplotype_id_1=np.array([1], dtype=int),
        sample_id_2=np.array(["B"], dtype=object),
        haplotype_id_2=np.array([2], dtype=int),
        chrom=np.array(["1"], dtype=object),
        start=np.array([1000], dtype=int),
        end=np.array([3000], dtype=int),
        length_cm=np.array([6.0], dtype=float),
        segment_type=np.array(["IBD1"], dtype=object),
    )

    out = ibd.restrict_to_ancestry(laiobj=laiobj, ancestry=1)
    assert out.n_segments == 1
    # Approx half of original cM
    assert np.isclose(out.length_cm[0], 3.0, rtol=0.05, atol=0.15)


def test_restrict_to_ancestry_strict_mode_drop_segment():
    physical_pos, chromosomes, centimorgan_pos, haplotypes, lai = _make_lai_object()

    # Set ancestry=1 for A.0 and B.1 in the middle window only
    lai[1, :] = [1, 0, 0, 1]

    laiobj = LocalAncestryObject(
        haplotypes=haplotypes,
        lai=lai,
        samples=["A", "B"],
        ancestry_map=None,
        window_sizes=np.array([1, 1, 1]),
        centimorgan_pos=centimorgan_pos,
        chromosomes=chromosomes,
        physical_pos=physical_pos,
    )

    # IBD segment covers all three windows; haplotypes are known: A hap1 (.0), B hap2 (.1)
    ibd = IBDObject(
        sample_id_1=np.array(["A"], dtype=object),
        haplotype_id_1=np.array([1], dtype=int),
        sample_id_2=np.array(["B"], dtype=object),
        haplotype_id_2=np.array([2], dtype=int),
        chrom=np.array(["1"], dtype=object),
        start=np.array([1000], dtype=int),
        end=np.array([3000], dtype=int),
        length_cm=np.array([5.0], dtype=float),
        segment_type=np.array(["IBD1"], dtype=object),
    )

    # Strict mode should drop entire segment because first and third windows lack target ancestry
    out_strict = ibd.restrict_to_ancestry(laiobj=laiobj, ancestry=1, method='strict')
    assert out_strict is not None
    assert out_strict.n_segments == 0

    # Clip mode should return trimmed segment for comparison
    out_clip = ibd.restrict_to_ancestry(laiobj=laiobj, ancestry=1, method='clip')
    assert out_clip is not None
    assert out_clip.n_segments == 1
    assert out_clip.start.tolist() == [1501]
    assert out_clip.end.tolist() == [2500]


def test_restrict_to_ancestry_strict_mode_keep_segment():
    physical_pos, chromosomes, centimorgan_pos, haplotypes, lai = _make_lai_object()

    # Set ancestry=1 for A.0 and B.1 in ALL windows
    lai[:, :] = [[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]]

    laiobj = LocalAncestryObject(
        haplotypes=haplotypes,
        lai=lai,
        samples=["A", "B"],
        ancestry_map=None,
        window_sizes=np.array([1, 1, 1]),
        centimorgan_pos=centimorgan_pos,
        chromosomes=chromosomes,
        physical_pos=physical_pos,
    )

    # IBD segment covers all three windows
    ibd = IBDObject(
        sample_id_1=np.array(["A"], dtype=object),
        haplotype_id_1=np.array([1], dtype=int),
        sample_id_2=np.array(["B"], dtype=object),
        haplotype_id_2=np.array([2], dtype=int),
        chrom=np.array(["1"], dtype=object),
        start=np.array([1000], dtype=int),
        end=np.array([3000], dtype=int),
        length_cm=np.array([5.0], dtype=float),
        segment_type=np.array(["IBD1"], dtype=object),
    )

    # Strict mode should keep entire segment with original coordinates
    out_strict = ibd.restrict_to_ancestry(laiobj=laiobj, ancestry=1, method='strict')
    assert out_strict is not None
    assert out_strict.n_segments == 1
    assert out_strict.start.tolist() == [1000]
    assert out_strict.end.tolist() == [3000]
    assert out_strict.length_cm.tolist() == [5.0]
    assert out_strict.sample_id_1.tolist() == ["A"]
    assert out_strict.sample_id_2.tolist() == ["B"]


def test_restrict_to_ancestry_strict_vs_clip_partial_overlap():
    physical_pos, chromosomes, centimorgan_pos, haplotypes, lai = _make_lai_object()

    # Set ancestry=1 for A.0 and B.1 in first window only
    lai[0, :] = [1, 0, 0, 1]

    laiobj = LocalAncestryObject(
        haplotypes=haplotypes,
        lai=lai,
        samples=["A", "B"],
        ancestry_map=None,
        window_sizes=np.array([1, 1, 1]),
        centimorgan_pos=centimorgan_pos,
        chromosomes=chromosomes,
        physical_pos=physical_pos,
    )

    # IBD segment overlaps first two windows only
    ibd = IBDObject(
        sample_id_1=np.array(["A"], dtype=object),
        haplotype_id_1=np.array([1], dtype=int),
        sample_id_2=np.array(["B"], dtype=object),
        haplotype_id_2=np.array([2], dtype=int),
        chrom=np.array(["1"], dtype=object),
        start=np.array([1000], dtype=int),
        end=np.array([2000], dtype=int),
        length_cm=np.array([3.0], dtype=float),
        segment_type=np.array(["IBD1"], dtype=object),
    )

    # Strict mode should drop segment because second window lacks target ancestry
    out_strict = ibd.restrict_to_ancestry(laiobj=laiobj, ancestry=1, method='strict')
    assert out_strict is not None
    assert out_strict.n_segments == 0

    # Clip mode should trim to first window portion
    out_clip = ibd.restrict_to_ancestry(laiobj=laiobj, ancestry=1, method='clip')
    assert out_clip is not None
    assert out_clip.n_segments == 1
    assert out_clip.start.tolist() == [1000]
    assert out_clip.end.tolist() == [1500]


def test_restrict_to_ancestry_strict_unknown_haplotypes():
    physical_pos, chromosomes, centimorgan_pos, haplotypes, lai = _make_lai_object()

    # Mixed pattern: first window has A.0=1, B.1=1; second has A.1=1, B.0=1; third has none
    lai[0, :] = [1, 0, 0, 1]
    lai[1, :] = [0, 1, 1, 0]

    laiobj = LocalAncestryObject(
        haplotypes=haplotypes,
        lai=lai,
        samples=["A", "B"],
        ancestry_map=None,
        window_sizes=np.array([1, 1, 1]),
        centimorgan_pos=centimorgan_pos,
        chromosomes=chromosomes,
        physical_pos=physical_pos,
    )

    # IBD segment with unknown haplotype IDs
    ibd = IBDObject(
        sample_id_1=np.array(["A"], dtype=object),
        haplotype_id_1=np.array([-1], dtype=int),
        sample_id_2=np.array(["B"], dtype=object),
        haplotype_id_2=np.array([-1], dtype=int),
        chrom=np.array(["1"], dtype=object),
        start=np.array([1000], dtype=int),
        end=np.array([3000], dtype=int),
        length_cm=np.array([5.0], dtype=float),
        segment_type=np.array(["IBD1"], dtype=object),
    )

    # With require_both_haplotypes=False, at least one hap must match per individual
    # This should fail strict mode because third window has no matches
    out_any = ibd.restrict_to_ancestry(laiobj=laiobj, ancestry=1, method='strict', require_both_haplotypes=False)
    assert out_any is not None
    assert out_any.n_segments == 0

    # With require_both_haplotypes=True, should also fail because no window has both haps matching for both individuals
    out_both = ibd.restrict_to_ancestry(laiobj=laiobj, ancestry=1, method='strict', require_both_haplotypes=True)
    assert out_both is not None
    assert out_both.n_segments == 0


def test_restrict_to_ancestry_strict_length_filters():
    physical_pos, chromosomes, centimorgan_pos, haplotypes, lai = _make_lai_object()

    # Set ancestry=1 for A.0 and B.1 in ALL windows
    lai[:, :] = [[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]]

    laiobj = LocalAncestryObject(
        haplotypes=haplotypes,
        lai=lai,
        samples=["A", "B"],
        ancestry_map=None,
        window_sizes=np.array([1, 1, 1]),
        centimorgan_pos=centimorgan_pos,
        chromosomes=chromosomes,
        physical_pos=physical_pos,
    )

    # Short IBD segment (101 bp, 0.05 cM)
    ibd = IBDObject(
        sample_id_1=np.array(["A"], dtype=object),
        haplotype_id_1=np.array([1], dtype=int),
        sample_id_2=np.array(["B"], dtype=object),
        haplotype_id_2=np.array([2], dtype=int),
        chrom=np.array(["1"], dtype=object),
        start=np.array([1000], dtype=int),
        end=np.array([1100], dtype=int),
        length_cm=np.array([0.05], dtype=float),
        segment_type=np.array(["IBD1"], dtype=object),
    )

    # No filters: should keep segment
    out_no_filter = ibd.restrict_to_ancestry(laiobj=laiobj, ancestry=1, method='strict')
    assert out_no_filter is not None
    assert out_no_filter.n_segments == 1

    # With min_bp=500: should drop segment (101 < 500)
    out_bp_filter = ibd.restrict_to_ancestry(laiobj=laiobj, ancestry=1, method='strict', min_bp=500)
    assert out_bp_filter is not None
    assert out_bp_filter.n_segments == 0

    # With min_cm=0.1: should drop segment (0.05 < 0.1)
    out_cm_filter = ibd.restrict_to_ancestry(laiobj=laiobj, ancestry=1, method='strict', min_cm=0.1)
    assert out_cm_filter is not None
    assert out_cm_filter.n_segments == 0


def test_restrict_to_ancestry_method_validation():
    physical_pos, chromosomes, centimorgan_pos, haplotypes, lai = _make_lai_object()

    laiobj = LocalAncestryObject(
        haplotypes=haplotypes,
        lai=lai,
        samples=["A", "B"],
        ancestry_map=None,
        window_sizes=np.array([1, 1, 1]),
        centimorgan_pos=centimorgan_pos,
        chromosomes=chromosomes,
        physical_pos=physical_pos,
    )

    ibd = IBDObject(
        sample_id_1=np.array(["A"], dtype=object),
        haplotype_id_1=np.array([1], dtype=int),
        sample_id_2=np.array(["B"], dtype=object),
        haplotype_id_2=np.array([2], dtype=int),
        chrom=np.array(["1"], dtype=object),
        start=np.array([1000], dtype=int),
        end=np.array([3000], dtype=int),
        length_cm=np.array([5.0], dtype=float),
        segment_type=np.array(["IBD1"], dtype=object),
    )

    # Invalid method should raise ValueError
    with pytest.raises(ValueError, match="Method must be 'strict' or 'clip', got 'invalid'"):
        ibd.restrict_to_ancestry(laiobj=laiobj, ancestry=1, method='invalid')
