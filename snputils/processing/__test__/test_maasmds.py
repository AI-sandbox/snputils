import numpy as np
import pandas as pd

from snputils.ancestry.genobj.local import LocalAncestryObject
from snputils.processing._utils.gen_tools import process_calldata_gt, process_labels_weights
from snputils.processing._utils.mds_distance import (
    binary_intersection,
    conversion_metrics,
    distance_mat,
    distance_overlap,
    mds_transform,
    overlap_blocks,
)
from snputils.processing.maasmds import maasMDS
from snputils.snp.genobj.snpobj import SNPObject


def _make_snpobj(
    samples,
    variant_ids,
    sample_genotypes,
    chrom="1",
    pos_start=100,
    variants_ref=None,
    variants_alt=None,
):
    sample_genotypes = np.asarray(sample_genotypes, dtype=np.int8)
    calldata_gt = np.repeat(sample_genotypes[:, :, None], 2, axis=2)
    variants_pos = np.arange(pos_start, pos_start + sample_genotypes.shape[0], dtype=np.int64)
    variants_chrom = np.full(sample_genotypes.shape[0], chrom)
    if variants_ref is None:
        variants_ref = np.full(sample_genotypes.shape[0], "A")
    if variants_alt is None:
        variants_alt = np.full(sample_genotypes.shape[0], "G")
    return SNPObject(
        calldata_gt=calldata_gt,
        samples=np.asarray(samples),
        variants_id=np.asarray(variant_ids),
        variants_pos=variants_pos,
        variants_chrom=variants_chrom,
        variants_ref=np.asarray(variants_ref),
        variants_alt=np.asarray(variants_alt),
    )


def _make_laiobj(samples, chrom="1", start=50, end=500):
    haplotypes = [f"{sample}.0" for sample in samples] + [f"{sample}.1" for sample in samples]
    haplotypes = np.array(haplotypes).reshape(2, len(samples)).T.reshape(-1).tolist()
    lai = np.zeros((1, len(samples) * 2), dtype=np.int16)
    return LocalAncestryObject(
        haplotypes=haplotypes,
        lai=lai,
        samples=list(samples),
        ancestry_map={"0": "Target"},
        chromosomes=np.array([chrom]),
        physical_pos=np.array([[start, end]], dtype=np.int64),
    )


def _write_labels(labels_path, rows):
    pd.DataFrame(rows, columns=["indID", "label"]).to_csv(labels_path, sep="\t", index=False)


def _pairwise_distances(embedding):
    diffs = embedding[:, None, :] - embedding[None, :, :]
    return np.sqrt(np.sum(diffs * diffs, axis=2))


def _assert_embedding_equivalent(actual, expected):
    np.testing.assert_allclose(
        _pairwise_distances(actual),
        _pairwise_distances(expected),
        atol=1e-6,
    )


def _build_three_array_inputs():
    snpobjs = [
        _make_snpobj(
            samples=["A1", "A2", "A3"],
            variant_ids=["rs1", "rs2", "rs3", "rs4"],
            sample_genotypes=[
                [0, 0, 1],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0],
            ],
            pos_start=100,
        ),
        _make_snpobj(
            samples=["B1", "B2", "B3", "B4"],
            variant_ids=["rs2", "rs3", "rs4", "rs5"],
            sample_genotypes=[
                [0, 1, 1, 0],
                [1, 1, 0, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 1],
            ],
            pos_start=200,
        ),
        _make_snpobj(
            samples=["C1", "C2", "C3"],
            variant_ids=["rs1", "rs3", "rs4", "rs5"],
            sample_genotypes=[
                [1, 0, 1],
                [0, 1, 1],
                [1, 0, 0],
                [0, 1, 1],
            ],
            pos_start=300,
        ),
    ]
    laiobjs = [_make_laiobj(snpobj.samples) for snpobj in snpobjs]
    labels_rows = [
        ("A1", "Pop1"),
        ("A2", "Pop2"),
        ("A3", "Pop1"),
        ("B1", "Pop2"),
        ("B2", "Pop2"),
        ("B3", "Pop1"),
        ("B4", "Pop1"),
        ("C1", "Pop1"),
        ("C2", "Pop2"),
        ("C3", "Pop1"),
    ]
    return snpobjs, laiobjs, labels_rows


def _manual_multi_array_run(
    snpobjs,
    laiobjs,
    labels_file,
    ancestry,
    groups_to_remove=None,
):
    num_arrays = len(snpobjs)
    groups_to_remove = maasMDS._normalize_groups_to_remove(groups_to_remove, num_arrays)

    masks = []
    variants_id_list = []
    haplotypes_list = []
    groups = []
    weights = []

    variants_ref_map = {}
    for array_index, (snpobj, laiobj) in enumerate(zip(snpobjs, laiobjs)):
        mask, variants_id, haplotypes, variants_ref_map = process_calldata_gt(
            snpobj,
            laiobj,
            ancestry,
            average_strands=True,
            force_nan_incomplete_strands=False,
            is_masked=True,
            rsid_or_chrompos=1,
            variants_ref_map=variants_ref_map,
        )
        mask, haplotypes, current_groups, current_weights = process_labels_weights(
            labels_file,
            mask,
            variants_id,
            haplotypes,
            True,
            ancestry,
            0,
            True,
            groups_to_remove[array_index],
            False,
            False,
            "unused.npz",
        )
        masks.append(mask)
        variants_id_list.append(np.asarray(variants_id))
        haplotypes_list.append(np.asarray(haplotypes))
        groups.append(np.asarray(current_groups))
        weights.append(np.asarray(current_weights))

    groups = np.concatenate(groups)
    weights = np.concatenate(weights)

    if num_arrays > 1:
        ref_row = 0
        ref_col = 1
        binary = binary_intersection(variants_id_list)
        overlap = overlap_blocks(
            ancestry,
            ref_col,
            ref_row,
            num_arrays,
            variants_id_list,
            binary,
            masks,
        )
        conversion, intercept = conversion_metrics(
            ancestry,
            ref_col,
            ref_row,
            num_arrays,
            variants_id_list,
            binary,
            masks,
            "AP",
        )
        distance_list = distance_overlap(
            ref_col,
            ref_row,
            num_arrays,
            overlap,
            conversion,
            intercept,
            "AP",
        )
        ind_id_arg = haplotypes_list
    else:
        distance_list = [[distance_mat(first=masks[0][ancestry], dist_func="AP")]]
        ind_id_arg = haplotypes_list[0]

    transformed, haplotypes, groups, array_labels = mds_transform(
        distance_list,
        groups,
        weights,
        ind_id_arg,
        2,
        num_arrays=num_arrays,
        imputation_method="mean",
        return_metadata=True,
    )
    return transformed, haplotypes, groups, array_labels, variants_id_list


def test_maasmds_multi_array_matches_manual_pipeline(tmp_path):
    snpobjs, laiobjs, labels_rows = _build_three_array_inputs()
    labels_file = tmp_path / "labels.tsv"
    _write_labels(labels_file, labels_rows)

    expected = _manual_multi_array_run(
        snpobjs,
        laiobjs,
        str(labels_file),
        ancestry=0,
        groups_to_remove={2: ["Pop2"]},
    )

    model = maasMDS(
        snpobj=snpobjs,
        laiobj=laiobjs,
        labels_file=str(labels_file),
        ancestry=0,
        is_masked=True,
        average_strands=True,
        groups_to_remove={2: ["Pop2"]},
        min_percent_snps=0,
        distance_type="AP",
        rsid_or_chrompos=1,
    )

    expected_embedding, expected_haplotypes, _, expected_array_labels, expected_variants = expected
    _assert_embedding_equivalent(model.X_new_, expected_embedding)
    assert model.haplotypes_ == expected_haplotypes.tolist()
    np.testing.assert_array_equal(model.array_labels_, expected_array_labels)
    assert "B1" not in model.samples_
    assert "B2" not in model.samples_
    assert len(model.variants_id_) == 3
    for observed, expected_ids in zip(model.variants_id_, expected_variants):
        np.testing.assert_array_equal(observed, expected_ids)


def test_maasmds_groups_to_remove_supports_flat_list(tmp_path):
    snpobjs, laiobjs, labels_rows = _build_three_array_inputs()
    labels_file = tmp_path / "labels.tsv"
    _write_labels(labels_file, labels_rows)

    model = maasMDS(
        snpobj=snpobjs,
        laiobj=laiobjs,
        labels_file=str(labels_file),
        ancestry=0,
        is_masked=True,
        average_strands=True,
        groups_to_remove=["Pop2"],
        min_percent_snps=0,
        distance_type="AP",
        rsid_or_chrompos=1,
    )

    assert set(model.samples_) == {"A1", "A3", "B3", "B4", "C1", "C3"}


def test_maasmds_save_and_load_masks_round_trip(tmp_path):
    snpobjs, laiobjs, labels_rows = _build_three_array_inputs()
    labels_file = tmp_path / "labels.tsv"
    masks_file = tmp_path / "masks.npz"
    _write_labels(labels_file, labels_rows)

    saved = maasMDS(
        snpobj=snpobjs,
        laiobj=laiobjs,
        labels_file=str(labels_file),
        ancestry=0,
        is_masked=True,
        average_strands=True,
        min_percent_snps=0,
        distance_type="AP",
        rsid_or_chrompos=1,
        save_masks=True,
        masks_file=masks_file,
    )

    loaded = maasMDS(
        snpobj=snpobjs,
        laiobj=laiobjs,
        labels_file=str(labels_file),
        ancestry=0,
        is_masked=True,
        average_strands=True,
        min_percent_snps=0,
        distance_type="AP",
        rsid_or_chrompos=1,
        load_masks=True,
        masks_file=masks_file,
    )

    assert masks_file.exists()
    _assert_embedding_equivalent(saved.X_new_, loaded.X_new_)
    assert loaded.haplotypes_ == saved.haplotypes_
    np.testing.assert_array_equal(loaded.array_labels_, saved.array_labels_)


def test_process_calldata_gt_harmonizes_flipped_reference_alleles_across_arrays():
    array1 = _make_snpobj(
        samples=["A1", "A2"],
        variant_ids=["rs1", "rs2", "rs3"],
        sample_genotypes=[
            [0, 1],
            [1, 0],
            [0, 0],
        ],
        variants_ref=["A", "C", "G"],
        variants_alt=["G", "T", "A"],
    )
    array2_flipped = _make_snpobj(
        samples=["B1", "B2"],
        variant_ids=["rs1", "rs2", "rs3"],
        sample_genotypes=[
            [1, 0],
            [0, 1],
            [1, 1],
        ],
        variants_ref=["G", "T", "A"],
        variants_alt=["A", "C", "G"],
    )
    array2_expected = _make_snpobj(
        samples=["B1", "B2"],
        variant_ids=["rs1", "rs2", "rs3"],
        sample_genotypes=[
            [0, 1],
            [1, 0],
            [0, 0],
        ],
        variants_ref=["A", "C", "G"],
        variants_alt=["G", "T", "A"],
    )
    laiobjs = [_make_laiobj(["A1", "A2"]), _make_laiobj(["B1", "B2"])]
    variants_ref_map = {}
    _, _, _, variants_ref_map = process_calldata_gt(
        array1,
        laiobjs[0],
        0,
        average_strands=True,
        force_nan_incomplete_strands=False,
        is_masked=True,
        rsid_or_chrompos=1,
        variants_ref_map=variants_ref_map,
    )
    flipped_mask, flipped_variants, flipped_haplotypes, variants_ref_map = process_calldata_gt(
        array2_flipped,
        laiobjs[1],
        0,
        average_strands=True,
        force_nan_incomplete_strands=False,
        is_masked=True,
        rsid_or_chrompos=1,
        variants_ref_map=variants_ref_map,
    )
    expected_mask, expected_variants, expected_haplotypes, _ = process_calldata_gt(
        array2_expected,
        laiobjs[1],
        0,
        average_strands=True,
        force_nan_incomplete_strands=False,
        is_masked=True,
        rsid_or_chrompos=1,
        variants_ref_map={},
    )

    np.testing.assert_array_equal(flipped_variants, expected_variants)
    np.testing.assert_array_equal(flipped_haplotypes, expected_haplotypes)
    np.testing.assert_allclose(flipped_mask[0], expected_mask[0], atol=1e-8)
