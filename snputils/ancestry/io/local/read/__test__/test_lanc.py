from pathlib import Path

import numpy as np
import pytest

from snputils.ancestry.genobj.local import LocalAncestryObject
from snputils.ancestry.io.local.read import LAIReader, LANCReader, read_lai, read_lanc
from snputils.ancestry.io.local.write import LANCWriter


def _write_lanc(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "6 3",
                "2:01 6:00",
                "3:11 5:10 6:10",
                "6:22",
            ]
        ),
        encoding="utf-8",
    )


def _write_pvar(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "#CHROM\tPOS\tID\tREF\tALT\tCM",
                "1\t101\trs1\tA\tC\t0.1",
                "1\t205\trs2\tG\tT\t0.2",
                "2\t300\trs3\tC\tG\t0.3",
                "2\t450\trs4\tT\tA\t0.4",
                "3\t500\trs5\tG\tA\t0.5",
                "3\t650\trs6\tC\tT\t0.6",
            ]
        ),
        encoding="utf-8",
    )


def _write_psam(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "#FID\tIID\tSEX",
                "P1\tS1\t1",
                "P1\tS2\t2",
                "P2\tS3\tNA",
            ]
        ),
        encoding="utf-8",
    )


def test_lanc_reader_reads_sidecar_metadata_and_autodetects(tmp_path: Path):
    lanc_path = tmp_path / "toy.lanc"
    _write_lanc(lanc_path)
    _write_pvar(tmp_path / "toy.pvar")
    _write_psam(tmp_path / "toy.psam")

    lai = read_lai(lanc_path)

    assert isinstance(LAIReader(lanc_path), LANCReader)
    assert lai.samples == ["S1", "S2", "S3"]
    assert lai.haplotypes == [
        "S1.0",
        "S1.1",
        "S2.0",
        "S2.1",
        "S3.0",
        "S3.1",
    ]
    assert lai.ancestry_map is None
    np.testing.assert_array_equal(lai.chromosomes, np.array(["1", "1", "2", "2", "3", "3"], dtype=object))
    np.testing.assert_array_equal(
        lai.physical_pos,
        np.array([[101, 101], [205, 205], [300, 300], [450, 450], [500, 500], [650, 650]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        lai.centimorgan_pos,
        np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.5, 0.5], [0.6, 0.6]], dtype=float),
    )
    np.testing.assert_array_equal(lai.window_sizes, np.ones(6, dtype=np.int64))
    np.testing.assert_array_equal(
        lai.lai,
        np.array(
            [
                [0, 1, 1, 1, 2, 2],
                [0, 1, 1, 1, 2, 2],
                [0, 0, 1, 1, 2, 2],
                [0, 0, 1, 0, 2, 2],
                [0, 0, 1, 0, 2, 2],
                [0, 0, 1, 0, 2, 2],
            ],
            dtype=np.uint8,
        ),
    )


def test_lanc_iter_windows_matches_full_read_and_subsets_samples(tmp_path: Path):
    lanc_path = tmp_path / "toy.lanc"
    _write_lanc(lanc_path)
    _write_pvar(tmp_path / "toy.pvar")
    _write_psam(tmp_path / "toy.psam")
    full = read_lanc(lanc_path)

    chunks = list(LANCReader(lanc_path).iter_windows(chunk_size=4, sample_indices=np.array([1])))

    np.testing.assert_array_equal(
        np.concatenate([chunk["window_indexes"] for chunk in chunks]),
        np.arange(full.n_windows, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.concatenate([chunk["physical_pos"] for chunk in chunks]),
        full.physical_pos,
    )
    np.testing.assert_array_equal(
        np.concatenate([chunk["lai"] for chunk in chunks]),
        full.lai[:, 2:4],
    )


def test_lanc_reader_accepts_explicit_sidecar_paths(tmp_path: Path):
    lanc_path = tmp_path / "toy.lanc"
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    _write_lanc(lanc_path)
    _write_pvar(meta_dir / "alt.pvar")
    _write_psam(meta_dir / "alt.psam")

    lai = read_lanc(lanc_path, pvar_file=meta_dir / "alt.pvar", psam_file=meta_dir / "alt.psam")

    assert lai.samples == ["S1", "S2", "S3"]
    np.testing.assert_array_equal(lai.physical_pos[:, 0], np.array([101, 205, 300, 450, 500, 650], dtype=np.int64))


def test_lanc_reader_warns_and_falls_back_without_sidecars(tmp_path: Path):
    lanc_path = tmp_path / "toy.lanc"
    _write_lanc(lanc_path)

    with pytest.warns(UserWarning, match="Please specify pvar_file, psam_file"):
        lai = read_lanc(lanc_path)

    assert lai.samples == ["sample_0", "sample_1", "sample_2"]
    assert lai.chromosomes is None
    assert lai.physical_pos is None


def test_lanc_reader_warns_partially_when_one_sidecar_is_missing(tmp_path: Path):
    lanc_path = tmp_path / "toy.lanc"
    _write_lanc(lanc_path)
    _write_psam(tmp_path / "toy.psam")

    with pytest.warns(UserWarning, match="Please specify pvar_file"):
        lai = read_lanc(lanc_path)

    assert lai.samples == ["S1", "S2", "S3"]
    assert lai.chromosomes is None
    assert lai.physical_pos is None


def test_lanc_reader_asserts_psam_count_matches_header(tmp_path: Path):
    lanc_path = tmp_path / "toy.lanc"
    _write_lanc(lanc_path)
    _write_pvar(tmp_path / "toy.pvar")
    (tmp_path / "toy.psam").write_text("#IID\nS1\nS2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="PSAM sample count"):
        read_lanc(lanc_path)


def test_lanc_reader_asserts_pvar_count_matches_header(tmp_path: Path):
    lanc_path = tmp_path / "toy.lanc"
    _write_lanc(lanc_path)
    (tmp_path / "toy.pvar").write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "#CHROM\tPOS\tID\tREF\tALT",
                "1\t101\trs1\tA\tC",
            ]
        ),
        encoding="utf-8",
    )
    _write_psam(tmp_path / "toy.psam")

    with pytest.raises(ValueError, match="PVAR variant count"):
        read_lanc(lanc_path)


def test_lanc_writer_roundtrips_matrix_and_save_dispatch(tmp_path: Path):
    lai_matrix = np.array(
        [
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [2, 2, 1, 0],
            [2, 2, 1, 0],
        ],
        dtype=np.uint8,
    )
    laiobj = LocalAncestryObject(
        haplotypes=["S1.0", "S1.1", "S2.0", "S2.1"],
        samples=["S1", "S2"],
        ancestry_map={"0": "AFR", "1": "EUR", "2": "AMR"},
        chromosomes=np.array(["1", "1", "1", "1"], dtype=object),
        physical_pos=np.array([[10, 10], [20, 20], [30, 30], [40, 40]], dtype=np.int64),
        lai=lai_matrix,
    )

    direct_path = tmp_path / "direct.lanc"
    LANCWriter(laiobj, direct_path).write()
    with pytest.warns(UserWarning, match="Please specify pvar_file, psam_file"):
        roundtrip_direct = read_lanc(direct_path)
    np.testing.assert_array_equal(roundtrip_direct.lai, lai_matrix)
    assert roundtrip_direct.samples == ["sample_0", "sample_1"]

    save_path = tmp_path / "saved"
    laiobj.save(save_path.with_suffix(".lanc"))
    with pytest.warns(UserWarning, match="Please specify pvar_file, psam_file"):
        roundtrip_saved = read_lanc(save_path.with_suffix(".lanc"))
    np.testing.assert_array_equal(roundtrip_saved.lai, lai_matrix)


def test_lanc_writer_rejects_multi_digit_ancestry_codes(tmp_path: Path):
    laiobj = LocalAncestryObject(
        haplotypes=["S1.0", "S1.1"],
        samples=["S1"],
        lai=np.array([[10, 0]], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="single-digit ancestry codes"):
        LANCWriter(laiobj, tmp_path / "bad.lanc").write()
