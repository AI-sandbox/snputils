from pathlib import Path

import numpy as np

from snputils.ancestry.io.local.read import MSPReader
from snputils.ancestry.io.local.read.__test__.fixtures import make_small_dataset, write_msp


def test_iter_windows_matches_full_read(tmp_path: Path):
    sample_ids, lai, chromosomes, starts, ends, ancestry_map = make_small_dataset(
        n_samples=6,
        n_windows=11,
        seed=7,
    )
    msp_path = tmp_path / "toy.msp"
    write_msp(msp_path, sample_ids, lai, chromosomes, starts, ends, ancestry_map)

    reader = MSPReader(msp_path)
    metadata = reader.read_metadata()
    full = reader.read()

    assert metadata.samples == sample_ids
    assert metadata.ancestry_map is not None

    chunk_chrom = []
    chunk_phys = []
    chunk_lai = []
    for chunk in reader.iter_windows(chunk_size=3):
        chunk_chrom.append(chunk["chromosomes"])
        chunk_phys.append(chunk["physical_pos"])
        chunk_lai.append(chunk["lai"])

    streamed_chrom = np.concatenate(chunk_chrom, axis=0)
    streamed_phys = np.concatenate(chunk_phys, axis=0)
    streamed_lai = np.concatenate(chunk_lai, axis=0)

    np.testing.assert_array_equal(streamed_chrom, full.chromosomes.astype(str))
    np.testing.assert_array_equal(streamed_phys, full.physical_pos.astype(np.int64))
    np.testing.assert_array_equal(streamed_lai, full.lai)


def test_iter_windows_with_sample_subset(tmp_path: Path):
    sample_ids, lai, chromosomes, starts, ends, ancestry_map = make_small_dataset(
        n_samples=8,
        n_windows=9,
        seed=13,
    )
    msp_path = tmp_path / "toy_subset.msp"
    write_msp(msp_path, sample_ids, lai, chromosomes, starts, ends, ancestry_map)

    reader = MSPReader(msp_path)
    full = reader.read()
    sample_indices = np.array([1, 4, 6], dtype=np.int64)

    streamed_lai = []
    for chunk in reader.iter_windows(chunk_size=2, sample_indices=sample_indices):
        streamed_lai.append(chunk["lai"])
    streamed_lai = np.concatenate(streamed_lai, axis=0)

    hap_indices = np.empty(sample_indices.size * 2, dtype=np.int64)
    hap_indices[0::2] = 2 * sample_indices
    hap_indices[1::2] = 2 * sample_indices + 1
    expected = full.lai[:, hap_indices]
    np.testing.assert_array_equal(streamed_lai, expected)


def test_msp_zst_and_gz(tmp_path: Path):
    import gzip
    import zstandard as zstd
    from snputils.ancestry.io.local.read import MSPReader, read_lai
    from snputils.ancestry.genobj.local import LocalAncestryObject

    sample_ids, lai, chromosomes, starts, ends, ancestry_map = make_small_dataset(
        n_samples=5,
        n_windows=10,
        seed=42,
    )

    # 1. Test Writing & Reading uncompressed
    msp_path = tmp_path / "toy.msp"
    write_msp(msp_path, sample_ids, lai, chromosomes, starts, ends, ancestry_map)
    obj_uncompressed = MSPReader(msp_path).read()

    # 2. Test writing & reading .zst
    zst_path = tmp_path / "toy.msp.zst"
    obj_uncompressed.save(zst_path)
    # Check that file exists and has .zst extension
    assert zst_path.exists()
    # Read it back and assert equality
    obj_zst = MSPReader(zst_path).read()
    obj_zst_auto = read_lai(zst_path)
    np.testing.assert_array_equal(obj_uncompressed.lai, obj_zst.lai)
    np.testing.assert_array_equal(obj_uncompressed.lai, obj_zst_auto.lai)
    np.testing.assert_array_equal(obj_uncompressed.samples, obj_zst.samples)
    assert obj_uncompressed.ancestry_map == obj_zst.ancestry_map

    # 3. Test writing & reading .gz
    gz_path = tmp_path / "toy.msp.gz"
    obj_uncompressed.save(gz_path)
    # Check that file exists and has .gz extension
    assert gz_path.exists()
    # Read it back and assert equality
    obj_gz = MSPReader(gz_path).read()
    obj_gz_auto = read_lai(gz_path)
    np.testing.assert_array_equal(obj_uncompressed.lai, obj_gz.lai)
    np.testing.assert_array_equal(obj_uncompressed.lai, obj_gz_auto.lai)
    np.testing.assert_array_equal(obj_uncompressed.samples, obj_gz.samples)
    assert obj_uncompressed.ancestry_map == obj_gz.ancestry_map
