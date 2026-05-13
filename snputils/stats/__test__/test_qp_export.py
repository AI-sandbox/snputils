import json
import shutil
import struct
import subprocess

import numpy as np
import pytest

from snputils.stats import export_qp, f2


def _toy_data():
    afs = np.array(
        [
            [0.10, 0.20, 0.80],
            [0.20, 0.25, 0.70],
            [0.30, 0.40, 0.60],
            [0.40, 0.50, 0.55],
        ],
        dtype=float,
    )
    counts = np.full_like(afs, 20.0)
    pops = ["A", "B", "C"]
    return afs, counts, pops


def _read_int(data, offset):
    return struct.unpack(">i", data[offset : offset + 4])[0], offset + 4


def _read_rds_int_vector(path):
    data = path.read_bytes()
    assert data[:2] == b"X\n"
    offset = 14
    sexptype, offset = _read_int(data, offset)
    assert sexptype == 13
    length, offset = _read_int(data, offset)
    values = []
    for _ in range(length):
        value, offset = _read_int(data, offset)
        values.append(value)
    return np.asarray(values, dtype=np.int32)


def _read_rds_real_payload(path):
    data = path.read_bytes()
    assert data[:2] == b"X\n"
    offset = 14
    sexptype, offset = _read_int(data, offset)
    assert sexptype == (14 | 512)
    length, offset = _read_int(data, offset)
    values = np.frombuffer(data[offset : offset + 8 * length], dtype=">f8").astype(float)
    return values


def _read_pair_file(path, n_blocks):
    values = _read_rds_real_payload(path)
    return values.reshape((n_blocks, 2), order="F")


def test_export_qp_writes_blocked_pair_layout(tmp_path):
    afs, counts, pops = _toy_data()

    result = export_qp((afs, counts, pops), tmp_path, block_size=2)

    assert result.populations == ("A", "B", "C")
    assert result.statistics == ("f2", "ap")
    assert result.n_blocks == 2
    assert _read_rds_int_vector(tmp_path / "block_lengths_f2.rds").tolist() == [2, 2]
    assert _read_rds_int_vector(tmp_path / "block_lengths_ap.rds").tolist() == [2, 2]
    assert (tmp_path / "A" / "A_f2.rds").exists()
    assert (tmp_path / "A" / "B_f2.rds").exists()
    assert not (tmp_path / "B" / "A_f2.rds").exists()

    ab = _read_pair_file(tmp_path / "A" / "B_f2.rds", result.n_blocks)
    expected = []
    for start in (0, 2):
        stop = start + 2
        pa = afs[start:stop, 0]
        pb = afs[start:stop, 1]
        na = counts[start:stop, 0]
        nb = counts[start:stop, 1]
        expected.append(np.mean((pa - pb) ** 2 - pa * (1.0 - pa) / (na - 1.0) - pb * (1.0 - pb) / (nb - 1.0)))
    assert np.allclose(ab[:, 0], expected)
    assert np.allclose(ab[:, 1], [1.0, 1.0])

    manifest = json.loads((tmp_path / "qp_export.json").read_text())
    assert manifest["tools"] == ["qpadm", "qpgraph", "qpwave"]
    assert manifest["statistics"] == ["f2", "ap"]
    assert manifest["block_lengths"] == [2, 2]


def test_export_qp_f2_weighted_mean_matches_native_f2(tmp_path):
    afs, counts, pops = _toy_data()
    result = export_qp((afs, counts, pops), tmp_path, tools="qpGraph", block_size=2)

    ab = _read_pair_file(tmp_path / "A" / "B_f2.rds", result.n_blocks)
    block_lengths = _read_rds_int_vector(tmp_path / "block_lengths_f2.rds").astype(float)
    exported_est = np.average(ab[:, 0], weights=block_lengths)
    native_est = f2((afs, counts, pops), pop1=["A"], pop2=["B"], block_size=2).est.iloc[0]

    assert result.statistics == ("f2",)
    assert np.isclose(exported_est, native_est, atol=1e-15)
    assert not (tmp_path / "block_lengths_ap.rds").exists()


def test_export_qp_ap_values_for_qpadm_qpwave(tmp_path):
    afs, counts, pops = _toy_data()
    result = export_qp((afs, counts, pops), tmp_path, tools=("qpAdm", "qpWave"), block_size=2)

    ac = _read_pair_file(tmp_path / "A" / "C_ap.rds", result.n_blocks)
    expected = []
    for start in (0, 2):
        stop = start + 2
        pa = afs[start:stop, 0]
        pc = afs[start:stop, 2]
        expected.append(np.mean((pa * pc + (1.0 - pa) * (1.0 - pc)) / 2.0))

    assert result.statistics == ("ap",)
    assert np.allclose(ac[:, 0], expected)
    assert not (tmp_path / "block_lengths_f2.rds").exists()


def test_export_qp_rejects_unsafe_population_names(tmp_path):
    afs, counts, _ = _toy_data()
    with pytest.raises(ValueError, match="path separators"):
        export_qp((afs, counts, ["A", "bad/name", "C"]), tmp_path)


def test_export_qp_overwrite_guard(tmp_path):
    afs, counts, pops = _toy_data()
    export_qp((afs, counts, pops), tmp_path, block_size=2)
    with pytest.raises(FileExistsError):
        export_qp((afs, counts, pops), tmp_path, block_size=2)
    export_qp((afs, counts, pops), tmp_path, block_size=2, overwrite=True)


def test_export_qp_rds_readable_by_base_r_when_available(tmp_path):
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript is not available")

    afs, counts, pops = _toy_data()
    export_qp((afs, counts, pops), tmp_path, block_size=2)

    script = f"""
    bl <- readRDS({str(tmp_path / "block_lengths_f2.rds")!r})
    x <- readRDS({str(tmp_path / "A" / "B_f2.rds")!r})
    stopifnot(identical(as.integer(bl), c(2L, 2L)))
    stopifnot(is.matrix(x))
    stopifnot(identical(colnames(x), c("f2", "counts")))
    stopifnot(nrow(x) == 2L, ncol(x) == 2L)
    """
    subprocess.run([rscript, "-e", script], check=True)
