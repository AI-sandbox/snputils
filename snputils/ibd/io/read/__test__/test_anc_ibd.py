import gzip
import tempfile
from io import StringIO
from pathlib import Path

import numpy as np

from snputils.ibd.io.read.anc_ibd import AncIBDReader


HEADER = "\t".join([
    "iid1", "iid2", "ch", "Start", "End", "length", "StartM", "EndM", "lengthM", "StartBP", "EndBP", "segment_type"
])

ROWS = [
    ["A", "B", "1", 10, 20, 11, 0.12, 0.22, 0.10, 1000, 2000, "IBD1"],
    ["C", "D", "2", 30, 40, 11, 0.30, 0.45, 0.15, 3000, 4000, "IBD2"],
]


def _tsv_lines(rows):
    lines = [HEADER]
    for r in rows:
        lines.append("\t".join(map(str, r)))
    return lines


def _write_text(path: Path, lines):
    path.write_text("\n".join(lines) + "\n")


def _write_gz(path: Path, lines):
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")


def test_ancibd_read_file_plain():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        file = tmp_path / "ch_all.tsv"
        _write_text(file, _tsv_lines(ROWS))

        reader = AncIBDReader(file)
        ibd = reader.read()

        assert ibd.n_segments == 2
        assert ibd.sample_id_1.tolist() == ["A", "C"]
        assert ibd.sample_id_2.tolist() == ["B", "D"]
        assert ibd.chrom.tolist() == ["1", "2"]
        assert ibd.start.tolist() == [1000, 3000]
        assert ibd.end.tolist() == [2000, 4000]
        # length_cm is lengthM * 100
        np.testing.assert_allclose(ibd.length_cm.tolist(), [10.0, 15.0])
        # ancIBD does not provide haplotype IDs
        assert set(ibd.haplotype_id_1.tolist()) == {-1}
        assert set(ibd.haplotype_id_2.tolist()) == {-1}


def test_ancibd_read_file_gz():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        file = tmp_path / "ch_all.tsv.gz"
        _write_gz(file, _tsv_lines(ROWS))

        reader = AncIBDReader(file)
        ibd = reader.read()

        assert ibd.n_segments == 2
        assert ibd.start.tolist() == [1000, 3000]
        assert ibd.end.tolist() == [2000, 4000]


def test_ancibd_read_directory_with_ch_files():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        ch1 = tmp_path / "ch1.tsv"
        ch2 = tmp_path / "ch2.tsv.gz"
        _write_text(ch1, _tsv_lines([ROWS[0]]))
        _write_gz(ch2, _tsv_lines([ROWS[1]]))

        reader = AncIBDReader(tmp_path)
        ibd = reader.read()

        assert ibd.n_segments == 2
        assert ibd.chrom.tolist() == ["1", "2"]


def test_ancibd_filter_segment_type():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        file = tmp_path / "ch_all.tsv"
        _write_text(file, _tsv_lines(ROWS))

        reader = AncIBDReader(file)
        ibd = reader.read(include_segment_types=["IBD1"])  # filter to IBD1 only

        assert ibd.n_segments == 1
        assert ibd.sample_id_1.tolist() == ["A"]
        assert ibd.sample_id_2.tolist() == ["B"]
        assert ibd.chrom.tolist() == ["1"]


