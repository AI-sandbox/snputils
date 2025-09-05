import gzip
import tempfile
from pathlib import Path

import numpy as np

from snputils.ibd.io.read.hap_ibd import HapIBDReader


def _write_text(path: Path, lines):
    path.write_text("\n".join(lines) + "\n")


def _write_gz(path: Path, lines):
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")


def _build_lines(tokens, joiner: str) -> list:
    return [joiner.join(map(str, row)) for row in tokens]


TOKENS = [
    ["SUBJ-A", 1, "SUBJ-B", 2, "chr10", 46535262, 48183075, 2.320],
    ["SUBJ-C", 2, "SUBJ-D", 1, "chr2", 100, 200, 1.23],
    ["SUBJ-E", 1, "SUBJ-E", 2, "chrX", 123, 456, 0.01],
]


def test_hap_ibd_whitespace_default_separator_plain():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Mix spaces and tabs, with variable spacing
        whitespace_lines = [
            "SUBJ-A    1    SUBJ-B\t2   chr10   46535262   48183075   2.320",
            "SUBJ-C  2  SUBJ-D  1  chr2   100  200  1.23",
            "SUBJ-E\t1   SUBJ-E  2  chrX   123   456  0.01",
        ]
        file = tmp_path / "example.ibd"
        _write_text(file, whitespace_lines)

        reader = HapIBDReader(file)
        ibd = reader.read()

        assert ibd.n_segments == 3
        assert ibd.sample_id_1.tolist() == [r[0] for r in TOKENS]
        assert ibd.haplotype_id_1.tolist() == [r[1] for r in TOKENS]
        assert ibd.sample_id_2.tolist() == [r[2] for r in TOKENS]
        assert ibd.haplotype_id_2.tolist() == [r[3] for r in TOKENS]
        assert ibd.chrom.tolist() == [r[4] for r in TOKENS]
        assert ibd.start.tolist() == [r[5] for r in TOKENS]
        assert ibd.end.tolist() == [r[6] for r in TOKENS]
        np.testing.assert_allclose(ibd.length_cm.tolist(), [r[7] for r in TOKENS], rtol=0, atol=0)


def test_hap_ibd_tab_explicit_separator_plain():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tab_lines = _build_lines(TOKENS, "\t")
        file = tmp_path / "example_tabs.ibd"
        _write_text(file, tab_lines)

        reader = HapIBDReader(file)
        ibd = reader.read(separator="\t")

        assert ibd.n_segments == 3
        assert ibd.sample_id_1.tolist() == [r[0] for r in TOKENS]
        assert ibd.haplotype_id_1.tolist() == [r[1] for r in TOKENS]
        assert ibd.sample_id_2.tolist() == [r[2] for r in TOKENS]
        assert ibd.haplotype_id_2.tolist() == [r[3] for r in TOKENS]
        assert ibd.chrom.tolist() == [r[4] for r in TOKENS]
        assert ibd.start.tolist() == [r[5] for r in TOKENS]
        assert ibd.end.tolist() == [r[6] for r in TOKENS]
        np.testing.assert_allclose(ibd.length_cm.tolist(), [r[7] for r in TOKENS], rtol=0, atol=0)


def test_hap_ibd_whitespace_default_separator_gz():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        whitespace_lines = [
            "SUBJ-A    1    SUBJ-B\t2   chr10   46535262   48183075   2.320",
            "SUBJ-C  2  SUBJ-D  1  chr2   100  200  1.23",
            "SUBJ-E\t1   SUBJ-E  2  chrX   123   456  0.01",
        ]
        file = tmp_path / "example.ibd.gz"
        _write_gz(file, whitespace_lines)

        reader = HapIBDReader(file)
        ibd = reader.read()

        assert ibd.n_segments == 3
        assert ibd.sample_id_1.tolist() == [r[0] for r in TOKENS]
        assert ibd.haplotype_id_1.tolist() == [r[1] for r in TOKENS]
        assert ibd.sample_id_2.tolist() == [r[2] for r in TOKENS]
        assert ibd.haplotype_id_2.tolist() == [r[3] for r in TOKENS]
        assert ibd.chrom.tolist() == [r[4] for r in TOKENS]
        assert ibd.start.tolist() == [r[5] for r in TOKENS]
        assert ibd.end.tolist() == [r[6] for r in TOKENS]
        np.testing.assert_allclose(ibd.length_cm.tolist(), [r[7] for r in TOKENS], rtol=0, atol=0)


def test_hap_ibd_tab_explicit_separator_gz():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tab_lines = _build_lines(TOKENS, "\t")
        file = tmp_path / "example_tabs.ibd.gz"
        _write_gz(file, tab_lines)

        reader = HapIBDReader(file)
        ibd = reader.read(separator="\t")

        assert ibd.n_segments == 3
        assert ibd.sample_id_1.tolist() == [r[0] for r in TOKENS]
        assert ibd.haplotype_id_1.tolist() == [r[1] for r in TOKENS]
        assert ibd.sample_id_2.tolist() == [r[2] for r in TOKENS]
        assert ibd.haplotype_id_2.tolist() == [r[3] for r in TOKENS]
        assert ibd.chrom.tolist() == [r[4] for r in TOKENS]
        assert ibd.start.tolist() == [r[5] for r in TOKENS]
        assert ibd.end.tolist() == [r[6] for r in TOKENS]
        np.testing.assert_allclose(ibd.length_cm.tolist(), [r[7] for r in TOKENS], rtol=0, atol=0)


