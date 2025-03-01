import tempfile
import pytest
import numpy as np
from pathlib import Path
import os

from snputils.snp.io.read import BEDReader, PGENReader


@pytest.fixture
def test_bed_files():
    """Create test BED files with non-numeric phenotypes including text values."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create a test FAM file with various non-numeric phenotype values
        fam_lines = [
            "FAM1 IND1 0 0 1 -9",     # Missing (PLINK format)
            "FAM2 IND2 0 0 2 NA",     # Missing (common alternative)
            "FAM3 IND3 0 0 0 0",      # Missing (alternative format)
            "FAM4 IND4 0 0 1 1",      # Control
            "FAM5 IND5 0 0 2 2",      # Case
            "FAM6 IND6 0 0 1 case",   # Text value - categorical
        ]

        # Create minimal BIM file with 2 variants
        bim_lines = [
            "1 SNP1 0 1000 A G",
            "1 SNP2 0 2000 C T",
        ]

        # Set up paths
        base_path = tmp_path / "test"
        bed_base = str(base_path)

        fam_path = base_path.with_suffix(".fam")
        bim_path = base_path.with_suffix(".bim")
        bed_path = base_path.with_suffix(".bed")

        # Write FAM and BIM files
        with open(fam_path, "w") as f:
            f.write("\n".join(fam_lines))

        with open(bim_path, "w") as f:
            f.write("\n".join(bim_lines))

        # Create a minimal BED file (binary format)
        # (actual content doesn't matter since we're only testing phenotype handling)
        with open(bed_path, "wb") as f:
            f.write(b"\x6c\x1b\x01\x00")  # Magic bytes + minimal data

        yield bed_base


@pytest.fixture
def test_pgen_files():
    """Create minimal test PGEN fileset with non-numeric phenotypes."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        pgen_base = str(tmp_path / "test_pgen")

        # 1. Create PSAM file with various phenotype values
        psam_lines = [
            "#FID\tIID\tSEX\tPHENO1",
            "FAM1\tIND1\t1\t-9",      # Missing (PLINK format)
            "FAM2\tIND2\t2\tNA",      # Missing (common alternative)
            "FAM3\tIND3\t0\t0",       # Missing (alternative format)
            "FAM4\tIND4\t1\t1",       # Control
            "FAM5\tIND5\t2\t2",       # Case
            "FAM6\tIND6\t1\tcase",    # Text value - categorical
        ]

        # 2. Create a minimal PVAR file
        pvar_lines = [
            "##fileformat=VCFv4.2",
            "#CHROM\tPOS\tID\tREF\tALT",
            "1\t1000\tSNP1\tG\tA",
            "1\t2000\tSNP2\tT\tC"
        ]

        # 3. Create an empty PGEN file (content doesn't matter since we're not reading genotypes)
        pgen_path = Path(pgen_base + ".pgen")
        pvar_path = Path(pgen_base + ".pvar")
        psam_path = Path(pgen_base + ".psam")

        # Make directory if needed
        os.makedirs(os.path.dirname(pgen_path), exist_ok=True)

        # Write the files
        with open(psam_path, "w") as f:
            f.write("\n".join(psam_lines))

        with open(pvar_path, "w") as f:
            f.write("\n".join(pvar_lines))

        with open(pgen_path, "wb") as f:
            f.write(b"\0")  # Just a placeholder

        yield pgen_base


def test_bed_text_phenotypes(test_bed_files):
    """Test that BEDReader can handle text phenotype values."""
    reader = BEDReader(test_bed_files)

    # Test reading sample IDs
    snpobj = reader.read(fields=["IID"])
    assert snpobj.samples is not None
    assert len(snpobj.samples) == 6
    expected_samples = ["IND1", "IND2", "IND3", "IND4", "IND5", "IND6"]
    assert np.array_equal(snpobj.samples, np.array(expected_samples))


def test_pgen_text_phenotypes(test_pgen_files):
    """Test that PGENReader can handle text phenotype values."""
    reader = PGENReader(test_pgen_files)

    # Test reading sample IDs
    snpobj = reader.read(fields=["IID"])
    assert snpobj.samples is not None
    assert len(snpobj.samples) == 6
    expected_samples = ["IND1", "IND2", "IND3", "IND4", "IND5", "IND6"]
    assert np.array_equal(snpobj.samples, np.array(expected_samples))


def test_bed_filter_text_phenotype_samples(test_bed_files):
    """Test filtering samples with text phenotypes in BED files."""
    reader = BEDReader(test_bed_files)

    # Include the sample with text phenotype ("case")
    sample_subset = ["IND1", "IND5", "IND6"]  # IND6 has text phenotype
    snpobj = reader.read(fields=["IID"], sample_ids=sample_subset)

    # Verify filtered samples
    assert snpobj.samples is not None
    assert len(snpobj.samples) == 3
    assert set(snpobj.samples) == set(sample_subset)


def test_pgen_filter_text_phenotype_samples(test_pgen_files):
    """Test filtering samples with text phenotypes in PGEN files."""
    reader = PGENReader(test_pgen_files)

    # Include the sample with text phenotype ("case")
    sample_subset = ["IND1", "IND5", "IND6"]  # IND6 has text phenotype
    snpobj = reader.read(fields=["IID"], sample_ids=sample_subset)

    # Verify filtered samples
    assert snpobj.samples is not None
    assert len(snpobj.samples) == 3
    assert set(snpobj.samples) == set(sample_subset)
