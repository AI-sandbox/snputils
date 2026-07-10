import numpy as np
from snputils import PGENReader


# VCF - BED

def test_vcf_bed_samples(snpobj_vcf, snpobj_bed):
    assert snpobj_vcf.samples is not None
    assert snpobj_bed.samples is not None
    assert np.array_equal(snpobj_vcf.samples, snpobj_bed.samples)


def test_vcf_bed_variants_ref(snpobj_vcf, snpobj_bed):
    assert snpobj_vcf.variants_ref is not None
    assert snpobj_bed.variants_ref is not None
    assert np.array_equal(snpobj_vcf.variants_ref, snpobj_bed.variants_ref)


def test_vcf_bed_variants_alt(snpobj_vcf, snpobj_bed):
    assert snpobj_vcf.variants_alt is not None
    assert snpobj_bed.variants_alt is not None
    assert np.array_equal(snpobj_vcf.variants_alt, snpobj_bed.variants_alt)


def test_vcf_bed_variants_chrom(snpobj_vcf, snpobj_bed):
    assert snpobj_vcf.variants_chrom is not None
    assert snpobj_bed.variants_chrom is not None
    assert np.array_equal(snpobj_vcf.variants_chrom, snpobj_bed.variants_chrom)


def test_vcf_bed_variants_id(snpobj_vcf, snpobj_bed):
    assert snpobj_vcf.variants_id is not None
    assert snpobj_bed.variants_id is not None
    assert np.array_equal(snpobj_vcf.variants_id, snpobj_bed.variants_id)


def test_vcf_bed_variants_pos(snpobj_vcf, snpobj_bed):
    assert snpobj_vcf.variants_pos is not None
    assert snpobj_bed.variants_pos is not None
    assert np.array_equal(snpobj_vcf.variants_pos, snpobj_bed.variants_pos)


# BED - PGEN


def test_bed_pgen_samples(snpobj_bed, snpobj_pgen):
    assert snpobj_bed.samples is not None
    assert snpobj_pgen.samples is not None
    assert np.array_equal(snpobj_bed.samples, snpobj_pgen.samples)


def test_bed_pgen_variants_ref(snpobj_bed, snpobj_pgen):
    assert snpobj_bed.variants_ref is not None
    assert snpobj_pgen.variants_ref is not None
    assert np.array_equal(snpobj_bed.variants_ref, snpobj_pgen.variants_ref)


def test_bed_pgen_variants_alt(snpobj_bed, snpobj_pgen):
    assert snpobj_bed.variants_alt is not None
    assert snpobj_pgen.variants_alt is not None
    assert np.array_equal(snpobj_bed.variants_alt, snpobj_pgen.variants_alt)


def test_bed_pgen_variants_chrom(snpobj_bed, snpobj_pgen):
    assert snpobj_bed.variants_chrom is not None
    assert snpobj_pgen.variants_chrom is not None
    assert np.array_equal(snpobj_bed.variants_chrom, snpobj_pgen.variants_chrom)


def test_bed_pgen_variants_id(snpobj_bed, snpobj_pgen):
    assert snpobj_bed.variants_id is not None
    assert snpobj_pgen.variants_id is not None
    assert np.array_equal(snpobj_bed.variants_id, snpobj_pgen.variants_id)


def test_bed_pgen_variants_pos(snpobj_bed, snpobj_pgen):
    assert snpobj_bed.variants_pos is not None
    assert snpobj_pgen.variants_pos is not None
    assert np.array_equal(snpobj_bed.variants_pos, snpobj_pgen.variants_pos)


# VCF - PGEN


def test_vcf_pgen_genotypes(snpobj_vcf, snpobj_pgen):
    assert np.array_equal(snpobj_vcf.genotypes, snpobj_pgen.genotypes)


def test_vcf_pgen_samples(snpobj_vcf, snpobj_pgen):
    assert snpobj_vcf.samples is not None
    assert snpobj_pgen.samples is not None
    assert np.array_equal(snpobj_vcf.samples, snpobj_pgen.samples)


def test_vcf_pgen_variants_ref(snpobj_vcf, snpobj_pgen):
    assert snpobj_vcf.variants_ref is not None
    assert snpobj_pgen.variants_ref is not None
    assert np.array_equal(snpobj_vcf.variants_ref, snpobj_pgen.variants_ref)


def test_vcf_pgen_variants_alt(snpobj_vcf, snpobj_pgen):
    assert snpobj_vcf.variants_alt is not None
    assert snpobj_pgen.variants_alt is not None
    assert np.array_equal(snpobj_vcf.variants_alt, snpobj_pgen.variants_alt)


def test_vcf_pgen_variants_chrom(snpobj_vcf, snpobj_pgen):
    assert snpobj_vcf.variants_chrom is not None
    assert snpobj_pgen.variants_chrom is not None
    assert np.array_equal(snpobj_vcf.variants_chrom, snpobj_pgen.variants_chrom)


def test_vcf_pgen_variants_id(snpobj_vcf, snpobj_pgen):
    assert snpobj_vcf.variants_id is not None
    assert snpobj_pgen.variants_id is not None
    assert np.array_equal(snpobj_vcf.variants_id, snpobj_pgen.variants_id)


def test_vcf_pgen_variants_pos(snpobj_vcf, snpobj_pgen):
    assert snpobj_vcf.variants_pos is not None
    assert snpobj_pgen.variants_pos is not None
    assert np.array_equal(snpobj_vcf.variants_pos, snpobj_pgen.variants_pos)


def test_variants_alt_shape(snpobj_vcf):
    assert snpobj_vcf.variants_alt.shape == (snpobj_vcf.n_snps,)


# Compressed VCF
def test_vcf_gz_and_zst(data_path, snpobj_vcf):
    import gzip
    import zstandard as zstd
    from snputils.snp.io.read.vcf import VCFReader, VCFReaderPolars
    import pathlib

    vcf_path = pathlib.Path(data_path) / "vcf" / "subset.vcf"
    gz_path = pathlib.Path(data_path) / "vcf" / "subset.vcf.gz"
    zst_path = pathlib.Path(data_path) / "vcf" / "subset.vcf.zst"

    # Create compressed files if they don't exist
    if not gz_path.exists():
        with open(vcf_path, "rb") as f_in:
            with gzip.open(gz_path, "wb") as f_out:
                f_out.writelines(f_in)

    if not zst_path.exists():
        with open(vcf_path, "rb") as f_in:
            cctx = zstd.ZstdCompressor()
            with open(zst_path, "wb") as f_out:
                cctx.copy_stream(f_in, f_out)

    # Test default VCFReader on .gz
    snpobj_gz = VCFReader(gz_path).read(sum_strands=False)
    assert np.array_equal(snpobj_vcf.genotypes, snpobj_gz.genotypes)
    assert np.array_equal(snpobj_vcf.variants_pos, snpobj_gz.variants_pos)

    # Test default VCFReader on .zst
    snpobj_zst = VCFReader(zst_path).read(sum_strands=False)
    assert np.array_equal(snpobj_vcf.genotypes, snpobj_zst.genotypes)
    assert np.array_equal(snpobj_vcf.variants_pos, snpobj_zst.variants_pos)

    # Test VCFReaderPolars on .gz
    snpobj_polars_gz = VCFReaderPolars(gz_path).read(sum_strands=False)
    assert np.array_equal(snpobj_vcf.genotypes, snpobj_polars_gz.genotypes)
    assert np.array_equal(snpobj_vcf.variants_pos, snpobj_polars_gz.variants_pos)

    # Test VCFReaderPolars on .zst
    snpobj_polars_zst = VCFReaderPolars(zst_path).read(sum_strands=False)
    assert np.array_equal(snpobj_vcf.genotypes, snpobj_polars_zst.genotypes)
    assert np.array_equal(snpobj_vcf.variants_pos, snpobj_polars_zst.variants_pos)


# PGEN with compressed pvar
def test_pgen_pvar_zst(data_path, snpobj_pgen):
    snpobj = PGENReader(data_path + "/pgen_zst/subset").read(sum_strands=False)
    assert np.array_equal(snpobj_pgen.genotypes, snpobj.genotypes)
    assert np.array_equal(snpobj_pgen.variants_ref, snpobj.variants_ref)
    assert np.array_equal(snpobj_pgen.variants_alt, snpobj.variants_alt)
    assert np.array_equal(snpobj_pgen.variants_chrom, snpobj.variants_chrom)
    assert np.array_equal(snpobj_pgen.variants_id, snpobj.variants_id)
    assert np.array_equal(snpobj_pgen.variants_pos, snpobj.variants_pos)
    assert np.array_equal(snpobj_pgen.variants_filter_pass, snpobj.variants_filter_pass)
    assert np.array_equal(snpobj_pgen.variants_qual, snpobj.variants_qual)


# BED with compressed bim and fam
def test_bed_bim_fam_zst(data_path, snpobj_bed, tmp_path):
    import gzip
    import zstandard as zstd
    from snputils import BEDReader
    import shutil
    import pathlib

    bed_src = pathlib.Path(data_path) / "bed" / "subset.bed"
    bim_src = pathlib.Path(data_path) / "bed" / "subset.bim"
    fam_src = pathlib.Path(data_path) / "bed" / "subset.fam"

    # Copy bed file to tmp_path
    shutil.copy(bed_src, tmp_path / "subset.bed")

    # Compress bim and fam to .zst in tmp_path
    cctx = zstd.ZstdCompressor()
    with open(bim_src, "rb") as f_in, open(tmp_path / "subset.bim.zst", "wb") as f_out:
        cctx.copy_stream(f_in, f_out)
    with open(fam_src, "rb") as f_in, open(tmp_path / "subset.fam.zst", "wb") as f_out:
        cctx.copy_stream(f_in, f_out)

    # Read from zst compressed bim/fam
    snpobj_zst = BEDReader(tmp_path / "subset").read(sum_strands=True)
    assert np.array_equal(snpobj_bed.genotypes, snpobj_zst.genotypes)
    assert np.array_equal(snpobj_bed.variants_pos, snpobj_zst.variants_pos)
    assert np.array_equal(snpobj_bed.samples, snpobj_zst.samples)

    # Test .gz as well
    # Compress bim and fam to .gz in tmp_path
    with open(bim_src, "rb") as f_in, gzip.open(tmp_path / "subset_gz.bim.gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    with open(fam_src, "rb") as f_in, gzip.open(tmp_path / "subset_gz.fam.gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    shutil.copy(bed_src, tmp_path / "subset_gz.bed")

    snpobj_gz = BEDReader(tmp_path / "subset_gz").read(sum_strands=True)
    assert np.array_equal(snpobj_bed.genotypes, snpobj_gz.genotypes)
    assert np.array_equal(snpobj_bed.variants_pos, snpobj_gz.variants_pos)
    assert np.array_equal(snpobj_bed.samples, snpobj_gz.samples)


# PGEN with compressed pvar and psam
def test_pgen_pvar_psam_zst_and_gz(data_path, snpobj_pgen, tmp_path):
    import gzip
    import zstandard as zstd
    from snputils import PGENReader
    import shutil
    import pathlib

    pgen_src = pathlib.Path(data_path) / "pgen" / "subset.pgen"
    pvar_src = pathlib.Path(data_path) / "pgen" / "subset.pvar"
    psam_src = pathlib.Path(data_path) / "pgen" / "subset.psam"

    # Test .zst fileset
    # Copy pgen file
    shutil.copy(pgen_src, tmp_path / "subset.pgen")
    
    # Compress pvar and psam to .zst in tmp_path
    cctx = zstd.ZstdCompressor()
    with open(pvar_src, "rb") as f_in, open(tmp_path / "subset.pvar.zst", "wb") as f_out:
        cctx.copy_stream(f_in, f_out)
    with open(psam_src, "rb") as f_in, open(tmp_path / "subset.psam.zst", "wb") as f_out:
        cctx.copy_stream(f_in, f_out)

    snpobj_zst = PGENReader(tmp_path / "subset").read(sum_strands=False)
    assert np.array_equal(snpobj_pgen.genotypes, snpobj_zst.genotypes)
    assert np.array_equal(snpobj_pgen.variants_pos, snpobj_zst.variants_pos)
    assert np.array_equal(snpobj_pgen.samples, snpobj_zst.samples)

    # Test .gz fileset
    # Copy pgen file
    shutil.copy(pgen_src, tmp_path / "subset_gz.pgen")
    
    # Compress pvar and psam to .gz in tmp_path
    with open(pvar_src, "rb") as f_in, gzip.open(tmp_path / "subset_gz.pvar.gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    with open(psam_src, "rb") as f_in, gzip.open(tmp_path / "subset_gz.psam.gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    snpobj_gz = PGENReader(tmp_path / "subset_gz").read(sum_strands=False)
    assert np.array_equal(snpobj_pgen.genotypes, snpobj_gz.genotypes)
    assert np.array_equal(snpobj_pgen.variants_pos, snpobj_gz.variants_pos)
    assert np.array_equal(snpobj_pgen.samples, snpobj_gz.samples)
