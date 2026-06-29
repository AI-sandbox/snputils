# File I/O

High-level readers dispatch to the right implementation based on file extension.

```python
import snputils as su

snpobj = su.read_snp("cohort.vcf.gz")   # auto-detects format
```

## SNP Formats

| Format | Read | Write |
|--------|------|-------|
| PLINK BED (`.bed/.bim/.fam`) | `read_bed` / `BEDReader` | `BEDWriter` |
| PLINK2 PGEN (`.pgen/.psam/.pvar`) | `read_pgen` / `PGENReader` | `PGENWriter` |
| VCF / VCF.gz | `read_vcf` / `VCFReader` | `VCFWriter` |
| BCF (`.bcf`) | `read_bcf` / `BCFReader` | `BCFWriter` |
| BGEN | `read_bgen` / `BGENReader` | `BGENWriter` |
| GRG | `read_grg` / `GRGReader` | `GRGWriter` |

```python
from snputils import read_bcf, read_bed, read_bgen, read_pgen, read_vcf

bcf  = read_bcf("cohort.bcf")
bed  = read_bed("cohort.bed")
bgen = read_bgen("cohort.bgen")
pgen = read_pgen("cohort.pgen")
vcf  = read_vcf("cohort.vcf.gz")
```

**Reader options:**

```python
from snputils import BCFReader, PGENReader, BGENReader, VCFReader

# PGEN: read only specific samples or chromosomes
pgen = PGENReader("cohort.pgen").read(
    samples=["sample1", "sample2"],
    variants_chrom=["1", "2"],
    sum_strands=False,
)

# VCF: read only phased GT, specific region
vcf = VCFReader("cohort.vcf.gz").read(
    region="chr1:1000000-2000000",
    sum_strands=False,
)

# BCF: read only specific samples and variants
bcf = BCFReader("cohort.bcf").read(
    sample_ids=["sample1", "sample2"],
    variant_ids=["chr1:1000000", "rs123"],
)
```

**BCF notes:** BCF reads use a native snputils parser over BGZF-compressed BCF2.2 records. Genotypes are stored on `SNPObject.genotypes` just like VCF input. ``region=...`` works without an index by scanning and filtering matching records.

**BGEN notes:** Genotype probabilities are stored on `SNPObject.calldata_gp` with shape `(n_snps, n_samples, n_probs)`. Hard calls are not inferred during reads. Mixed-width BGEN records are padded with `NaN` columns.

**Writer options:**

```python
from snputils import BEDWriter, PGENWriter, VCFWriter, BCFWriter, BGENWriter

BEDWriter(snpobj, "out.bed").write()
PGENWriter(snpobj, "out.pgen").write(vzs=True)              # compressed .pvar.zst
VCFWriter(snpobj, "out.vcf", phased=True).write(
    chrom_partition=True                                     # one file per chromosome
)
BCFWriter(snpobj, "out.bcf", phased=True).write(
    chrom_partition=True                                     # one file per chromosome
)
BGENWriter(snpobj, "out.bgen").write(
    compression="zstd", layout=2, bit_depth=16, phased=True
)
```

## Local Ancestry Formats

| Format | Read | Write |
|--------|------|-------|
| MSP (`.msp`, `.msp.tsv`) | `read_msp` / `MSPReader` | `MSPWriter` |
| FLARE (`.anc.vcf[.gz]`) | `read_flare` / `FLAREReader` | `FLAREWriter` |
| admix-kit LANC (`.lanc`) | `read_lanc` / `LANCReader` | `LANCWriter` |

```python
msp   = su.read_lai("local_ancestry.msp")
flare = su.read_lai("flare.out.anc.vcf.gz")
lanc  = su.read_lai("cohort.lanc")

su.MSPWriter(laiobj, "out.msp").write()
su.LANCWriter(laiobj, "out.lanc").write()

# FLARE VCF output requires matching SNP data (GT + variant metadata)
su.FLAREWriter(laiobj, "out.anc.vcf", snpobj=snpobj).write()
```

`.lanc` stores only the SNP-level diploid ancestry matrix in run-length encoded form. To keep snputils round-trips self-contained, `LANCWriter(...).write()` and `LocalAncestryObject.save_lanc(...)` write sibling `.pvar` and `.psam` sidecars by default when the necessary metadata is present on the object.

```python
su.LANCWriter(laiobj, "out.lanc").write()           # also writes out.pvar + out.psam
laiobj.save_lanc("out.lanc")                        # same default behavior

# sparse stream only
su.LANCWriter(laiobj, "out.lanc", write_sidecars=False).write()
```

On reads, `read_lanc(...)` and `read_lai(...)` look for sibling `.pvar`/`.pvar.zst` and `.psam` files with the same prefix as the `.lanc` file and use them to reconstruct SNP coordinates and sample IDs.

```python
lanc = su.read_lanc("cohort.lanc")  # expects cohort.pvar[.zst] and cohort.psam

# or point at sidecars elsewhere
lanc = su.read_lanc(
    "cohort.lanc",
    pvar_file="metadata/chr1.pvar",
    psam_file="metadata/cohort.psam",
)
```

If the sidecars are missing, snputils warns and falls back to loading the LAI calls alone: it generates sample and haplotype IDs (`sample_0`, `sample_0.0`, ...), sets `window_sizes` to one SNP per row, and leaves unavailable coordinate metadata unset.

## Global Ancestry / ADMIXTURE

```python
admobj = su.read_admixture("run_prefix")   # reads run_prefix.Q and run_prefix.P
su.AdmixtureWriter(admobj, "out_prefix").write()
```

## Admixture Mapping Output

```python
# Write a VCF for downstream admixture-mapping tools
su.AdmixtureMappingVCFWriter(laiobj, snpobj, "admix_map.vcf").write()
```

## IBD Formats

```python
hapibd = su.read_ibd("segments.hapibd")     # hap-IBD
ancibd = su.AncIBDReader("ch_all.tsv").read(
    include_segment_types=["IBD1", "IBD2"]  # filter by segment type
)
```

## Phenotype Formats

```python
phen  = su.read_pheno("pheno.tsv")
mphen = su.MultiPhenReader("multi.tsv").read()
```

## Covariate Formats

Covariate files feed {func}`~snputils.run_gwas` and {func}`~snputils.run_admixture_mapping` through the `covar` argument. Use whitespace-separated text with `IID` in the header and numeric columns after it. An optional `#FID` column before `IID` is allowed.

```text
#FID IID age sex
sample1 sample1 58 1
sample2 sample2 49 2
```

In Python you can pass the same data as a {class}`~snputils.CovariateObject` instead of a path.

## Datasets

Built-in datasets and synthetic data builders for quick experimentation:

```python
su.available_datasets_list()                    # list bundled datasets
ds = su.load_dataset("1kgp")                    # returns a SNPObject

snpobj   = su.build_synthetic_snp_dataset()
laiobj   = su.build_synthetic_chromosome_painting_dataset()
admobj   = su.build_synthetic_admixture_dataset()
phen     = su.build_synthetic_phenotype_dataset()
grg      = su.build_synthetic_grg()
mdpca_ds = su.build_synthetic_mdpca_dataset()
maas_ds  = su.build_synthetic_maasmds_dataset()
```
