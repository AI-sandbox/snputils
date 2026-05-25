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
| BGEN | `read_bgen` / `BGENReader` | `BGENWriter` |
| GRG | `read_grg` / `GRGReader` | `GRGWriter` |

```python
from snputils import read_bed, read_bgen, read_pgen, read_vcf

bed  = read_bed("cohort.bed")
bgen = read_bgen("cohort.bgen")
pgen = read_pgen("cohort.pgen")
vcf  = read_vcf("cohort.vcf.gz")
```

**Reader options:**

```python
from snputils import PGENReader, BGENReader, VCFReader

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
```

**BGEN notes:** Genotype probabilities are stored on `SNPObject.calldata_gp` with shape `(n_snps, n_samples, n_probs)`. Hard calls are not inferred during reads. Mixed-width BGEN records are padded with `NaN` columns.

**Writer options:**

```python
from snputils import BEDWriter, PGENWriter, VCFWriter, BGENWriter

BEDWriter(snpobj, "out.bed").write()
PGENWriter(snpobj, "out.pgen").write(vzs=True)              # compressed .pvar.zst
VCFWriter(snpobj, "out.vcf", phased=True).write(
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

```python
msp   = su.read_lai("local_ancestry.msp")
flare = su.read_lai("flare.out.anc.vcf.gz")

su.MSPWriter(laiobj, "out.msp").write()

# FLARE VCF output requires matching SNP data (GT + variant metadata)
su.FLAREWriter(laiobj, "out.anc.vcf", snpobj=snpobj).write()
```

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
