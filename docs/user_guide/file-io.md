# File I/O

High-level readers dispatch to the right implementation for common genomic formats.

```python
import snputils as su

snpobj = su.read_snp("cohort.vcf.gz")
```

## SNP Formats

```python
from snputils import read_bed, read_bgen, read_pgen, read_vcf

bed = read_bed("cohort.bed")
bgen = read_bgen("cohort.bgen")
pgen = read_pgen("cohort.pgen")
vcf = read_vcf("cohort.vcf.gz")
```

BGEN genotype probabilities are stored on `SNPObject.calldata_gp`; they are not converted to hard-call genotypes during reads.
When a BGEN file mixes probability widths, shorter records are padded with `NaN` columns so the object remains a single array.

Use the explicit reader classes when you need constructor options, streaming, or staged reads:

```python
from snputils import BGENReader, PGENReader

reader = PGENReader("cohort.pgen")
snpobj = reader.read()

bgen_reader = BGENReader("cohort.bgen")
bgen_snpobj = bgen_reader.read()
```

## Ancestry and IBD Formats

```python
msp_lai = su.read_lai("local_ancestry.msp")          # MSP
flare_lai = su.read_lai("flare.out.anc.vcf.gz")  # FLARE
adm = su.read_admixture("admixture_prefix")
ibd = su.read_ibd("segments.hapibd")
```

Local ancestry readers support MSP (`.msp`, `.msp.tsv`) and FLARE (`.anc.vcf`, `.anc.vcf.gz`) outputs. FLARE writing requires genotype data from a `SNPObject` or genotype file because FLARE VCF output includes the phased `GT` field and variant metadata alongside `AN1`/`AN2`. Writers are also available for SNP files, global ancestry files, and admixture-mapping VCF output.
