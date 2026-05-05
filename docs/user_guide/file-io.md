# File I/O

High-level readers dispatch to the right implementation for common genomic formats.

```python
import snputils as su

snpobj = su.read_snp("cohort.vcf.gz")
```

## SNP Formats

```python
from snputils import read_bed, read_pgen, read_vcf

bed = read_bed("cohort.bed")
pgen = read_pgen("cohort.pgen")
vcf = read_vcf("cohort.vcf.gz")
```

Use the explicit reader classes when you need constructor options, streaming, or staged reads:

```python
from snputils import PGENReader

reader = PGENReader("cohort.pgen")
snpobj = reader.read()
```

## Ancestry and IBD Formats

```python
lai = su.read_msp("local_ancestry.msp")
adm = su.read_admixture("admixture_prefix")
ibd = su.read_ibd("segments.hapibd")
```

Writers are available for SNP files, local ancestry files, global ancestry files, and admixture-mapping VCF output.
