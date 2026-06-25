# Command-Line Interface

Installing **snputils** registers a `snputils` executable for file-backed workflows. The Python API remains the full surface for object manipulation, custom pipelines, and notebook use; use the CLI when inputs and outputs are primarily paths on disk.

```bash
snputils --help
snputils --version
```

## Subcommands

| Command | Purpose |
|---------|---------|
| `pca` | Standard PCA with optional coordinate/component export and scatter plot |
| `mdpca` | Missing-data PCA with ancestry masking |
| `maasmds` | Multi-array ancestry-specific MDS |
| `gwas` | Variant-level association testing |
| `admixture-map` | Admixture mapping from local ancestry |
| `simulate` | Simulate admixed haplotype batches (requires `[torch]`) |
| `plot-manhattan` | Manhattan plot from association results |
| `plot-qq` | Q–Q plot from association results |

Run `snputils <command> --help` for the full argument list of any subcommand.

## Examples

**PCA** — compute two components, save coordinates and a scatter plot:

```bash
snputils pca \
    --snp-path cohort.pgen \
    --n-components 2 \
    --coords pca_coords.tsv \
    --plot pca.pdf
```

**mdPCA** — ancestry-specific embedding with optional plot:

```bash
snputils mdpca \
    --snp-path cohort.pgen \
    --lai-path local_ancestry.msp \
    --labels-file labels.tsv \
    --ancestry AFR \
    --coords mdpca_coords.tsv \
    --plot mdpca.pdf
```

**GWAS** — association scan from phenotype and genotype files:

```bash
snputils gwas \
    --phe-id trait \
    --phe-path phenotypes.tsv \
    --snp-path cohort.pgen \
    --covar-path covariates.txt \
    --results-path gwas.tsv.gz
```

**Admixture mapping** — window-level association from local ancestry:

```bash
snputils admixture-map \
    --phe-id trait \
    --phe-path phenotypes.tsv \
    --lai-path local_ancestry.msp \
    --covar-path covariates.txt \
    --results-path admixmap.tsv.gz
```

**Simulation** — generate admixed haplotype batches (PyTorch required):

Use a phased VCF, PGEN, or BGEN input. PLINK1 BED/BIM/FAM is not accepted for simulation because it does not preserve phase.

```bash
snputils simulate \
    --snp cohort.pgen \
    --metadata metadata.tsv \
    --genetic-map gmap.tsv \
    --output-dir sim_batches/ \
    --batch-size 256 \
    --n-batches 10 \
    --num-generations 10 \
    --device cuda:0
```

**Association plots**:

```bash
snputils plot-manhattan --results-path gwas.tsv.gz --output-path manhattan.png
snputils plot-qq        --results-path gwas.tsv.gz --output-path qq.png
```

## CLI entry point

```{eval-rst}
.. autofunction:: snputils.tools.cli.main
```
