<p align="center">
  <a href="https://snputils.org">
    <img src="https://raw.githubusercontent.com/AI-sandbox/snputils/refs/heads/main/assets/logo.png" width="300" alt="snputils logo">
  </a>
</p>

# snputils: A Python Library for Processing Genetic Variation and Population Structure

[![License BSD-3](https://img.shields.io/pypi/l/snputils.svg?color=green)](https://github.com/ai-sandbox/snputils/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/snputils.svg?color=green)](https://pypi.org/project/snputils)
[![Python Version](https://img.shields.io/pypi/pyversions/snputils.svg?color=green)](https://python.org)
[![Test, Docs & Publish](https://github.com/AI-sandbox/snputils/actions/workflows/ci-cd.yml/badge.svg?event=release)](https://github.com/AI-sandbox/snputils/actions/workflows/ci-cd.yml)

**snputils** is a Python package designed to ease the processing and analysis of genomic datasets, while handling all the complexities of different genome formats and operations very efficiently. The library provides robust tools for handling sequencing and ancestry data, with a focus on performance, ease of use, and advanced visualization capabilities. 

Developed in collaboration between Stanford University's Department of Biomedical Data Science, UC Santa Cruz Genomics Institute, and more collaborators worldwide.

**snputils** is stable and ready for production workflows. The core API is documented, tested, and suitable for day-to-day genomic analysis. The project is actively maintained: we ship regular releases, welcome contributions, and continue to extend format support, analyses, and performance.

## Why snputils?

- One API across genotype, local ancestry, global ancestry, phenotype, and IBD data
- Fast readers and writers for common population-genetics formats
- In-memory Python workflows and file-backed CLI workflows in the same package
- Ancestry-aware analyses including PCA and advanced alternatives, admixture mapping, and ancestry-specific allele frequencies
- Built-in plotting for embeddings, local ancestry, admixture, and association results

## Quickstart

```python
import snputils as su

snp = su.read_snp("cohort.vcf.gz")                    # VCF, BGEN, BED, PGEN
snp = snp.filter_biallelic_variants()
snp.save("cohort.pgen")                               # convert to PGEN

lai = su.read_lai("local_ancestry.msp")               # MSP or FLARE local ancestry
adm = su.read_admixture("admixture_prefix")           # ADMIXTURE-style global ancestry
pheno = su.read_pheno("phenotypes.tsv", col="trait")
ibd = su.read_ibd("segments.hapibd")

pcs = su.PCA(n_components=2).fit_transform(snp)
afr_af = snp.allele_freq(ancestry="AFR", laiobj=lai)
gwas = su.run_gwas(pheno, snp)
admix = su.run_admixture_mapping(pheno, lai)

su.viz.scatter(pcs, "labels.tsv", save_path="pca.png", show=False)
su.viz.chromosome_painting(lai, "chr_paintings/")
su.viz.qq_plot(gwas)
su.viz.manhattan_plot(admix)
```

## Installation

Basic installation using pip:
```bash
pip install snputils
```

Optionally, for PyTorch-backed features, install with the `[torch]` extra:
```bash
pip install 'snputils[torch]'
```

Optional extras:

- `pip install 'snputils[tests]'` for the test stack
- `pip install 'snputils[docs]'` for local documentation builds
- `pip install 'snputils[demos]'` for notebook demos

## Key Features

### File Format Support

**snputils** provides high-level dispatchers like `read_snp`, `read_lai`, `read_admixture`, `read_pheno`, and `read_ibd`, plus explicit reader and writer classes when you need finer control.

- **VCF**: Support for `.vcf` and `.vcf.gz` files
- **BGEN**: Support for `.bgen` files
- **PLINK1**: Support for `.bed`, `.bim`, `.fam` filesets
- **PLINK2**: Support for `.pgen`, `.pvar`, `.psam` filesets
- **GRG**: Read and write graph-based genome representation files
- **Local Ancestry**: Handle `.msp` and FLARE `.anc.vcf.gz` local ancestry formats
- **Global Ancestry / ADMIXTURE**: Read and write `.Q` and `.P` files
- **IBD**: Read `hap-IBD` and `ancIBD` outputs into a unified object

### Data Objects and Utilities

- **SNPObject** for genotype data, including filtering, saving, and allele-frequency helpers
- **LocalAncestryObject** and **GlobalAncestryObject** for ancestry-aware workflows
- **PhenotypeObject**, **MultiPhenotypeObject**, and **CovariateObject** for trait data
- **IBDObject** for segment filtering and ancestry-restricted trimming
- Synthetic dataset builders for SNP, mdPCA, maasMDS, chromosome-painting, admixture, and GRG examples
- Conversion helpers such as VCF-to-GRG workflows

### Processing & Analysis Tools

- **Basic manipulation**
  - Filter variants and samples, correct SNP flips, and filter ambiguous SNPs
  - Compute cohort and ancestry-specific allele frequencies via `SNPObject.allele_freq(...)`
  - Stream allele frequencies with `snputils.stats.allele_freq_stream(...)` for memory efficiency

- **Dimensionality reduction**
  - Standard PCA with optional PyTorch acceleration
  - Missing-data PCA (`mdPCA`)
  - Multi-array ancestry-specific MDS (`maasMDS`)

- **Population-genetic statistics**
  - Compute $D$, $f_2$, $f_3$, $f_4$, the $f_4$-ratio, and $F_{ST}$ (Hudson, Weir-Cockerham, and Tsallis $F_{q}$)
  - Block jackknife standard errors where applicable
  - Optional ancestry masking in relevant workflows

- **Association analysis**
  - GWAS on SNP dosages for binary and quantitative traits
  - Admixture mapping from local ancestry dosage
  - Built-in Manhattan and Q–Q plotting utilities

- **IBD and ancestry-aware trimming**
  - Unified IBD ingestion from common upstream tools
  - Segment filtering and ancestry-restricted trimming using local ancestry

- **Simulation**
  - Lightweight haplotype-based simulation of admixed mosaics from founder haplotypes

### Visualization

- Scatter plots for PCA, mdPCA, and maasMDS embeddings
- Global ancestry bar plots
- Local ancestry visualization
  - Chromosome painting
  - Dataset-level cohort summaries
- Association plots
  - Manhattan plots
  - Q–Q plots

<p align="center">
    <img src="https://raw.githubusercontent.com/AI-sandbox/snputils/refs/heads/main/assets/snputils_composite.png" width="800">
</p>


### Performance

- Fast file I/O through built-in methods or optimized wrappers (e.g., [Pgenlib](https://pypi.org/project/Pgenlib/) for PLINK files)
- Memory-efficient operations using [NumPy](https://numpy.org) and [Polars](https://pola.rs), including streaming workflows
- Optional GPU acceleration via [PyTorch](https://pytorch.org) for computationally intensive tasks
- Support for large-scale genomic datasets through efficient memory management

Our benchmark demonstrates superior performance compared to existing tools:

<p align="center">
    <img src="https://raw.githubusercontent.com/AI-sandbox/snputils/refs/heads/main/benchmark/readers_benchmark.png" width="800">
</p>

*Reading time and peak-memory comparison for chromosome 22 data across different tools. See the [benchmark directory](https://github.com/AI-sandbox/snputils/tree/main/benchmark) for detailed methodology and results.*

## Command-Line Interface

Installing the package provides a `snputils` command for common file-backed workflows:

```bash
snputils --help
snputils --version
```

Available subcommands include:

- `pca`: run standard PCA and save coordinates/components and a scatter plot.
- `mdpca`: run missing-data PCA and save an embedding table.
- `maasmds`: run ancestry-specific MDS and save an embedding table.
- `admixture-map`: run admixture mapping from phenotype and local ancestry files.
- `gwas`: run variant-level association testing from phenotype and genotype files.
- `simulate`: simulate admixed haplotype batches from phased founder haplotypes.
- `plot-manhattan` and `plot-qq`: render association result visualizations.

The Python API remains the full surface for low-level readers/writers, object manipulation, IBD filtering and trimming, f-statistics, allele-frequency helpers, custom visualizations, and notebook-oriented workflows. Use the CLI when a workflow naturally starts from files and produces files; use Python when you need programmatic composition or in-memory objects.

## Documentation and Examples

- **Documentation**: [docs.snputils.org](https://docs.snputils.org)
- **Quickstart**: [Quickstart guide](https://docs.snputils.org/en/latest/quickstart.html)
- **Tutorials**: PCA, mdPCA, maasMDS, SNP objects, allele frequency, local ancestry visualization, admixture mapping, and GRG workflows
- **API Reference**: Readers, writers, data objects, processing classes, statistics, datasets, and visualization helpers
- **Issues and feature requests**: [GitHub Issues](https://github.com/AI-sandbox/snputils/issues)

## Public API Highlights

Top-level imports include:

- Readers and objects: `read_snp`, `read_lai`, `read_admixture`, `read_ibd`, `read_pheno`, `SNPObject`, `LocalAncestryObject`, `GlobalAncestryObject`, `IBDObject`
- Analysis: `PCA`, `mdPCA`, `maasMDS`, `run_gwas`, `run_admixture_mapping`, `allele_freq_stream`
- Datasets: `load_dataset`, `available_datasets_list`, `build_synthetic_*`
- Visualization namespace: `snputils.viz`

## Citation

If you use **snputils** in your research, please cite [our paper](https://www.biorxiv.org/content/10.64898/2026.02.28.708618):

```bibtex
@article{snputils2026,
    author    = {Bonet, David and Comajoan Cara, Marçal and Barrabés, Míriam and Smeriglio, Riccardo and Agrawal, Devang and Aounallah, Khaled and Geleta, Margarita and Dominguez Mantes, Albert and Thomassin, Christophe and Shanks, Cole and Huang, Edward C. and Franquesa Monés, Marc and Luis, Aina and Saurina, Joan and Perera, Maria and López, Cayetana and Sabat, Benet Oriol and Abante, Jordi and Moreno-Grau, Sonia and Mas Montserrat, Daniel and Ioannidis, Alexander G.},
    title     = {{snputils}: A High-Performance {Python} Library for Genetic Variation and Population Structure},
    year      = {2026},
    doi       = {10.64898/2026.02.28.708618},
    url       = {https://www.biorxiv.org/content/10.64898/2026.02.28.708618},
    journal   = {bioRxiv},
    publisher = {Cold Spring Harbor Laboratory},
}
```

## Acknowledgments

We would like to thank the open-source packages that make **snputils** possible.
