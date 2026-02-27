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

*Note: **snputils** is under active development. While the core API is stabilizing, we are continuously adding features, optimizing performance, and expanding format support.*

## Installation

Basic installation using pip:
```bash
pip install snputils
```

Optionally, for GPU-accelerated functionalities, install the package with the `[gpu]` extra:
```bash
pip install 'snputils[gpu]'
```

## Key Features

### Ease of Use

**snputils** is designed to be user-friendly and intuitive, with a simple API that allows you to quickly load, process, and visualize genomic data. For example, reading a whole genome VCF file is as simple as:
```python
import snputils as su
snpobj = su.read_snp("path/to/file.vcf.gz")
```

Similarly, reading BED or PGEN filesets is straightforward:
```python
snpobj = su.read_snp("path/to/file.pgen")
```

Working with ancestry files, performing processing operations, and creating visualizations is just as straightforward. See the [demos directory](https://github.com/AI-sandbox/snputils/tree/main/demos) for examples.

### File Format Support
**snputils** aims to provide the fastest available readers and writers for various genomic data formats:
- **VCF**: Support for `.vcf` and `.vcf.gz` files
- **PLINK1**: Support for `.bed`, `.bim`, `.fam` filesets
- **PLINK2**: Support for `.pgen`, `.pvar`, `.psam` filesets
- **Local Ancestry**: Handle `.msp` local ancestry format
- **Admixture**: Read and write `.Q` and `.P` files

### Processing & Analysis Tools
- **Basic Data Manipulation**
  - Filter variants and samples, correct SNP flips, and filter out ambiguous SNPs
  - Standardized querying across genotype, local ancestry, global ancestry, and IBD data

- **Dimensionality Reduction**
  - Standard PCA with optional GPU acceleration
  - Missing-data PCA (mdPCA)
  - Multi-array ancestry-specific MDS (maasMDS)

- **Population Genetic Statistics**
  - Compute $D$, $f_2$, $f_3$, $f_4$, the $f_4$-ratio, and $F_{ST}$ (Hudson and Weir-Cockerham)
  - Includes block jackknife standard errors and optional ancestry masking

- **Identity-by-Descent (IBD) & Relatedness**
  - Read `hap-IBD` and `ancIBD` outputs into a unified format
  - Fast filtering and ancestry-restricted segment trimming using local ancestry

- **Admixture Analysis & Simulation**
  - **Admixture Mapping:** Locus-by-locus regression of local ancestry dosage on traits
  - **Simulation:** Lightweight haplotype-based simulation of admixed mosaics from real founder haplotypes

### Visualization
- Interactive global ancestry bar plots
- Detailed scatter plots of PCA, mdPCA, and maasMDS
- Admixture mapping Manhattan plots
- Local ancestry visualization 
  - Chromosome painting (with [Tagore](https://github.com/jordanlab/tagore))
  - Dataset-level

<p align="center">
    <img src="https://raw.githubusercontent.com/AI-sandbox/snputils/refs/heads/main/assets/lai_dataset_level.png" width="800">
</p>


### Performance

- Fast file I/O through built-in methods or optimized wrappers (e.g., [Pgenlib](https://pypi.org/project/Pgenlib/) for PLINK files)
- Memory-efficient operations using [NumPy](https://numpy.org) and [Polars](https://pola.rs)
- Optional GPU acceleration via [PyTorch](https://pytorch.org) for computationally intensive tasks
- Support for large-scale genomic datasets through efficient memory management

Our benchmark demonstrates superior performance compared to existing tools:

<p align="center">
    <img src="https://raw.githubusercontent.com/AI-sandbox/snputils/refs/heads/main/benchmark/benchmark.png" width="800">
</p>

*Reading performance comparison for chromosome 22 data across different tools. See the [benchmark directory](https://github.com/AI-sandbox/snputils/tree/main/benchmark) for detailed methodology and results.*

The **snputils** package is continuously updated with new features and improvements.

## Documentation & Support

- **Documentation**: Comprehensive API reference at [docs.snputils.org](https://docs.snputils.org).
- **Examples & Tutorials**: Check out our interactive notebooks in the [demos directory](https://github.com/AI-sandbox/snputils/tree/main/demos).
- **Issues & Community**: Report bugs, ask questions, or request features via [GitHub Issues](https://github.com/AI-sandbox/snputils/issues).

## Acknowledgments

We would like to thank the open-source Python packages that make **snputils** possible: matplotlib, NumPy, pandas, Pgenlib, polars, pong, PyTorch, scikit-allel, scikit-learn, Tagore.

## Citation

If you use **snputils** in your research, please cite:

> Bonet, D.\*, Comajoan Cara, M.\*, Barrabés, M.\*, Smeriglio, R., Agrawal, D., Dominguez Mantes, A., López, C., Thomassin, C., Calafell, A., Luis, A., Saurina, J., Franquesa, M., Perera, M., Geleta, M., Jaras, A., Sabat, B. O., Abante, J., Moreno-Grau, S., Mas Montserrat, D., Ioannidis, A. G., snputils: A Python library for processing diverse genomes. Annual Meeting of The American Society of Human Genetics, November 2024, Denver, Colorado, USA. \*Equal contribution.

Journal paper coming soon!


