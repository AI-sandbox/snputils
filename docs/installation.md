# Installation

## Requirements

- **Python** 3.9 or newer (3.9–3.14 supported)
- **pip** with a recent setuptools build backend

Core dependencies (NumPy, pandas, scikit-learn, Polars, matplotlib, Pgenlib, bgen, and others) are installed automatically with the package. See [`pyproject.toml`](https://github.com/AI-sandbox/snputils/blob/main/pyproject.toml) for the full list.

## PyPI install

Install the latest release from PyPI:

```bash
pip install snputils
```

This installs the library, the `snputils` command-line tool, and readers/writers for VCF, BGEN, PLINK BED/PGEN, MSP, FLARE, ADMIXTURE, phenotype, and IBD formats.

## Optional extras

Install feature-specific dependencies with pip extras:

| Extra | Command | Purpose |
|-------|---------|---------|
| `torch` | `pip install "snputils[torch]"` | GPU-accelerated PCA (`TorchPCA`) and the `simulate` CLI |
| `docs` | `pip install "snputils[docs]"` | Build this documentation locally |
| `demos` | `pip install "snputils[demos]"` | Run or edit tutorial notebooks |
| `tests` | `pip install "snputils[tests]"` | pytest and coverage tooling |

Extras can be combined: `pip install "snputils[torch,docs]"`.

## Format-specific notes

- **PLINK2 PGEN** — uses [Pgenlib](https://pypi.org/project/Pgenlib/), included as a core dependency.
- **GRG** — reading and writing genotype representation graphs requires [pygrgl](https://github.com/aprilweilab/grgl#installing-from-pip). Install it separately.
- **PyTorch workflows** — install the `torch` extra before using `TorchPCA`, `OnlineSimulator`, or `snputils simulate`.

## Build documentation locally

From a clone of the repository:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser. The same build runs in CI when documentation is published to [docs.snputils.org](https://docs.snputils.org).

## Getting help

- **Documentation**: [docs.snputils.org](https://docs.snputils.org)
- **Source code**: [github.com/AI-sandbox/snputils](https://github.com/AI-sandbox/snputils)
- **Issues and feature requests**: [GitHub Issues](https://github.com/AI-sandbox/snputils/issues)
