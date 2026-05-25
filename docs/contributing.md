# Contributing

Contributions to **snputils** are welcome. The project is developed in the open at [github.com/AI-sandbox/snputils](https://github.com/AI-sandbox/snputils).

## Development setup

Clone the repository and install the package in editable mode with test and documentation dependencies:

```bash
git clone https://github.com/AI-sandbox/snputils.git
cd snputils
python -m venv .venv
source .venv/bin/activate
pip install -e ".[tests,docs,torch]"
```

Optional extras such as `[demos]` are listed in {doc}`installation`.

## Running tests

```bash
python -m pytest
```

The CI workflow runs the test suite on supported Python versions before releases.

## Building documentation

```bash
sphinx-build -b html docs docs/_build/html
```

See {doc}`installation` for details. User-facing docs live in `docs/`; API pages use Sphinx autodoc against the Python sources.

## Pull requests

1. Open an issue or comment on an existing one to discuss substantial changes.
2. Fork the repository and create a feature branch from `main`.
3. Add or update tests for behavioral changes.
4. Update relevant documentation under `docs/` when you change public API or CLI behavior.
5. Ensure `pytest` passes and, when docs change, that `sphinx-build` completes without errors or warnings you introduced.
6. Open a pull request with a concise summary and test plan.

## Code style

Match the surrounding module: type hints where already used, Google/NumPy-style docstrings for public functions, and minimal scope per change. Avoid drive-by refactors unrelated to the issue at hand.
