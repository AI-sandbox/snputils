import builtins
import importlib
import sys

import pytest


def _block_torch_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError("No module named 'torch'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def _clear_processing_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    for module_name in ("snputils.processing.pca", "snputils.processing"):
        sys.modules.pop(module_name, None)

    import snputils

    monkeypatch.delitem(snputils.__dict__, "processing", raising=False)


def test_processing_import_and_sklearn_pca_do_not_require_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_processing_modules(monkeypatch)
    _block_torch_imports(monkeypatch)

    processing = importlib.import_module("snputils.processing")

    assert processing.mdPCA.__name__ == "mdPCA"
    assert processing.PCA().backend == "sklearn"
    assert processing.PCA(backend="sklearn").backend == "sklearn"


def test_pytorch_pca_reports_optional_torch_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_processing_modules(monkeypatch)
    _block_torch_imports(monkeypatch)
    processing = importlib.import_module("snputils.processing")

    with pytest.raises(ImportError, match="snputils\\[torch\\]"):
        processing.PCA(backend="pytorch")

    with pytest.raises(ImportError, match="snputils\\[torch\\]"):
        processing.TorchPCA()
