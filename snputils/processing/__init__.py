from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["TorchPCA", "PCA", "maasMDS", "mdPCA"]


if TYPE_CHECKING:
    from .maasmds import maasMDS
    from .mdpca import mdPCA
    from .pca import PCA, TorchPCA


def __getattr__(name):
    if name in {"TorchPCA", "PCA"}:
        module = import_module(".pca", __name__)
        return getattr(module, name)
    if name == "maasMDS":
        module = import_module(".maasmds", __name__)
        return getattr(module, name)
    if name == "mdPCA":
        module = import_module(".mdpca", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
