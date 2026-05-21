"""PCA, MDS, and related dimensionality-reduction classes.

Optional TSV/CSV export of fitted embeddings is implemented in
:mod:`snputils.processing.dimred_tabular` and can be enabled via
``embedding_table_path`` on :class:`~snputils.processing.pca.PCA`,
:class:`~snputils.processing.mdpca.mdPCA`, and
:class:`~snputils.processing.maasmds.maasMDS`.

Public names are re-exported here via lazy loading (:pep:`562`) so that
``import snputils.processing`` remains lightweight.
"""

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "TorchPCA",
    "PCA",
    "maasMDS",
    "mdPCA",
    "save_embedding_table",
    "save_embedding_table_from_model",
    "try_save_embedding_table",
    "build_embedding_dataframe",
    "embedding_dataframe_from_model",
    "embedding_column_names",
    "pca_row_haplotype_ids",
]


if TYPE_CHECKING:
    from .maasmds import maasMDS
    from .mdpca import mdPCA
    from .pca import PCA, TorchPCA
    from .dimred_tabular import (
        build_embedding_dataframe,
        embedding_dataframe_from_model,
        embedding_column_names,
        pca_row_haplotype_ids,
        save_embedding_table,
        save_embedding_table_from_model,
        try_save_embedding_table,
    )


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
    if name in {
        "build_embedding_dataframe",
        "embedding_dataframe_from_model",
        "embedding_column_names",
        "pca_row_haplotype_ids",
        "save_embedding_table",
        "save_embedding_table_from_model",
        "try_save_embedding_table",
    }:
        module = import_module(".dimred_tabular", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
