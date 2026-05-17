"""Unified tabular export for SNP dimensionality-reduction embeddings.

Supports :class:`~snputils.processing.pca.PCA`,
:class:`~snputils.processing.mdpca.mdPCA`, and
:class:`~snputils.processing.maasmds.maasMDS` via
:func:`save_embedding_table_from_model` and the low-level
:func:`save_embedding_table` / :func:`build_embedding_dataframe`.
"""

from __future__ import annotations

import pathlib
import warnings
from typing import Any, List, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd

from snputils.snp.genobj.snpobj import SNPObject

ComponentStyle = Literal["PC", "MDS"]


def _to_numpy2d(X: Any) -> np.ndarray:
    """Coerce embedding matrix to a 2D float ndarray (handles torch.Tensor)."""
    if hasattr(X, "detach"):
        return np.asarray(X.detach().cpu().numpy(), dtype=float)
    a = np.asarray(X, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"embedding must be 2D, got shape {a.shape}")
    return a


def embedding_column_names(
    n_components: int,
    style: ComponentStyle = "PC",
) -> List[str]:
    """Return column names for embedding coordinates (1-based, e.g. ``PC1``)."""
    if n_components < 0:
        raise ValueError("n_components must be non-negative")
    if style == "PC":
        prefix = "PC"
    elif style == "MDS":
        prefix = "MDS"
    else:
        raise ValueError(f"unknown component style: {style!r}")
    return [f"{prefix}{i + 1}" for i in range(n_components)]


def pca_row_haplotype_ids(
    snpobj: SNPObject,
    average_strands: bool,
    samples_subset: Optional[Union[int, Sequence[int]]] = None,
) -> List[str]:
    """
    Row identifiers aligned with :meth:`snputils.processing.pca.PCA.fit_transform` output rows.

    Each string uniquely identifies one row of the projection (one haplotype row when
    ``average_strands`` is False). Format uses ``\"indID|strand\"`` with strand ``0`` or ``1``
    when two strands are expanded; use :func:`pca_row_individual_ids` for sample IDs alone.

    Raises:
        ValueError: If ``snpobj.samples`` is missing while IDs are required for export.
    """
    if snpobj.samples is None:
        raise ValueError(
            "Cannot derive PCA row IDs: snpobj.samples is None. "
            "Set sample IDs on the SNPObject or pass explicit row IDs to save_embedding_table."
        )
    s = np.asarray(snpobj.samples, dtype=str)
    if isinstance(samples_subset, int):
        s = s[: int(samples_subset)]
    elif samples_subset is not None:
        s = s[np.asarray(samples_subset, dtype=int)]

    gt = snpobj.genotypes
    if gt.ndim == 2:
        return [str(x) for x in s.tolist()]
    if gt.ndim == 3:
        if average_strands:
            return [str(x) for x in s.tolist()]
        # Same tensor layout as PCA._get_data_from_snpobj: (n_samples, n_snps, 2) then ravel rows.
        n_samples, n_snps, _ = np.transpose(gt.astype(float), (1, 0, 2)).shape
        if len(s) != n_samples:
            raise ValueError(
                f"Length of samples ({len(s)}) does not match genotype matrix ({n_samples} samples)."
            )
        out: List[str] = []
        for r in range(n_samples * 2):
            lin = int(r * n_snps)
            i, _, k = np.unravel_index(lin, (n_samples, n_snps, 2))
            out.append(f"{s[int(i)]}|{int(k)}")
        return out
    raise ValueError(f"genotypes must be 2D or 3D, got {gt.ndim}D")


def pca_row_individual_ids(haplotype_row_ids: Sequence[str]) -> List[str]:
    """Map haplotype-level row IDs to individual IDs (part before ``|`` when present)."""
    return [h.split("|", 1)[0] if "|" in h else str(h) for h in haplotype_row_ids]


def build_embedding_dataframe(
    X_new: Any,
    *,
    ind_ids: Sequence[str],
    haplotype_ids: Optional[Sequence[str]] = None,
    array_index: Optional[Sequence[Any]] = None,
    method: str,
    component_style: ComponentStyle = "PC",
    component_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Build a table of identifiers plus embedding columns.

    Args:
        X_new: Matrix of shape ``(n_rows, n_components)`` (``numpy`` or ``torch``).
        ind_ids: Per-row individual / sample identifiers (length ``n_rows``).
        haplotype_ids: Optional per-row haplotype or replicate IDs; omitted from the frame
            when it would duplicate ``ind_ids`` on every row.
        array_index: Optional per-row genotyping array index (multi-array ``maasMDS``).
        method: Short name stored in ``method`` column (e.g. ``\"pca\"``, ``\"mdpca\"``, ``\"maasmds\"``).
        component_style: ``\"PC\"`` (``PC1``, ...) or ``\"MDS\"`` (``MDS1``, ...).
        component_names: If set, column names for coordinates; length must match ``n_components``.
    """
    x = _to_numpy2d(X_new)
    n_rows, n_comp = x.shape
    if len(ind_ids) != n_rows:
        raise ValueError(
            f"ind_ids length {len(ind_ids)} does not match embedding rows {n_rows}"
        )
    if haplotype_ids is not None and len(haplotype_ids) != n_rows:
        raise ValueError("haplotype_ids length mismatch")
    if array_index is not None and len(array_index) != n_rows:
        raise ValueError("array_index length mismatch")

    if component_names is None:
        names = embedding_column_names(n_comp, component_style)
    else:
        names = list(component_names)
        if len(names) != n_comp:
            raise ValueError("component_names length must match n_components")

    coldata = {
        "indID": list(ind_ids),
        "method": [str(method)] * n_rows,
    }
    if haplotype_ids is not None:
        hap = [str(h) for h in haplotype_ids]
        if any(h != i for h, i in zip(hap, coldata["indID"])):
            coldata["haplotype_id"] = hap
    if array_index is not None:
        coldata["array_index"] = list(array_index)

    df = pd.DataFrame(coldata)
    for j, name in enumerate(names):
        df[name] = x[:, j]
    return df


def save_embedding_table(
    path: Union[str, pathlib.Path],
    X_new: Any,
    *,
    ind_ids: Sequence[str],
    haplotype_ids: Optional[Sequence[str]] = None,
    array_index: Optional[Sequence[Any]] = None,
    method: str = "dimred",
    component_style: ComponentStyle = "PC",
    component_names: Optional[Sequence[str]] = None,
    sep: str = "\t",
    float_format: Optional[str] = "%.8g",
) -> pathlib.Path:
    """
    Write embedding coordinates and identifiers to a CSV/TSV file on disk.

    Compression is inferred from the file suffix (e.g. ``.gz``) via :meth:`pandas.DataFrame.to_csv`.
    Tab separation is used by default; use ``sep=\",\"`` for CSV.

    Returns:
        Resolved path that was written.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = build_embedding_dataframe(
        X_new,
        ind_ids=ind_ids,
        haplotype_ids=haplotype_ids,
        array_index=array_index,
        method=method,
        component_style=component_style,
        component_names=component_names,
    )
    df.to_csv(path, index=False, sep=sep, float_format=float_format)
    return path


def _samples_and_haplotypes_from_dimred(obj: Any) -> tuple[List[str], Optional[List[str]], Optional[np.ndarray]]:
    """Return (ind_ids, haplotype_ids_or_none, array_labels_or_none) from a fitted model."""
    h = getattr(obj, "haplotypes_", None) if hasattr(obj, "haplotypes_") else None
    hap_list: Optional[List[str]]
    if h is None:
        hap_list = None
    else:
        hap_list = [str(x) for x in h]

    if hasattr(obj, "samples_") and obj.samples_ is not None:
        ind_ids = [str(x) for x in obj.samples_]
    elif hap_list is not None:
        ind_ids = hap_list
    else:
        ind_ids = []

    arr = getattr(obj, "array_labels_", None)
    if arr is not None:
        arr = np.asarray(arr)

    return ind_ids, hap_list, arr


def save_embedding_table_from_model(
    obj: Any,
    path: Union[str, pathlib.Path],
    *,
    sep: str = "\t",
    float_format: Optional[str] = "%.8g",
) -> pathlib.Path:
    """
    Write ``obj.X_new_`` using identifiers from a fitted dimensionality-reduction object.

    Expects ``X_new_`` and either ``samples_`` or ``haplotypes_`` (as produced by ``mdPCA`` /
    ``maasMDS`` / ``PCA`` after :meth:`fit_transform`). Includes ``array_index`` when
    ``array_labels_`` is present (``maasMDS``).

    Args:
        obj: A fitted ``PCA``, ``mdPCA``, or ``maasMDS`` instance.
        path: Output path (``.tsv``, ``.csv``, or compressed variants).
    """
    x = getattr(obj, "X_new_", None)
    if x is None:
        raise ValueError("Nothing to save: X_new_ is None (call fit_transform first).")

    cls_name = obj.__class__.__name__
    if cls_name == "PCA":
        style: ComponentStyle = "PC"
        method = "pca"
    elif cls_name == "mdPCA":
        style = "PC"
        method = "mdpca"
    elif cls_name == "maasMDS":
        style = "MDS"
        method = "maasmds"
    else:
        warnings.warn(
            f"Unknown class {cls_name!r}; using component prefix PC and method name {cls_name!r}.",
            UserWarning,
            stacklevel=2,
        )
        style = "PC"
        method = cls_name.lower()

    ind_ids, hap_list, arr = _samples_and_haplotypes_from_dimred(obj)

    if len(ind_ids) == 0:
        n = int(_to_numpy2d(x).shape[0])
        ind_ids = [f"row{i}" for i in range(n)]
        warnings.warn(
            "No sample/haplotype IDs on the model; using placeholder indID row0, row1, ...",
            UserWarning,
            stacklevel=2,
        )

    x_arr = _to_numpy2d(x)
    if len(ind_ids) != x_arr.shape[0]:
        raise ValueError(
            f"indID count ({len(ind_ids)}) does not match X_new_ rows ({x_arr.shape[0]})"
        )

    hap_arg: Optional[List[str]] = None
    if hap_list is not None and len(hap_list) == x_arr.shape[0]:
        if not all(str(a) == str(b) for a, b in zip(hap_list, ind_ids)):
            hap_arg = [str(h) for h in hap_list]

    arr_arg = arr.tolist() if arr is not None and len(arr) == x_arr.shape[0] else None

    return save_embedding_table(
        path,
        x,
        ind_ids=ind_ids,
        haplotype_ids=hap_arg,
        array_index=arr_arg,
        method=method,
        component_style=style,
        sep=sep,
        float_format=float_format,
    )


def try_save_embedding_table(
    obj: Any,
    path: Optional[Union[str, pathlib.Path]],
    *,
    sep: str = "\t",
    float_format: Optional[str] = "%.8g",
) -> Optional[pathlib.Path]:
    """
    Call :func:`save_embedding_table_from_model` when ``path`` is not ``None``.

    Use this from ``fit_transform`` implementations so writing is a no-op unless a path
    was configured on the object.
    """
    if path is None:
        return None
    return save_embedding_table_from_model(obj, path, sep=sep, float_format=float_format)
