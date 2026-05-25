"""Covariate construction and composition for association analyses."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from snputils.phenotype.genobj.covarobj import CovariateObject

log = logging.getLogger(__name__)


def parse_covar_col_nums(col_nums: Optional[str], n_covariates: int) -> List[int]:
    if n_covariates <= 0:
        return []
    if col_nums is None or str(col_nums).strip() == "":
        return list(range(n_covariates))

    selected: List[int] = []
    seen: Set[int] = set()
    for token in str(col_nums).split(","):
        part = token.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if start <= 0 or end <= 0:
                raise ValueError("--covar-col-nums must be 1-indexed and positive.")
            if end < start:
                raise ValueError(f"Invalid covariate range '{part}' in --covar-col-nums.")
            indices = range(start - 1, end)
        else:
            idx = int(part)
            if idx <= 0:
                raise ValueError("--covar-col-nums must be 1-indexed and positive.")
            indices = [idx - 1]

        for idx0 in indices:
            if idx0 < 0 or idx0 >= n_covariates:
                raise ValueError(
                    f"--covar-col-nums selects column {idx0 + 1}, but only "
                    f"{n_covariates} covariate columns are available."
                )
            if idx0 not in seen:
                seen.add(idx0)
                selected.append(idx0)

    if not selected:
        raise ValueError("--covar-col-nums did not select any covariate column.")
    return selected


def read_covar_file(
    path: Union[str, Path],
    col_nums: Optional[str] = None,
    *,
    variance_standardize: bool = False,
) -> Tuple[List[str], List[str], np.ndarray]:
    """Read a PLINK-style covariate table with an ``IID`` column."""
    covar_df = pd.read_csv(path, sep=r"\s+", dtype=str)
    if covar_df.empty:
        raise ValueError("Empty covariate file.")

    columns = list(covar_df.columns)
    lowered = [str(col).lstrip("#").upper() for col in columns]
    if "IID" not in lowered:
        raise ValueError("Covariate file must include an IID column in the header.")
    iid_col = columns[lowered.index("IID")]
    iid_series = covar_df[iid_col].astype(str)
    if iid_series.duplicated().any():
        raise ValueError("Covariate IID values must be unique.")
    sample_ids = iid_series.tolist()

    covar_start = lowered.index("IID") + 1
    if covar_start >= len(columns):
        raise ValueError("Covariate file header does not contain covariate columns.")
    all_covar_cols = columns[covar_start:]

    selected_rel = parse_covar_col_nums(col_nums, len(all_covar_cols))
    selected_cols = [all_covar_cols[idx] for idx in selected_rel]
    covar_names = [str(name) for name in selected_cols]

    covar_numeric = covar_df[selected_cols].copy()
    for col in selected_cols:
        coldata = covar_numeric[col].astype(str).str.strip()
        if str(col).upper() == "SEX":
            coldata = coldata.str.upper().replace(
                {"MALE": "1", "M": "1", "FEMALE": "2", "F": "2", "UNKNOWN": "", "NA": "", "NAN": ""}
            )
        covar_numeric[col] = pd.to_numeric(coldata, errors="coerce")
    complete = ~covar_numeric.isna().any(axis=1)
    n_dropped = int((~complete).sum())
    if n_dropped > 0:
        log.info(
            "Dropping %d samples with missing covariate values (listwise deletion).",
            n_dropped,
        )
    covar_numeric = covar_numeric.loc[complete].to_numpy(dtype=np.float64, copy=True)
    sample_ids = [sid for sid, keep in zip(sample_ids, complete) if keep]
    if len(sample_ids) == 0:
        raise ValueError(
            "No samples remain after dropping rows with missing covariates. "
            "Check covariate file for non-numeric or missing values."
        )

    covar_matrix = covar_numeric
    if covar_matrix.ndim != 2 or covar_matrix.shape[1] == 0:
        raise ValueError("No covariates selected from covariate file.")

    if variance_standardize:
        means = np.mean(covar_matrix, axis=0)
        stds = np.std(covar_matrix, axis=0, ddof=0)
        if np.any(stds <= 0.0):
            zero_var_names = [covar_names[i] for i in np.where(stds <= 0.0)[0]]
            raise ValueError(
                "Cannot variance-standardize covariates with zero variance: "
                f"{zero_var_names}"
            )
        covar_matrix = (covar_matrix - means[None, :]) / stds[None, :]

    return sample_ids, covar_names, covar_matrix


def covariate_object_from_file(
    path: Union[str, Path],
    col_nums: Optional[str] = None,
) -> CovariateObject:
    samples, names, matrix = read_covar_file(path, col_nums=col_nums)
    return CovariateObject(samples, matrix, covariate_names=names)


def covariate_object_from_embedding(
    model: Any,
    n_components: Optional[int] = None,
    component_names: Optional[Sequence[str]] = None,
) -> CovariateObject:
    """Build covariates from a fitted PCA, mdPCA, or maasMDS model."""
    from snputils.processing.dimred_tabular import (
        _samples_and_haplotypes_from_dimred,
        _to_numpy2d,
        embedding_column_names,
    )

    x = getattr(model, "X_new_", None)
    if x is None:
        raise ValueError("Nothing to tabulate: X_new_ is None (call fit_transform first).")

    ind_ids, hap_list, _ = _samples_and_haplotypes_from_dimred(model)
    x_arr = _to_numpy2d(x)

    if len(ind_ids) == 0:
        ind_ids = [f"row{i}" for i in range(x_arr.shape[0])]

    if hap_list is not None and len(hap_list) == x_arr.shape[0]:
        if not all(str(a) == str(b) for a, b in zip(hap_list, ind_ids)):
            raise ValueError(
                "Embedding has haplotype-expanded rows. Use average_strands=True on PCA "
                "or provide sample-level coordinates."
            )

    if len(ind_ids) != x_arr.shape[0]:
        raise ValueError(
            f"Sample ID count ({len(ind_ids)}) does not match embedding rows ({x_arr.shape[0]})."
        )

    n_comp_total = int(x_arr.shape[1])
    if n_components is None:
        n_sel = n_comp_total
    else:
        n_sel = int(n_components)
        if n_sel <= 0 or n_sel > n_comp_total:
            raise ValueError(
                f"n_components must be between 1 and {n_comp_total}; got {n_sel}."
            )

    x_sel = x_arr[:, :n_sel]

    if component_names is None:
        cls_name = model.__class__.__name__
        style = "MDS" if cls_name == "maasMDS" else "PC"
        names = embedding_column_names(n_sel, style)
    else:
        names = [str(name) for name in component_names]
        if len(names) != n_sel:
            raise ValueError(
                f"component_names length ({len(names)}) must match selected components ({n_sel})."
            )

    return CovariateObject(ind_ids, x_sel, covariate_names=names)


def _resolve_global_ancestry_columns(
    n_ancestries: int,
    columns: Optional[Sequence[int]],
    drop_ancestry: int,
) -> List[int]:
    if columns is not None:
        selected = [int(col) for col in columns]
        for col in selected:
            if col < 0 or col >= n_ancestries:
                raise ValueError(
                    f"Ancestry column index {col} out of range for {n_ancestries} ancestries."
                )
        if len(selected) == 0:
            raise ValueError("At least one ancestry column must be selected.")
        if len(set(selected)) != len(selected):
            raise ValueError("Ancestry column indices must be unique.")
        return selected

    if n_ancestries <= 1:
        raise ValueError(
            "Global ancestry has only one column; cannot drop one and retain covariates."
        )

    drop_idx = n_ancestries - 1 if drop_ancestry == -1 else int(drop_ancestry)
    if drop_idx < 0 or drop_idx >= n_ancestries:
        raise ValueError(
            f"drop_ancestry index {drop_idx} out of range for {n_ancestries} ancestries."
        )

    return [idx for idx in range(n_ancestries) if idx != drop_idx]


def covariate_object_from_global_ancestry(
    admobj: Any,
    columns: Optional[Sequence[int]] = None,
    drop_ancestry: int = -1,
    ancestry_names: Optional[Sequence[str]] = None,
) -> CovariateObject:
    """Build covariates from ADMIXTURE-style global ancestry proportions."""
    q = np.asarray(admobj.Q, dtype=np.float64)
    if q.ndim != 2:
        raise ValueError("GlobalAncestryObject.Q must be a 2D array.")

    n_samples, n_ancestries = q.shape
    samples = [str(sample) for sample in admobj.samples]

    selected_cols = _resolve_global_ancestry_columns(n_ancestries, columns, drop_ancestry)
    matrix = q[:, selected_cols]

    if ancestry_names is None:
        names = [f"ANC{col}" for col in selected_cols]
    else:
        names = [str(name) for name in ancestry_names]
        if len(names) != len(selected_cols):
            raise ValueError(
                f"ancestry_names length ({len(names)}) must match selected columns "
                f"({len(selected_cols)})."
            )

    return CovariateObject(samples, matrix, covariate_names=names)


def merge_covariates(*objs: CovariateObject) -> CovariateObject:
    """Inner-join covariate blocks on sample ID and concatenate columns."""
    if not objs:
        raise ValueError("At least one CovariateObject is required.")
    if len(objs) == 1:
        obj = objs[0]
        return CovariateObject(obj.samples, obj.values, covariate_names=obj.covariate_names)

    common: Optional[Set[str]] = None
    for obj in objs:
        sample_set = set(obj.samples)
        common = sample_set if common is None else common & sample_set
    assert common is not None
    if not common:
        raise ValueError("No overlapping samples between covariate blocks.")

    order = [sample for sample in objs[0].samples if sample in common]
    all_names: List[str] = []
    blocks: List[np.ndarray] = []

    for obj in objs:
        index_map = {sample: idx for idx, sample in enumerate(obj.samples)}
        block = np.array(
            [obj.values[index_map[sample]] for sample in order],
            dtype=np.float64,
        )
        blocks.append(block)
        all_names.extend(obj.covariate_names)

    if len(all_names) != len(set(all_names)):
        duplicates = sorted({name for name in all_names if all_names.count(name) > 1})
        raise ValueError(f"Duplicate covariate names after merge: {duplicates}")

    matrix = np.hstack(blocks)
    return CovariateObject(order, matrix, covariate_names=all_names)


def build_association_covariates(
    *,
    embedding: Any = None,
    n_components: Optional[int] = None,
    global_ancestry: Any = None,
    drop_ancestry: int = -1,
    columns: Optional[Sequence[int]] = None,
    ancestry_names: Optional[Sequence[str]] = None,
    file: Optional[Union[str, Path]] = None,
    col_nums: Optional[str] = None,
) -> CovariateObject:
    """Compose optional embedding, global ancestry, and file covariate blocks."""
    parts: List[CovariateObject] = []

    if embedding is not None:
        parts.append(covariate_object_from_embedding(embedding, n_components=n_components))
    if global_ancestry is not None:
        parts.append(
            covariate_object_from_global_ancestry(
                global_ancestry,
                columns=columns,
                drop_ancestry=drop_ancestry,
                ancestry_names=ancestry_names,
            )
        )
    if file is not None:
        parts.append(covariate_object_from_file(file, col_nums=col_nums))

    if not parts:
        raise ValueError("At least one covariate source must be provided.")
    if len(parts) == 1:
        return parts[0]
    return merge_covariates(*parts)


def write_covar_file(
    covar: CovariateObject,
    path: Union[str, Path],
) -> Path:
    """Write covariates to a PLINK-style whitespace-delimited file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["#FID", "IID", *covar.covariate_names]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(" ".join(header) + "\n")
        for sample, row in zip(covar.samples, covar.values):
            values = " ".join(f"{float(value):.12g}" for value in row.tolist())
            handle.write(f"{sample} {sample} {values}\n")
    return path
