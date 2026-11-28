"""PLINK sample tables and their correspondence to binary genotype columns."""

import gzip
import io

import numpy as np
import polars as pl


def open_plink_text(filename, mode="rt"):
    options = {"encoding": "utf-8"} if "t" in mode else {}
    if filename.endswith(".zst"):
        import zstandard
        return zstandard.open(filename, mode, **options)
    if filename.endswith(".gz"):
        return gzip.open(filename, mode, **options)
    return open(filename, mode, **options)


def read_sample_table(filename, separator, *, headerless_columns, schema, psam=False):
    with io.BufferedReader(open_plink_text(filename, "rb")) as stream:
        first = stream.readline()
        if separator and not separator.isspace() and separator.encode() in first:
            # Keep support for explicitly delimited, nonstandard filesets.
            fields = first.strip().split(separator.encode())
            source = filename
        else:
            fields = first.split()
            separator = "\t"
            source = io.BytesIO()
            source.write(b"\t".join(fields) + b"\n")
            for line in stream:
                source.write(b"\t".join(line.split()) + b"\n")
            source.seek(0)

    has_header = psam and bool(fields) and fields[0] in {b"#IID", b"IID", b"#FID", b"FID"}
    return pl.read_csv(
        source,
        separator=separator,
        has_header=has_header,
        new_columns=None if has_header else headerless_columns,
        schema_overrides=schema,
        null_values=["NA"],
    ).with_row_index()


def select_samples(table, sample_ids, sample_idxs):
    """Return requested rows, sorted unique pgenlib indices, and optional output order."""
    if sample_ids is not None:
        table = table.filter(pl.col("IID").is_in(sample_ids))
        return table, table["index"].to_numpy(), None
    if sample_idxs is None:
        return table, None, None

    requested = np.atleast_1d(sample_idxs)
    if requested.ndim != 1 or (requested.size and requested.dtype.kind not in "iu"):
        raise ValueError("sample_idxs must be a one-dimensional sequence of integer indices.")
    if np.any(requested < 0) or np.any(requested >= table.height):
        raise ValueError("One or more sample indexes are out of bounds.")
    requested = requested.astype(np.uint32, copy=False)
    rows = table[requested]
    if np.all(requested[1:] > requested[:-1]):
        return rows, requested, None
    native_indices, order = np.unique(requested, return_inverse=True)
    return rows, native_indices, order
