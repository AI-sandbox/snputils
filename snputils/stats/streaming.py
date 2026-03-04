from __future__ import annotations

from os import PathLike
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from snputils._utils.allele_freq import aggregate_pop_allele_freq


def _slice_variant_axis(arr: Any, start: int, stop: int) -> Any:
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr
    return arr[start:stop, ...]


def _iter_snpobject_chunks(snpobj: Any, chunk_size: int) -> Iterator[Any]:
    """
    Yield variant-axis SNPObject chunks from an in-memory SNPObject.
    """
    from snputils.snp.genobj.snpobj import SNPObject

    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1.")

    n_snps = snpobj.n_snps
    for start in range(0, n_snps, int(chunk_size)):
        stop = min(start + int(chunk_size), n_snps)
        yield SNPObject(
            calldata_gt=_slice_variant_axis(snpobj.calldata_gt, start, stop),
            samples=None if snpobj.samples is None else np.asarray(snpobj.samples),
            variants_ref=_slice_variant_axis(snpobj.variants_ref, start, stop),
            variants_alt=_slice_variant_axis(snpobj.variants_alt, start, stop),
            variants_chrom=_slice_variant_axis(snpobj.variants_chrom, start, stop),
            variants_filter_pass=_slice_variant_axis(snpobj.variants_filter_pass, start, stop),
            variants_id=_slice_variant_axis(snpobj.variants_id, start, stop),
            variants_pos=_slice_variant_axis(snpobj.variants_pos, start, stop),
            variants_qual=_slice_variant_axis(snpobj.variants_qual, start, stop),
            calldata_lai=_slice_variant_axis(snpobj.calldata_lai, start, stop),
            ancestry_map=snpobj.ancestry_map,
        )


def _canonical_chromosome(chromosome: Any) -> str:
    text = str(chromosome).strip()
    lower = text.lower()
    if lower.startswith("chr"):
        text = text[3:]
    text = text.strip()
    if text.isdigit():
        return str(int(text))
    return text.lower()


class _IterWindowsLAIMapper:
    def __init__(
        self,
        lai_reader: Any,
        *,
        chunk_size: int,
        iter_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("lai_chunk_size must be >= 1.")
        if not hasattr(lai_reader, "iter_windows") or not callable(getattr(lai_reader, "iter_windows")):
            raise TypeError("LAI reader must implement `iter_windows(...)`.")

        self._window_iter = iter(
            lai_reader.iter_windows(
                chunk_size=int(chunk_size),
                **(iter_kwargs or {}),
            )
        )
        self._window_chrom: Optional[np.ndarray] = None
        self._window_pos: Optional[np.ndarray] = None
        self._window_lai: Optional[np.ndarray] = None
        self._row_idx = 0
        self._exhausted = False

        self._active_window_chrom: Optional[str] = None
        self._completed_window_chroms: Set[str] = set()
        self._seen_snp_chroms: Set[str] = set()
        self._last_snp_chrom: Optional[str] = None
        self._last_snp_pos: Optional[int] = None

        self._n_samples_lai: Optional[int] = None

    def _load_next_window_chunk(self) -> bool:
        prev_active = self._active_window_chrom

        while True:
            try:
                chunk = next(self._window_iter)
            except StopIteration:
                self._exhausted = True
                self._window_chrom = None
                self._window_pos = None
                self._window_lai = None
                self._active_window_chrom = None
                return False

            lai = np.asarray(chunk.get("lai", None))
            if lai.size == 0:
                continue
            if lai.ndim != 2:
                raise ValueError("LAI `iter_windows` chunks must provide 2D `lai` arrays.")
            if lai.shape[1] % 2 != 0:
                raise ValueError("LAI `iter_windows` chunks must contain an even number of haplotype columns.")

            chrom = np.asarray(chunk.get("chromosomes", None))
            if chrom.ndim != 1 or chrom.shape[0] != lai.shape[0]:
                raise ValueError(
                    "LAI `iter_windows` chunks must provide `chromosomes` with one value per window."
                )
            chrom = np.asarray([_canonical_chromosome(c) for c in chrom], dtype=object)

            physical_pos = chunk.get("physical_pos", None)
            if physical_pos is None:
                raise ValueError(
                    "LAI reader windows must include physical positions (`physical_pos`) "
                    "to map ancestry to SNP chunks."
                )
            physical_pos = np.asarray(physical_pos)
            if physical_pos.ndim != 2 or physical_pos.shape[1] != 2 or physical_pos.shape[0] != lai.shape[0]:
                raise ValueError(
                    "LAI `iter_windows` chunks must provide `physical_pos` with shape (n_windows, 2)."
                )

            n_samples_lai = lai.shape[1] // 2
            if self._n_samples_lai is None:
                self._n_samples_lai = n_samples_lai
            elif self._n_samples_lai != n_samples_lai:
                raise ValueError("Inconsistent number of LAI samples across LAI window chunks.")

            self._window_chrom = chrom
            self._window_pos = physical_pos.astype(np.int64, copy=False)
            self._window_lai = lai
            self._row_idx = 0

            next_active = str(self._window_chrom[0])
            if prev_active is not None and prev_active != next_active:
                self._completed_window_chroms.add(prev_active)
            self._active_window_chrom = next_active
            return True

    def _current_window(self) -> Optional[Tuple[str, int, int, np.ndarray]]:
        if self._window_chrom is None:
            if not self._load_next_window_chunk():
                return None
        if self._window_chrom is None or self._window_pos is None or self._window_lai is None:
            return None
        chrom = str(self._window_chrom[self._row_idx])
        start = int(self._window_pos[self._row_idx, 0])
        end = int(self._window_pos[self._row_idx, 1])
        lai_row = self._window_lai[self._row_idx]
        return chrom, start, end, lai_row

    def _advance_window(self) -> bool:
        if self._window_chrom is None:
            return self._load_next_window_chunk()

        current_chrom = str(self._window_chrom[self._row_idx])
        self._row_idx += 1
        if self._window_chrom is not None and self._row_idx < len(self._window_chrom):
            next_chrom = str(self._window_chrom[self._row_idx])
            if current_chrom != next_chrom:
                self._completed_window_chroms.add(current_chrom)
            self._active_window_chrom = next_chrom
            return True

        self._active_window_chrom = current_chrom
        return self._load_next_window_chunk()

    def _assert_snp_order(self, chrom: str, pos: int) -> None:
        if self._last_snp_chrom is None:
            self._last_snp_chrom = chrom
            self._last_snp_pos = pos
            self._seen_snp_chroms.add(chrom)
            return

        if chrom == self._last_snp_chrom:
            if self._last_snp_pos is not None and pos < self._last_snp_pos:
                raise ValueError(
                    "SNP chunks must be sorted by ascending position within chromosome "
                    "when using a streaming LAI reader."
                )
        else:
            if chrom in self._seen_snp_chroms:
                raise ValueError(
                    "SNP chunks cannot revisit chromosomes when using a streaming LAI reader "
                    "(pass an in-memory LocalAncestryObject instead)."
                )
            self._seen_snp_chroms.add(chrom)

        self._last_snp_chrom = chrom
        self._last_snp_pos = pos

    def map_chunk(
        self,
        *,
        variants_chrom: np.ndarray,
        variants_pos: np.ndarray,
        n_samples_expected: int,
    ) -> np.ndarray:
        variants_chrom = np.asarray(variants_chrom)
        variants_pos = np.asarray(variants_pos)
        if variants_chrom.ndim != 1 or variants_pos.ndim != 1 or variants_chrom.shape[0] != variants_pos.shape[0]:
            raise ValueError("`variants_chrom` and `variants_pos` must be 1D arrays with matching length.")

        if self._n_samples_lai is None:
            _ = self._current_window()
        if self._n_samples_lai is None:
            raise ValueError("No LAI windows available in the provided LAI reader.")
        if self._n_samples_lai != int(n_samples_expected):
            raise ValueError(
                "LAI sample count does not match SNP sample count "
                f"({self._n_samples_lai} vs {n_samples_expected})."
            )

        n_snps = variants_pos.shape[0]
        calldata_lai = np.full((n_snps, n_samples_expected, 2), -1, dtype=np.int16)

        for snp_idx in range(n_snps):
            chrom = _canonical_chromosome(variants_chrom[snp_idx])
            try:
                pos = int(variants_pos[snp_idx])
            except Exception as exc:
                raise ValueError("SNP variant positions must be integer-like when using LAI reader streaming.") from exc

            self._assert_snp_order(chrom, pos)

            while True:
                window = self._current_window()
                if window is None:
                    break

                win_chrom, win_start, win_end, win_lai = window

                if win_chrom == chrom:
                    if pos < win_start:
                        break
                    if pos <= win_end:
                        calldata_lai[snp_idx] = np.asarray(win_lai).reshape(n_samples_expected, 2)
                        break
                else:
                    if chrom in self._completed_window_chroms:
                        raise ValueError(
                            "SNP chunks are not in the same chromosome order as LAI windows. "
                            "Use an in-memory LocalAncestryObject for out-of-order SNP access."
                        )

                if not self._advance_window():
                    break

        return calldata_lai


def _coerce_lai_source(
    laiobj: Any,
    *,
    lai_chunk_size: int,
    lai_iter_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Any], Optional[_IterWindowsLAIMapper]]:
    if laiobj is None:
        return None, None

    source = laiobj
    if isinstance(source, (str, PathLike)):
        from snputils.ancestry.io.local.read import LAIReader

        source = LAIReader(source)

    if hasattr(source, "iter_windows") and callable(getattr(source, "iter_windows")):
        return None, _IterWindowsLAIMapper(
            source,
            chunk_size=lai_chunk_size,
            iter_kwargs=lai_iter_kwargs,
        )

    if hasattr(source, "convert_to_snp_level") and callable(getattr(source, "convert_to_snp_level")):
        return source, None

    if hasattr(source, "read") and callable(getattr(source, "read")):
        loaded = source.read()
        if hasattr(loaded, "convert_to_snp_level") and callable(getattr(loaded, "convert_to_snp_level")):
            return loaded, None

    raise TypeError(
        "`laiobj` must be one of: LocalAncestryObject, LAI reader/path, "
        "or an object implementing `read()` that returns a LocalAncestryObject."
    )


def allele_freq_stream(
    data: Any,
    *,
    chunk_size: int = 10_000,
    sample_labels: Optional[Sequence[Any]] = None,
    ancestry: Optional[Union[str, int]] = None,
    laiobj: Optional[Any] = None,
    lai_chunk_size: int = 1024,
    lai_iter_kwargs: Optional[Dict[str, Any]] = None,
    return_counts: bool = False,
    as_dataframe: bool = False,
    **iter_kwargs,
) -> Any:
    """
    Compute allele frequencies in variant chunks.

    Args:
        data:
            One of:
            - in-memory SNPObject
            - a reader implementing `iter_read(...)`
            - an iterable yielding SNPObject chunks
        chunk_size:
            Number of SNPs per chunk.
        sample_labels:
            Population label per sample. If None, computes cohort-level frequencies.
        ancestry:
            Optional ancestry code for ancestry-specific masking.
        laiobj:
            Optional LAI source used to derive SNP-level LAI when missing in chunks.
            Supported inputs:
            - in-memory LocalAncestryObject
            - LAI reader implementing `iter_windows(...)`
            - path to an LAI file understood by `snputils.ancestry.io.local.read.LAIReader`
        lai_chunk_size:
            Number of LAI windows per chunk when `laiobj` is a streaming LAI reader/path.
        lai_iter_kwargs:
            Optional kwargs forwarded to `laiobj.iter_windows(...)`.
        return_counts:
            If True, also return called haplotype counts.
        as_dataframe:
            If True, return pandas DataFrames.
        **iter_kwargs:
            Additional args forwarded to `iter_read(...)` when `data` is a reader.
    """
    try:
        from snputils.snp.genobj.snpobj import SNPObject
    except Exception:
        SNPObject = None  # type: ignore

    grouped_output = sample_labels is not None

    if SNPObject is not None and isinstance(data, SNPObject):
        chunk_iter: Iterable[Any] = _iter_snpobject_chunks(data, chunk_size=chunk_size)
    elif hasattr(data, "iter_read") and callable(getattr(data, "iter_read")):
        chunk_iter = data.iter_read(chunk_size=chunk_size, **iter_kwargs)
    else:
        chunk_iter = iter(data)

    lai_object = None
    lai_window_mapper: Optional[_IterWindowsLAIMapper] = None
    if ancestry is not None and laiobj is not None:
        lai_object, lai_window_mapper = _coerce_lai_source(
            laiobj,
            lai_chunk_size=lai_chunk_size,
            lai_iter_kwargs=lai_iter_kwargs,
        )

    labels = None
    n_samples_ref = None
    pops_ref: Optional[List[Any]] = None
    afs_parts: List[np.ndarray] = []
    counts_parts: List[np.ndarray] = []

    for chunk in chunk_iter:
        if chunk is None or getattr(chunk, "calldata_gt", None) is None:
            continue

        gt_chunk = np.asarray(chunk.calldata_gt)
        if gt_chunk.ndim not in (2, 3):
            raise ValueError("'calldata_gt' must be 2D or 3D array")

        n_samples = gt_chunk.shape[1]
        if n_samples_ref is None:
            n_samples_ref = n_samples
            if sample_labels is None:
                labels = np.repeat("__all__", n_samples_ref)
            else:
                labels = np.asarray(sample_labels)
                if labels.ndim != 1:
                    labels = labels.ravel()
                if labels.shape[0] != n_samples_ref:
                    raise ValueError(
                        "'sample_labels' must have length equal to the number of samples in `calldata_gt`."
                    )
        elif n_samples != n_samples_ref:
            raise ValueError("All chunks must have the same number of samples.")

        calldata_lai = getattr(chunk, "calldata_lai", None)
        if ancestry is not None and calldata_lai is None:
            if lai_window_mapper is not None:
                variants_chrom = getattr(chunk, "variants_chrom", None)
                variants_pos = getattr(chunk, "variants_pos", None)
                if variants_chrom is None or variants_pos is None:
                    raise ValueError(
                        "Ancestry-specific masking with a streaming LAI reader requires "
                        "`variants_chrom` and `variants_pos` on SNP chunks."
                    )
                calldata_lai = lai_window_mapper.map_chunk(
                    variants_chrom=variants_chrom,
                    variants_pos=variants_pos,
                    n_samples_expected=n_samples,
                )
            elif lai_object is not None:
                try:
                    converted_lai = lai_object.convert_to_snp_level(snpobject=chunk, lai_format="3D")
                    calldata_lai = getattr(converted_lai, "calldata_lai", None)
                except Exception:
                    calldata_lai = None

        if ancestry is not None and calldata_lai is None:
            raise ValueError(
                "Ancestry-specific masking requires SNP-level LAI "
                "(provide `calldata_lai` on the chunks or pass `laiobj`)."
            )

        afs_chunk, counts_chunk, pops = aggregate_pop_allele_freq(
            calldata_gt=gt_chunk,
            sample_labels=labels,
            ancestry=ancestry,
            calldata_lai=calldata_lai,
        )

        if pops_ref is None:
            pops_ref = pops
        elif pops_ref != pops:
            raise ValueError("Population labels must be consistent across chunks.")

        afs_parts.append(afs_chunk)
        counts_parts.append(counts_chunk)

    if not afs_parts:
        raise ValueError("No genotype chunks were provided.")

    afs = np.vstack(afs_parts)
    counts = np.vstack(counts_parts)

    if grouped_output:
        freq_out = afs
        count_out = counts
        if as_dataframe:
            import pandas as pd

            freq_out = pd.DataFrame(afs, columns=pops_ref)
            count_out = pd.DataFrame(counts, columns=pops_ref)
    else:
        freq_out = afs[:, 0]
        count_out = counts[:, 0]
        if as_dataframe:
            import pandas as pd

            freq_out = pd.DataFrame({"allele_freq": freq_out})
            count_out = pd.DataFrame({"called_alleles": count_out})

    if return_counts:
        return freq_out, count_out
    return freq_out


__all__ = ["allele_freq_stream"]
