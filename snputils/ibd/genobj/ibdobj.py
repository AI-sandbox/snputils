import logging
import copy
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np


log = logging.getLogger(__name__)


class IBDObject:
    """
    A class for Identity-By-Descent (IBD) segment data.
    """

    def __init__(
        self,
        sample_id_1: np.ndarray,
        haplotype_id_1: np.ndarray,
        sample_id_2: np.ndarray,
        haplotype_id_2: np.ndarray,
        chrom: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
        length_cm: Optional[np.ndarray] = None,
        segment_type: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            sample_id_1 (array of shape (n_segments,)): Sample identifiers for the first individual.
            haplotype_id_1 (array of shape (n_segments,)): Haplotype identifiers for the first individual (values in {1, 2}, or -1 if unknown).
            sample_id_2 (array of shape (n_segments,)): Sample identifiers for the second individual.
            haplotype_id_2 (array of shape (n_segments,)): Haplotype identifiers for the second individual (values in {1, 2}, or -1 if unknown).
            chrom (array of shape (n_segments,)): Chromosome identifier for each IBD segment.
            start (array of shape (n_segments,)): Start physical position (1-based, bp) for each IBD segment.
            end (array of shape (n_segments,)): End physical position (1-based, bp) for each IBD segment.
            length_cm (array of shape (n_segments,), optional): Genetic length (cM) for each segment, if available.
        """
        # Store attributes
        self.__sample_id_1 = np.asarray(sample_id_1)
        self.__haplotype_id_1 = np.asarray(haplotype_id_1)
        self.__sample_id_2 = np.asarray(sample_id_2)
        self.__haplotype_id_2 = np.asarray(haplotype_id_2)
        self.__chrom = np.asarray(chrom)
        self.__start = np.asarray(start)
        self.__end = np.asarray(end)
        self.__length_cm = None if length_cm is None else np.asarray(length_cm)
        self.__segment_type = None if segment_type is None else np.asarray(segment_type)

        self._sanity_check()

    def __getitem__(self, key: str) -> Any:
        """
        To access an attribute of the class using the square bracket notation,
        similar to a dictionary.
        """
        try:
            return getattr(self, key)
        except Exception:
            raise KeyError(f"Invalid key: {key}.")

    def __setitem__(self, key: str, value: Any) -> None:
        """
        To set an attribute of the class using the square bracket notation,
        similar to a dictionary.
        """
        try:
            setattr(self, key, value)
        except Exception:
            raise KeyError(f"Invalid key: {key}.")

    @property
    def sample_id_1(self) -> np.ndarray:
        """
        Retrieve `sample_id_1`.

        Returns:
            **array of shape (n_segments,):** Sample identifiers for the first individual.
        """
        return self.__sample_id_1

    @sample_id_1.setter
    def sample_id_1(self, x: Sequence) -> None:
        """
        Update `sample_id_1`.
        """
        self.__sample_id_1 = np.asarray(x)

    @property
    def haplotype_id_1(self) -> np.ndarray:
        """
        Retrieve `haplotype_id_1`.

        Returns:
            **array of shape (n_segments,):** Haplotype identifiers for the first individual (values in {1, 2}).
        """
        return self.__haplotype_id_1

    @haplotype_id_1.setter
    def haplotype_id_1(self, x: Sequence) -> None:
        """
        Update `haplotype_id_1`.
        """
        self.__haplotype_id_1 = np.asarray(x)

    @property
    def sample_id_2(self) -> np.ndarray:
        """
        Retrieve `sample_id_2`.

        Returns:
            **array of shape (n_segments,):** Sample identifiers for the second individual.
        """
        return self.__sample_id_2

    @sample_id_2.setter
    def sample_id_2(self, x: Sequence) -> None:
        """
        Update `sample_id_2`.
        """
        self.__sample_id_2 = np.asarray(x)

    @property
    def haplotype_id_2(self) -> np.ndarray:
        """
        Retrieve `haplotype_id_2`.

        Returns:
            **array of shape (n_segments,):** Haplotype identifiers for the second individual (values in {1, 2}).
        """
        return self.__haplotype_id_2

    @haplotype_id_2.setter
    def haplotype_id_2(self, x: Sequence) -> None:
        """
        Update `haplotype_id_2`.
        """
        self.__haplotype_id_2 = np.asarray(x)

    @property
    def chrom(self) -> np.ndarray:
        """
        Retrieve `chrom`.

        Returns:
            **array of shape (n_segments,):** Chromosome identifier for each IBD segment.
        """
        return self.__chrom

    @chrom.setter
    def chrom(self, x: Sequence) -> None:
        """
        Update `chrom`.
        """
        self.__chrom = np.asarray(x)

    @property
    def start(self) -> np.ndarray:
        """
        Retrieve `start`.

        Returns:
            **array of shape (n_segments,):** Start physical position (1-based, bp) for each IBD segment.
        """
        return self.__start

    @start.setter
    def start(self, x: Sequence) -> None:
        """
        Update `start`.
        """
        self.__start = np.asarray(x)

    @property
    def end(self) -> np.ndarray:
        """
        Retrieve `end`.

        Returns:
            **array of shape (n_segments,):** End physical position (1-based, bp) for each IBD segment.
        """
        return self.__end

    @end.setter
    def end(self, x: Sequence) -> None:
        """
        Update `end`.
        """
        self.__end = np.asarray(x)

    @property
    def length_cm(self) -> Optional[np.ndarray]:
        """
        Retrieve `length_cm`.

        Returns:
            **array of shape (n_segments,):** Genetic length (cM) for each segment if available; otherwise None.
        """
        return self.__length_cm

    @length_cm.setter
    def length_cm(self, x: Optional[Sequence]) -> None:
        """
        Update `length_cm`.
        """
        self.__length_cm = None if x is None else np.asarray(x)

    @property
    def segment_type(self) -> Optional[np.ndarray]:
        """
        Retrieve `segment_type`.

        Returns:
            **array of shape (n_segments,):** Segment type labels (e.g., 'IBD1', 'IBD2'), or None if unavailable.
        """
        return self.__segment_type

    @segment_type.setter
    def segment_type(self, x: Optional[Sequence]) -> None:
        """
        Update `segment_type`.
        """
        self.__segment_type = None if x is None else np.asarray(x)

    @property
    def n_segments(self) -> int:
        """
        Retrieve `n_segments`.

        Returns:
            **int:** The total number of IBD segments.
        """
        return self.__chrom.shape[0]

    @property
    def pairs(self) -> np.ndarray:
        """
        Retrieve `pairs`.

        Returns:
            **array of shape (n_segments, 2):** Per-segment sample identifier pairs.
        """
        return np.column_stack([self.__sample_id_1, self.__sample_id_2])

    @property
    def haplotype_pairs(self) -> np.ndarray:
        """
        Retrieve `haplotype_pairs`.

        Returns:
            **array of shape (n_segments, 2):** Per-segment haplotype identifier pairs.
        """
        return np.column_stack([self.__haplotype_id_1, self.__haplotype_id_2])

    def copy(self) -> 'IBDObject':
        """
        Create and return a copy of `self`.

        Returns:
            **IBDObject:** A new instance of the current object.
        """
        return copy.deepcopy(self)

    def keys(self) -> List[str]:
        """
        Retrieve a list of public attribute names for `self`.

        Returns:
            **list of str:** A list of attribute names, with internal name-mangling removed.
        """
        return [attr.replace('_IBDObject__', '') for attr in vars(self)]

    def filter_segments(
        self,
        chrom: Optional[Sequence[str]] = None,
        samples: Optional[Sequence[str]] = None,
        min_length_cm: Optional[float] = None,
        segment_types: Optional[Sequence[str]] = None,
        inplace: bool = False,
    ) -> Optional['IBDObject']:
        """
        Filter IBD segments by chromosome, sample names, and/or minimum genetic length.

        Args:
            chrom (sequence of str, optional): Chromosome(s) to include.
            samples (sequence of str, optional): Sample names to include if present in either column.
            min_length_cm (float, optional): Minimum cM length threshold.
            inplace (bool, default=False): If True, modifies `self` in place. If False, returns a new `IBDObject`.

        Returns:
            **Optional[IBDObject]:** A filtered IBDObject if `inplace=False`. If `inplace=True`, returns None.
        """
        mask = np.ones(self.n_segments, dtype=bool)

        if chrom is not None:
            chrom = np.atleast_1d(chrom)
            mask &= np.isin(self.__chrom, chrom)

        if samples is not None:
            samples = np.atleast_1d(samples)
            mask &= np.isin(self.__sample_id_1, samples) | np.isin(self.__sample_id_2, samples)

        if min_length_cm is not None and self.__length_cm is not None:
            mask &= self.__length_cm >= float(min_length_cm)

        if segment_types is not None and self.__segment_type is not None:
            segment_types = np.atleast_1d(segment_types)
            mask &= np.isin(self.__segment_type, segment_types)

        def _apply_mask(x: Optional[np.ndarray]) -> Optional[np.ndarray]:
            return None if x is None else np.asarray(x)[mask]

        if inplace:
            self.__sample_id_1 = _apply_mask(self.__sample_id_1)
            self.__haplotype_id_1 = _apply_mask(self.__haplotype_id_1)
            self.__sample_id_2 = _apply_mask(self.__sample_id_2)
            self.__haplotype_id_2 = _apply_mask(self.__haplotype_id_2)
            self.__chrom = _apply_mask(self.__chrom)
            self.__start = _apply_mask(self.__start)
            self.__end = _apply_mask(self.__end)
            self.__length_cm = _apply_mask(self.__length_cm)
            self.__segment_type = _apply_mask(self.__segment_type)
            return None
        else:
            return IBDObject(
                sample_id_1=_apply_mask(self.__sample_id_1),
                haplotype_id_1=_apply_mask(self.__haplotype_id_1),
                sample_id_2=_apply_mask(self.__sample_id_2),
                haplotype_id_2=_apply_mask(self.__haplotype_id_2),
                chrom=_apply_mask(self.__chrom),
                start=_apply_mask(self.__start),
                end=_apply_mask(self.__end),
                length_cm=_apply_mask(self.__length_cm),
                segment_type=_apply_mask(self.__segment_type),
            )

    def restrict_to_ancestry(
        self,
        *,
        laiobj: Any,
        ancestry: Any,
        require_both_haplotypes: bool = False,
        min_bp: Optional[int] = None,
        min_cm: Optional[float] = None,
        inplace: bool = False,
        method: str = 'clip',
    ) -> Optional['IBDObject']:
        """
        Filter and/or trim IBD segments to intervals where both individuals carry the specified ancestry
        according to a `LocalAncestryObject`.

        This performs an interval intersection per segment against ancestry tracts:
        - If haplotype IDs are known (e.g., Hap-IBD), ancestry is checked on the specific
          haplotype of each individual.
        - If haplotype IDs are unknown (e.g., ancIBD; haplotype_id_* == -1), ancestry is
          considered present for an individual if at least one of their haplotypes matches
          the requested ancestry (unless `require_both_haplotypes=True`).

        Method 'strict':
            Drop entire IBD segments if ANY overlapping LAI window contains non-target ancestry
            for either individual. No trimming occurs - segments are kept whole or dropped completely.

        Method 'clip':
            Trim IBD segments to contiguous regions where both individuals have the target ancestry.
            Resulting subsegments are clipped to LAI window boundaries and original IBD start/end,
            with optional length filtering by bp or cM.

        Args:
            laiobj: LocalAncestryObject containing 2D `lai` of shape (n_windows, n_haplotypes),
                `physical_pos` (n_windows, 2), and `chromosomes` (n_windows,).
            ancestry: Target ancestry code or label. Compared as string, so both int and str work.
            require_both_haplotypes: If True, require both haplotypes of each individual to have
                the target ancestry within a window. When haplotypes are known per segment, this
                only affects cases with unknown haplotypes (== -1) or IBD2 segments.
            min_bp: Minimum base-pair length to retain a segment (strict) or subsegment (clip).
            min_cm: Minimum centiMorgan length to retain a segment (strict) or subsegment (clip).
            inplace: If True, replace `self` with the restricted object; else return a new object.
            method: Method to use for filtering. 'strict' drops entire segments that overlap with
                non-target ancestry. 'clip' trims segments to target ancestry regions.

        Returns:
            Optional[IBDObject]: A restricted IBDObject if `inplace=False`. If `inplace=True`,
                returns None.
        """
        if method not in ['strict', 'clip']:
            raise ValueError(f"Method must be 'strict' or 'clip', got '{method}'")

        # Basic LAI shape/metadata checks
        lai = getattr(laiobj, 'lai', None)
        physical_pos = getattr(laiobj, 'physical_pos', None)
        chromosomes = getattr(laiobj, 'chromosomes', None)
        centimorgan_pos = getattr(laiobj, 'centimorgan_pos', None)
        haplotypes = getattr(laiobj, 'haplotypes', None)

        if lai is None or physical_pos is None or chromosomes is None or haplotypes is None:
            raise ValueError(
                "`laiobj` must provide `lai`, `physical_pos`, `chromosomes`, and `haplotypes`."
            )

        if lai.ndim != 2:
            raise ValueError("`laiobj.lai` must be 2D with shape (n_windows, n_haplotypes).")

        # Build haplotype label -> column index map (labels like 'Sample.0', 'Sample.1')
        hap_to_col = {str(h): i for i, h in enumerate(haplotypes)}

        # Coerce ancestry to str for robust comparisons
        anc_str = str(ancestry)

        # Coerce LAI values to str once for comparisons
        lai_str = lai.astype(str)

        # Prepare arrays for the restricted segments
        out_sample_id_1: List[str] = []
        out_haplotype_id_1: List[int] = []
        out_sample_id_2: List[str] = []
        out_haplotype_id_2: List[int] = []
        out_chrom: List[str] = []
        out_start: List[int] = []
        out_end: List[int] = []
        out_length_cm: List[float] = []
        out_segment_type: List[str] = [] if self.__segment_type is not None else None  # type: ignore

        # Vectorize chrom compare by making LAI chromosome strings
        chr_lai = np.asarray(chromosomes).astype(str)

        # Helper to compute cM length for a trimmed interval using LAI windows
        def _approx_cm_len(chr_mask: np.ndarray, start_bp: int, end_bp: int) -> Optional[float]:
            if centimorgan_pos is None:
                return None
            win_st = physical_pos[chr_mask, 0]
            win_en = physical_pos[chr_mask, 1]
            win_cm_st = centimorgan_pos[chr_mask, 0]
            win_cm_en = centimorgan_pos[chr_mask, 1]
            cm_total = 0.0
            for ws, we, cs, ce in zip(win_st, win_en, win_cm_st, win_cm_en):
                # Overlap with [start_bp, end_bp]
                overlap_start = max(int(ws), int(start_bp))
                overlap_end = min(int(we), int(end_bp))
                if overlap_start > overlap_end:
                    continue
                wlen_bp = max(1, int(we) - int(ws) + 1)
                olen_bp = int(overlap_end) - int(overlap_start) + 1
                frac = float(olen_bp) / float(wlen_bp)
                cm_total += frac * float(ce - cs)
            return cm_total

        # Iterate over segments
        for i in range(self.n_segments):
            chrom = str(self.__chrom[i])
            seg_start = int(self.__start[i])
            seg_end = int(self.__end[i])
            if seg_end < seg_start:
                continue

            # Subset LAI windows on this chromosome that overlap the segment
            idx_chr = (chr_lai == chrom)
            if not np.any(idx_chr):
                continue
            lai_st = physical_pos[idx_chr, 0]
            lai_en = physical_pos[idx_chr, 1]
            overlaps = (lai_en >= seg_start) & (lai_st <= seg_end)
            if not np.any(overlaps):
                continue

            # Build per-window ancestry mask for both individuals
            s1 = str(self.__sample_id_1[i])
            s2 = str(self.__sample_id_2[i])
            h1 = int(self.__haplotype_id_1[i]) if self.__haplotype_id_1 is not None else -1
            h2 = int(self.__haplotype_id_2[i]) if self.__haplotype_id_2 is not None else -1

            # Resolve haplotype column indices for each sample
            # Known haplotypes are 1-based in inputs; convert to {0,1}
            def _get_cols(sample: str) -> Tuple[int, int]:
                a = hap_to_col.get(f"{sample}.0")
                b = hap_to_col.get(f"{sample}.1")
                if a is None or b is None:
                    raise ValueError(f"Sample '{sample}' not found in LAI haplotypes.")
                return a, b

            s1_a, s1_b = _get_cols(s1)
            s2_a, s2_b = _get_cols(s2)

            # LAI rows for this chromosome
            lai_rows = lai_str[idx_chr, :]

            # Determine ancestry presence per window for each individual
            if h1 in (1, 2) and h2 in (1, 2):
                # Use specific haplotypes
                s1_col = s1_a if (h1 - 1) == 0 else s1_b
                s2_col = s2_a if (h2 - 1) == 0 else s2_b
                s1_mask = (lai_rows[:, s1_col] == anc_str)
                s2_mask = (lai_rows[:, s2_col] == anc_str)
                if require_both_haplotypes:
                    # Additionally require the other hap of each sample to match
                    s1_other = s1_b if s1_col == s1_a else s1_a
                    s2_other = s2_b if s2_col == s2_a else s2_a
                    s1_mask = s1_mask & (lai_rows[:, s1_other] == anc_str)
                    s2_mask = s2_mask & (lai_rows[:, s2_other] == anc_str)
            else:
                # Unknown hap IDs: require at least one hap to match (or both if requested)
                if require_both_haplotypes:
                    s1_mask = (lai_rows[:, s1_a] == anc_str) & (lai_rows[:, s1_b] == anc_str)
                    s2_mask = (lai_rows[:, s2_a] == anc_str) & (lai_rows[:, s2_b] == anc_str)
                else:
                    s1_mask = (lai_rows[:, s1_a] == anc_str) | (lai_rows[:, s1_b] == anc_str)
                    s2_mask = (lai_rows[:, s2_a] == anc_str) | (lai_rows[:, s2_b] == anc_str)

            keep = overlaps & s1_mask & s2_mask

            if method == 'strict':
                # In strict mode, ALL overlapping windows must have target ancestry
                if not np.array_equal(overlaps, keep):
                    continue  # Drop entire segment

                # Apply length filters to original segment
                if min_bp is not None and (seg_end - seg_start + 1) < int(min_bp):
                    continue

                # In strict mode, preserve original length_cm
                cm_len = float(self.__length_cm[i]) if self.__length_cm is not None else None

                if min_cm is not None:
                    if cm_len is None or cm_len < float(min_cm):
                        continue

                # Keep entire original segment
                out_sample_id_1.append(s1)
                out_sample_id_2.append(s2)
                out_haplotype_id_1.append(h1)
                out_haplotype_id_2.append(h2)
                out_chrom.append(chrom)
                out_start.append(seg_start)
                out_end.append(seg_end)
                out_length_cm.append(float(cm_len) if cm_len is not None else float('nan'))
                if out_segment_type is not None:
                    out_segment_type.append(str(self.__segment_type[i]))  # type: ignore

            else:  # method == 'clip'
                if not np.any(keep):
                    continue

                # Identify contiguous windows where keep=True
                idx_keep = np.where(keep)[0]
                # Split into runs of consecutive indices
                breaks = np.where(np.diff(idx_keep) > 1)[0]
                run_starts = np.r_[0, breaks + 1]
                run_ends = np.r_[breaks, idx_keep.size - 1]

                # Create subsegments for each contiguous run
                for rs, re in zip(run_starts, run_ends):
                    i0 = idx_keep[rs]
                    i1 = idx_keep[re]
                    sub_start = int(max(seg_start, int(lai_st[i0])))
                    sub_end = int(min(seg_end, int(lai_en[i1])))
                    if sub_end < sub_start:
                        continue

                    # Length filters: bp first
                    if min_bp is not None and (sub_end - sub_start + 1) < int(min_bp):
                        continue

                    # Compute cM length if possible, else approximate or None
                    cm_len = _approx_cm_len(idx_chr, sub_start, sub_end)
                    if cm_len is None and self.__length_cm is not None:
                        # Scale the original segment length by bp fraction
                        total_bp = max(1, int(seg_end - seg_start + 1))
                        frac_bp = float(sub_end - sub_start + 1) / float(total_bp)
                        try:
                            cm_len = float(self.__length_cm[i]) * frac_bp
                        except Exception:
                            cm_len = None

                    # Apply cM filter if requested (treat None as 0)
                    if min_cm is not None:
                        if cm_len is None or cm_len < float(min_cm):
                            continue

                    # Append trimmed segment
                    out_sample_id_1.append(s1)
                    out_sample_id_2.append(s2)
                    out_haplotype_id_1.append(h1)
                    out_haplotype_id_2.append(h2)
                    out_chrom.append(chrom)
                    out_start.append(sub_start)
                    out_end.append(sub_end)
                    out_length_cm.append(float(cm_len) if cm_len is not None else float('nan'))
                    if out_segment_type is not None:
                        out_segment_type.append(str(self.__segment_type[i]))  # type: ignore

        # If nothing remains, return empty object with zero segments
        if len(out_start) == 0:
            # Build minimal arrays
            empty = IBDObject(
                sample_id_1=np.array([], dtype=object),
                haplotype_id_1=np.array([], dtype=int),
                sample_id_2=np.array([], dtype=object),
                haplotype_id_2=np.array([], dtype=int),
                chrom=np.array([], dtype=object),
                start=np.array([], dtype=int),
                end=np.array([], dtype=int),
                length_cm=None,
                segment_type=None if out_segment_type is None else np.array([], dtype=object),
            )
            if inplace:
                self.__sample_id_1 = empty.sample_id_1
                self.__haplotype_id_1 = empty.haplotype_id_1
                self.__sample_id_2 = empty.sample_id_2
                self.__haplotype_id_2 = empty.haplotype_id_2
                self.__chrom = empty.chrom
                self.__start = empty.start
                self.__end = empty.end
                self.__length_cm = empty.length_cm
                self.__segment_type = empty.segment_type
                return None
            return empty

        # Assemble outputs
        out_length_array: Optional[np.ndarray]
        if len(out_length_cm) > 0:
            # Convert NaNs to None-equivalent by using np.array with dtype float
            out_length_array = np.asarray(out_length_cm, dtype=float)
        else:
            out_length_array = None

        new_obj = IBDObject(
            sample_id_1=np.asarray(out_sample_id_1, dtype=object),
            haplotype_id_1=np.asarray(out_haplotype_id_1, dtype=int),
            sample_id_2=np.asarray(out_sample_id_2, dtype=object),
            haplotype_id_2=np.asarray(out_haplotype_id_2, dtype=int),
            chrom=np.asarray(out_chrom, dtype=object),
            start=np.asarray(out_start, dtype=int),
            end=np.asarray(out_end, dtype=int),
            length_cm=out_length_array,
            segment_type=None if out_segment_type is None else np.asarray(out_segment_type, dtype=object),
        )

        if inplace:
            self.__sample_id_1 = new_obj.sample_id_1
            self.__haplotype_id_1 = new_obj.haplotype_id_1
            self.__sample_id_2 = new_obj.sample_id_2
            self.__haplotype_id_2 = new_obj.haplotype_id_2
            self.__chrom = new_obj.chrom
            self.__start = new_obj.start
            self.__end = new_obj.end
            self.__length_cm = new_obj.length_cm
            self.__segment_type = new_obj.segment_type
            return None
        return new_obj

    def _sanity_check(self) -> None:
        """
        Perform sanity checks on the parsed data to ensure data integrity.
        """
        n = self.__chrom.shape[0]
        arrays = [
            self.__sample_id_1,
            self.__haplotype_id_1,
            self.__sample_id_2,
            self.__haplotype_id_2,
            self.__start,
            self.__end,
        ]
        if any(arr.shape[0] != n for arr in arrays):
            raise ValueError("All input arrays must have the same length.")

        if self.__length_cm is not None and self.__length_cm.shape[0] != n:
            raise ValueError("`length_cm` must have the same length as other arrays.")

        if self.__segment_type is not None and self.__segment_type.shape[0] != n:
            raise ValueError("`segment_type` must have the same length as other arrays.")

        # Validate haplotype identifiers are 1 or 2, or -1 when unknown
        valid_values = np.array([1, 2, -1])
        if not np.isin(self.__haplotype_id_1, valid_values).all() or not np.isin(self.__haplotype_id_2, valid_values).all():
            raise ValueError("Haplotype identifiers must be in {1, 2} or -1 if unknown.")

