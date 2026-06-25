from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Union

import numpy as np
from bgen import BgenReader as _BgenReader

from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.read.base import SNPBaseReader

log = logging.getLogger(__name__)


def _as_field_list(fields: Optional[Union[str, Sequence[str]]]) -> Optional[List[str]]:
    if fields is None:
        return None
    if isinstance(fields, str):
        return [fields]
    return list(fields)


def _variant_identifier(var) -> str:
    varid = str(var.varid)
    if varid and varid != ".":
        return varid
    return str(var.rsid)


def _is_strictly_increasing(values: np.ndarray) -> bool:
    return bool(values.size <= 1 or np.all(values[1:] > values[:-1]))


@SNPBaseReader.register
class BGENReader(SNPBaseReader):
    def read(
        self,
        fields: Optional[Union[str, Sequence[str]]] = None,
        exclude_fields: Optional[Union[str, Sequence[str]]] = None,
        sample_path: Optional[Union[str, bytes]] = None,
        sample_ids: Optional[Sequence[str]] = None,
        sample_idxs: Optional[Sequence[int]] = None,
        variant_ids: Optional[Sequence[str]] = None,
        variant_idxs: Optional[Sequence[int]] = None,
    ) -> SNPObject:
        """
        Read a BGEN file into a SNPObject.

        Args:
            fields: Fields to include. Available fields are ``GP``, ``IID``, ``REF``,
                ``ALT``, ``#CHROM``, ``ID``, and ``POS``. ``GT`` is intentionally
                unsupported because this reader preserves BGEN genotype probabilities
                instead of converting them to hard calls.
            exclude_fields: Fields to exclude from the returned SNPObject.
            sample_path: Optional Oxford ``.sample`` file for BGEN files without
                embedded sample identifiers.
            sample_ids: Sample IDs to read. If None and sample_idxs is None, all samples are read.
            sample_idxs: Sample indices to read. If None and sample_ids is None, all samples are read.
            variant_ids: Variant IDs to read. Matches BGEN varid, rsid, or ``chrom:pos``.
            variant_idxs: Variant indices to read. If None and variant_ids is None, all variants are read.

        Returns:
            SNPObject: A SNPObject with genotype probabilities in ``calldata_gp``.
            Mixed probability widths are padded with NaN columns.
        """
        if sample_idxs is not None and sample_ids is not None:
            raise ValueError("Only one of sample_idxs and sample_ids can be specified.")
        if variant_idxs is not None and variant_ids is not None:
            raise ValueError("Only one of variant_idxs and variant_ids can be specified.")

        fields_list = _as_field_list(fields) or ["GP", "IID", "REF", "ALT", "#CHROM", "ID", "POS"]
        exclude = set(_as_field_list(exclude_fields) or [])
        fields_set = {field for field in fields_list if field not in exclude}

        if "GT" in fields_set:
            raise NotImplementedError(
                "BGENReader preserves genotype probabilities in `calldata_gp` and does not hard-call GT."
            )

        sample_path_str = "" if sample_path is None else str(sample_path)
        log.info(f"Reading {self.filename}")

        with _BgenReader(str(self.filename), sample_path_str, delay_parsing=True) as bfile:
            file_samples = np.asarray(bfile.samples, dtype=object)
            sample_indices = self._resolve_sample_indices(file_samples, sample_ids, sample_idxs)
            variant_indices = self._resolve_variant_indices(bfile, variant_ids, variant_idxs)
            variant_indices = np.asarray(variant_indices, dtype=int)

            samples = file_samples[sample_indices] if "IID" in fields_set else None
            read_gp = "GP" in fields_set
            if not read_gp and not {"REF", "ALT"} & fields_set:
                metadata = self._bulk_variant_metadata(bfile, variant_indices, fields_set)
                calldata_gp = None
            else:
                variants = self._load_selected_variants(bfile, variant_indices)
                metadata, calldata_gp = self._variant_data_from_records(
                    variants=variants,
                    sample_indices=sample_indices,
                    fields_set=fields_set,
                    read_gp=read_gp,
                )

        return SNPObject(
            genotypes=None,
            calldata_gp=calldata_gp,
            samples=samples,
            variants_ref=metadata["variants_ref"],
            variants_alt=metadata["variants_alt"],
            variants_chrom=metadata["variants_chrom"],
            variants_id=metadata["variants_id"],
            variants_pos=metadata["variants_pos"],
        )

    @staticmethod
    def _variant_metadata_template() -> dict[str, Optional[np.ndarray]]:
        return {
            "variants_ref": None,
            "variants_alt": None,
            "variants_chrom": None,
            "variants_id": None,
            "variants_pos": None,
        }

    @classmethod
    def _bulk_variant_metadata(
        cls,
        bfile,
        variant_indices: np.ndarray,
        fields_set: set[str],
    ) -> dict[str, Optional[np.ndarray]]:
        metadata = cls._variant_metadata_template()
        if "ID" in fields_set:
            identifiers = cls._variant_identifiers_array(bfile)
            metadata["variants_id"] = identifiers[variant_indices]
        if "#CHROM" in fields_set:
            metadata["variants_chrom"] = np.asarray(bfile.chroms(), dtype=object)[variant_indices]
        if "POS" in fields_set:
            metadata["variants_pos"] = np.asarray(bfile.positions(), dtype=np.int64)[variant_indices]
        return metadata

    @classmethod
    def _variant_data_from_records(
        cls,
        variants: Sequence[Any],
        sample_indices: np.ndarray,
        fields_set: set[str],
        read_gp: bool,
    ) -> tuple[dict[str, Optional[np.ndarray]], Optional[np.ndarray]]:
        metadata = cls._variant_metadata_template()
        variants_ref: list[str] = []
        variants_alt: list[str] = []
        variants_chrom: list[str] = []
        variants_id: list[str] = []
        variants_pos: list[int] = []
        probabilities: list[np.ndarray] = []
        prob_width = None

        for var in variants:
            alleles = [str(allele) for allele in var.alleles]
            if "REF" in fields_set:
                variants_ref.append(alleles[0] if alleles else "")
            if "ALT" in fields_set:
                variants_alt.append(",".join(alleles[1:]) if len(alleles) > 1 else "")
            if "#CHROM" in fields_set:
                variants_chrom.append(str(var.chrom))
            if "ID" in fields_set:
                variants_id.append(_variant_identifier(var))
            if "POS" in fields_set:
                variants_pos.append(int(var.pos))

            if read_gp:
                probs = np.asarray(var.probabilities, dtype=np.float32)[sample_indices, :]
                if prob_width is None:
                    prob_width = probs.shape[1]
                elif probs.shape[1] != prob_width:
                    new_width = max(prob_width, probs.shape[1])
                    if new_width != prob_width:
                        probabilities = [cls._pad_probabilities(p, new_width) for p in probabilities]
                        prob_width = new_width
                    probs = cls._pad_probabilities(probs, prob_width)
                probabilities.append(probs)

        if "REF" in fields_set:
            metadata["variants_ref"] = np.asarray(variants_ref, dtype=object)
        if "ALT" in fields_set:
            metadata["variants_alt"] = np.asarray(variants_alt, dtype=object)
        if "#CHROM" in fields_set:
            metadata["variants_chrom"] = np.asarray(variants_chrom, dtype=object)
        if "ID" in fields_set:
            metadata["variants_id"] = np.asarray(variants_id, dtype=object)
        if "POS" in fields_set:
            metadata["variants_pos"] = np.asarray(variants_pos, dtype=np.int64)

        calldata_gp = None
        if read_gp:
            calldata_gp = (
                np.stack(probabilities, axis=0)
                if probabilities
                else np.empty((0, len(sample_indices), 0), dtype=np.float32)
            )
        return metadata, calldata_gp

    @staticmethod
    def _load_selected_variants(bfile, variant_indices: np.ndarray) -> list[Any]:
        if variant_indices.size == 0:
            return []

        if not _is_strictly_increasing(variant_indices):
            return [bfile[int(idx)] for idx in variant_indices]

        if getattr(bfile, "index", None) is not None:
            return [bfile[int(idx)] for idx in variant_indices]

        total_variants = len(bfile)
        scan_limit = int(variant_indices[-1])
        selection_count = int(variant_indices.size)
        should_scan = scan_limit <= total_variants // 2 or selection_count >= max(1024, scan_limit // 8)
        if not should_scan:
            return [bfile[int(idx)] for idx in variant_indices]

        selected: list[Any] = []
        target_pos = 0
        next_idx = int(variant_indices[target_pos])
        for idx, var in enumerate(bfile):
            if idx != next_idx:
                if idx > scan_limit:
                    break
                continue
            selected.append(var)
            target_pos += 1
            if target_pos == selection_count:
                break
            next_idx = int(variant_indices[target_pos])
        return selected

    @staticmethod
    def _variant_identifiers_array(bfile) -> np.ndarray:
        rsids = np.asarray(bfile.rsids(), dtype=object)
        try:
            varids = np.asarray(bfile.varids(), dtype=object)
        except ValueError:
            varids = rsids
        identifiers = rsids.copy()
        valid = (varids != "") & (varids != ".")
        identifiers[valid] = varids[valid]
        return identifiers

    @staticmethod
    def _resolve_sample_indices(
        file_samples: np.ndarray,
        sample_ids: Optional[Sequence[str]],
        sample_idxs: Optional[Sequence[int]],
    ) -> np.ndarray:
        if sample_idxs is not None:
            idx = np.asarray(sample_idxs, dtype=int).ravel()
            n_samples = len(file_samples)
            if np.any((idx < -n_samples) | (idx >= n_samples)):
                raise ValueError("One or more sample indexes are out of bounds.")
            return np.mod(idx, n_samples)

        if sample_ids is None:
            return np.arange(len(file_samples), dtype=int)

        requested = np.asarray(sample_ids, dtype=object).ravel()
        sample_lookup = {str(sample): i for i, sample in enumerate(file_samples)}
        missing = [str(sample) for sample in requested if str(sample) not in sample_lookup]
        if missing:
            raise ValueError(f"The following specified samples were not found: {missing}")
        return np.asarray([sample_lookup[str(sample)] for sample in requested], dtype=int)

    @staticmethod
    def _pad_probabilities(probabilities: np.ndarray, width: int) -> np.ndarray:
        if probabilities.shape[1] == width:
            return probabilities
        padded = np.full((probabilities.shape[0], width), np.nan, dtype=probabilities.dtype)
        padded[:, : probabilities.shape[1]] = probabilities
        return padded

    @staticmethod
    def _resolve_variant_indices(
        bfile,
        variant_ids: Optional[Sequence[str]],
        variant_idxs: Optional[Sequence[int]],
    ) -> np.ndarray:
        if variant_idxs is not None:
            idx = np.asarray(variant_idxs, dtype=int).ravel()
            n_variants = len(bfile)
            if np.any((idx < -n_variants) | (idx >= n_variants)):
                raise ValueError("One or more variant indexes are out of bounds.")
            return np.mod(idx, n_variants)

        if variant_ids is None:
            return np.arange(len(bfile), dtype=int)

        requested = {str(value) for value in np.asarray(variant_ids, dtype=object).ravel()}
        selected: list[int] = []
        varids = BGENReader._variant_identifiers_array(bfile)
        rsids = np.asarray(bfile.rsids(), dtype=object)
        chroms = np.asarray(bfile.chroms(), dtype=object)
        positions = np.asarray(bfile.positions(), dtype=np.int64)
        for idx, (varid, rsid, chrom, pos) in enumerate(zip(varids, rsids, chroms, positions)):
            aliases = {str(varid), str(rsid), f"{chrom}:{pos}"}
            if requested.intersection(aliases):
                selected.append(idx)

        found = set()
        for idx in selected:
            found.update({str(varids[idx]), str(rsids[idx]), f"{chroms[idx]}:{positions[idx]}"})
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"The following specified variants were not found: {missing}")
        return np.asarray(selected, dtype=int)
