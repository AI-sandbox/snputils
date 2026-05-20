from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Union

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

            samples = file_samples[sample_indices] if "IID" in fields_set else None
            variants_ref = []
            variants_alt = []
            variants_chrom = []
            variants_id = []
            variants_pos = []
            probabilities = []
            prob_width = None

            read_gp = "GP" in fields_set
            selected_set = set(int(idx) for idx in variant_indices)
            for idx, var in enumerate(bfile):
                if idx not in selected_set:
                    continue
                selected_set.remove(idx)

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
                            probabilities = [self._pad_probabilities(p, new_width) for p in probabilities]
                            prob_width = new_width
                        probs = self._pad_probabilities(probs, prob_width)
                    probabilities.append(probs)

                if not selected_set:
                    break

            if read_gp:
                calldata_gp = (
                    np.stack(probabilities, axis=0)
                    if probabilities
                    else np.empty((0, len(sample_indices), 0), dtype=np.float32)
                )
            else:
                calldata_gp = None

        return SNPObject(
            genotypes=None,
            calldata_gp=calldata_gp,
            samples=samples,
            variants_ref=np.asarray(variants_ref, dtype=object) if "REF" in fields_set else None,
            variants_alt=np.asarray(variants_alt, dtype=object) if "ALT" in fields_set else None,
            variants_chrom=np.asarray(variants_chrom, dtype=object) if "#CHROM" in fields_set else None,
            variants_id=np.asarray(variants_id, dtype=object) if "ID" in fields_set else None,
            variants_pos=np.asarray(variants_pos, dtype=np.int64) if "POS" in fields_set else None,
        )

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
        selected = []
        varids = bfile.varids()
        rsids = bfile.rsids()
        chroms = bfile.chroms()
        positions = bfile.positions()
        for idx, (varid, rsid, chrom, pos) in enumerate(zip(varids, rsids, chroms, positions)):
            identifiers = {str(varid), str(rsid), f"{chrom}:{pos}"}
            if requested.intersection(identifiers):
                selected.append(idx)

        found = set()
        for idx in selected:
            found.update({str(varids[idx]), str(rsids[idx]), f"{chroms[idx]}:{positions[idx]}"})
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"The following specified variants were not found: {missing}")
        return np.asarray(selected, dtype=int)
