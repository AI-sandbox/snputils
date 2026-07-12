from __future__ import annotations
from collections.abc import Mapping
import logging
from pathlib import Path
import numpy as np
import copy
import warnings
import re
from typing import Any, Union, Tuple, List, Sequence, Dict, Optional, TYPE_CHECKING
from scipy.stats import chi2, fisher_exact, mode

from snputils._utils.allele_freq import aggregate_pop_allele_freq
from snputils._utils.ancestry import known_lai_values
from snputils._utils.genotypes import sum_diploid_genotypes, validate_biallelic_hard_calls
from snputils._utils.printing import array_shape, format_repr

if TYPE_CHECKING:
    from snputils.ancestry.genobj.local import LocalAncestryObject


log = logging.getLogger(__name__)


def _paired_common_indices(
    query_identifiers: Sequence[Any],
    reference_identifiers: Sequence[Any],
) -> Tuple[List[Any], np.ndarray, np.ndarray]:
    query_positions: Dict[Any, List[int]] = {}
    reference_positions: Dict[Any, List[int]] = {}
    for index, identifier in enumerate(query_identifiers):
        query_positions.setdefault(identifier, []).append(index)
    for index, identifier in enumerate(reference_identifiers):
        reference_positions.setdefault(identifier, []).append(index)

    common_ids = [identifier for identifier in query_positions if identifier in reference_positions]
    duplicates = [
        identifier
        for identifier in common_ids
        if len(query_positions[identifier]) != 1 or len(reference_positions[identifier]) != 1
    ]
    if duplicates:
        preview = ", ".join(map(str, duplicates[:5]))
        raise ValueError(
            "Cannot match variants with duplicated shared identifiers: " + preview
        )

    query_idx = np.asarray([query_positions[identifier][0] for identifier in common_ids], dtype=int)
    reference_idx = np.asarray(
        [reference_positions[identifier][0] for identifier in common_ids], dtype=int
    )
    return common_ids, query_idx, reference_idx


class SNPObject:
    """
    A class for Single Nucleotide Polymorphism (SNP) data, with optional support for
    SNP-level Local Ancestry Information (LAI).
    """
    _DEFAULT_IMPUTATION_INFO_KEYS = (
        "INFO",
        "R2",
        "DR2",
        "IMP",
        "INFO_SCORE",
        "IMPUTE2_INFO",
        "RSQ",
        "ER2",
    )

    def __init__(
        self,
        genotypes: Optional[np.ndarray] = None,
        samples: Optional[np.ndarray] = None,
        variants_ref: Optional[np.ndarray] = None,
        variants_alt: Optional[np.ndarray] = None,
        variants_chrom: Optional[np.ndarray] = None,
        variants_cm: Optional[np.ndarray] = None,
        variants_filter_pass: Optional[np.ndarray] = None,
        variants_id: Optional[np.ndarray] = None,
        variants_pos: Optional[np.ndarray] = None,
        variants_qual: Optional[np.ndarray] = None,
        variants_info: Optional[np.ndarray] = None,
        calldata_lai: Optional[np.ndarray] = None,
        calldata_gp: Optional[np.ndarray] = None,
        ancestry_map: Optional[Dict[str, str]] = None,
        sample_fid: Optional[np.ndarray] = None,
        sample_sex: Optional[np.ndarray] = None,
        calldata_gt: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            genotypes (array, optional):
                An array containing genotype data for each sample. This array can be either 2D with shape
                `(n_snps, n_samples)` for per-sample dosages, or 3D with shape
                `(n_snps, n_samples, 2)` for phased diploid allele calls.
            samples (array of shape (n_samples,), optional):
                An array containing unique sample identifiers.
            sample_fid (array of shape (n_samples,), optional):
                PLINK-style family ID per sample (same order as ``samples``), for example population labels
                in column 1 of a ``.fam`` file. When present and not identical to ``samples``, f-statistics
                use these values as default group labels if ``sample_labels`` is omitted.
            sample_sex (array of shape (n_samples,), optional):
                PLINK-style sex code per sample, aligned with ``samples``.
            variants_ref (array of shape (n_snps,), optional):
                An array containing the reference allele for each SNP.
            variants_alt (array of shape (n_snps,), optional):
                An array containing the alternate allele for each SNP.
            variants_chrom (array of shape (n_snps,), optional):
                An array containing the chromosome for each SNP.
            variants_cm (array of shape (n_snps,), optional):
                An array containing the genetic-map position in centimorgans for each SNP.
            variants_filter_pass (array of shape (n_snps,), optional):
                An array indicating whether each SNP passed control checks.
            variants_id (array of shape (n_snps,), optional):
                An array containing unique identifiers (IDs) for each SNP.
            variants_pos (array of shape (n_snps,), optional):
                An array containing the chromosomal positions for each SNP.
            variants_qual (array of shape (n_snps,), optional):
                An array containing the Phred-scaled quality score for each SNP.
            variants_info (array of shape (n_snps,), optional):
                An array containing VCF/PVAR INFO column values for each SNP.
            calldata_lai (array, optional):
                An array containing the ancestry for each SNP. This array can be either 2D with shape
                `(n_snps, n_samples*2)`, or 3D with shape (n_snps, n_samples, 2).
            calldata_gp (array, optional):
                Genotype probabilities for each SNP and sample, with shape
                `(n_snps, n_samples, n_probabilities)`. For diploid biallelic BGEN data,
                unphased probabilities typically have three columns and phased probabilities
                typically have four columns. Mixed-width BGEN reads are padded with NaN
                columns to keep this as a single array.
            ancestry_map (dict of str to str, optional):
                A dictionary mapping ancestry codes to region names.
            calldata_gt (array, optional):
                Backward-compatible alias for `genotypes`.
        """
        if genotypes is not None and calldata_gt is not None:
            raise ValueError("Specify only one of `genotypes` or `calldata_gt`, not both.")
        if genotypes is None and calldata_gt is not None:
            genotypes = calldata_gt

        self.__genotypes = genotypes
        self.__samples = samples
        self.__variants_ref = variants_ref
        self.__variants_alt = variants_alt
        self.__variants_chrom = variants_chrom
        self.__variants_cm = variants_cm
        self.__variants_filter_pass = variants_filter_pass
        self.__variants_id = variants_id
        self.__variants_pos = variants_pos
        self.__variants_qual = variants_qual
        self.__variants_info = variants_info
        self.__calldata_lai = calldata_lai
        self.__calldata_gp = calldata_gp
        self.__ancestry_map = ancestry_map
        self.__sample_fid = None if sample_fid is None else np.asarray(sample_fid)
        self.__sample_sex = None if sample_sex is None else np.asarray(sample_sex)

        self._sanity_check()

    def __getitem__(self, key: str) -> Any:
        """
        To access an attribute of the class using the square bracket notation,
        similar to a dictionary.
        """
        try:
            return getattr(self, key)
        except:
            raise KeyError(f'Invalid key: {key}.')

    def __setitem__(self, key: str, value: Any):
        """
        To set an attribute of the class using the square bracket notation,
        similar to a dictionary.
        """
        try:
            setattr(self, key, value)
        except:
            raise KeyError(f'Invalid key: {key}.')

    def __repr__(self) -> str:
        return format_repr(
            self,
            shape=self.shape,
            n_snps=self._n_snps_or_none(),
            n_samples=self._n_samples_or_none(),
            genotypes_shape=array_shape(self.__genotypes),
            calldata_lai_shape=array_shape(self.__calldata_lai),
            calldata_gp_shape=array_shape(self.__calldata_gp),
            has_variant_metadata=any(
                attr is not None
                for attr in (
                    self.__variants_ref,
                    self.__variants_alt,
                    self.__variants_chrom,
                    self.__variants_cm,
                    self.__variants_filter_pass,
                    self.__variants_id,
                    self.__variants_pos,
                    self.__variants_qual,
                    self.__variants_info,
                )
            ),
            has_ancestry_map=self.__ancestry_map is not None,
        )

    def __str__(self) -> str:
        return self.__repr__()

    def _n_samples_or_none(self) -> Optional[int]:
        try:
            return self.n_samples
        except ValueError:
            return None

    def _n_snps_or_none(self) -> Optional[int]:
        try:
            return self.n_snps
        except ValueError:
            return None

    @property
    def genotypes(self) -> np.ndarray:
        """
        Retrieve `genotypes`.

        Returns:
            array:
                An array containing genotype data for each sample. This array can be either 2D with shape
                ``(n_snps, n_samples)`` for per-sample dosages, or 3D with shape
                ``(n_snps, n_samples, 2)`` for phased diploid allele calls.
        """
        return self.__genotypes

    @genotypes.setter
    def genotypes(self, x: np.ndarray):
        """
        Update `genotypes`.
        """
        self.__genotypes = x

    @property
    def calldata_gt(self) -> Optional[np.ndarray]:
        """
        Retrieve `genotypes` via legacy alias.
        """
        return self.__genotypes

    @calldata_gt.setter
    def calldata_gt(self, x: np.ndarray):
        """
        Update `genotypes` via legacy alias.
        """
        self.__genotypes = x

    @property
    def samples(self) -> Optional[np.ndarray]:
        """
        Retrieve `samples`.

        Returns:
            array of shape (n_samples,):
                An array containing unique sample identifiers.
        """
        return self.__samples

    @samples.setter
    def samples(self, x: Union[List, np.ndarray]):
        """
        Update `samples`.
        """
        self.__samples = np.asarray(x)
        if (
            self.__sample_fid is not None
            and self.__samples is not None
            and len(self.__sample_fid) != len(self.__samples)
        ):
            raise ValueError(
                f"Length mismatch after updating samples: sample_fid has {len(self.__sample_fid)} "
                f"entries but samples has {len(self.__samples)}."
            )
        if (
            self.__sample_sex is not None
            and self.__samples is not None
            and len(self.__sample_sex) != len(self.__samples)
        ):
            raise ValueError(
                f"Length mismatch after updating samples: sample_sex has {len(self.__sample_sex)} "
                f"entries but samples has {len(self.__samples)}."
            )

    @property
    def sample_fid(self) -> Optional[np.ndarray]:
        """
        PLINK Family ID (FID) per sample, aligned with ``samples``.
        """
        return self.__sample_fid

    @sample_fid.setter
    def sample_fid(self, x: Optional[Union[List, np.ndarray]]):
        self.__sample_fid = None if x is None else np.asarray(x)
        if self.__sample_fid is not None and self.__samples is not None:
            if len(self.__sample_fid) != len(self.__samples):
                raise ValueError(
                    f"Length mismatch: sample_fid has {len(self.__sample_fid)} entries "
                    f"but samples has {len(self.__samples)}."
                )

    @property
    def sample_sex(self) -> Optional[np.ndarray]:
        """
        PLINK sex code per sample, aligned with ``samples``.
        """
        return self.__sample_sex

    @sample_sex.setter
    def sample_sex(self, x: Optional[Union[List, np.ndarray]]):
        self.__sample_sex = None if x is None else np.asarray(x)
        if self.__sample_sex is not None and self.__samples is not None:
            if len(self.__sample_sex) != len(self.__samples):
                raise ValueError(
                    f"Length mismatch: sample_sex has {len(self.__sample_sex)} entries "
                    f"but samples has {len(self.__samples)}."
                )

    @property
    def variants_ref(self) -> Optional[np.ndarray]:
        """
        Retrieve `variants_ref`.

        Returns:
            array of shape (n_snps,): An array containing the reference allele for each SNP.
        """
        return self.__variants_ref

    @variants_ref.setter
    def variants_ref(self, x: np.ndarray):
        """
        Update `variants_ref`.
        """
        self.__variants_ref = x

    @property
    def variants_alt(self) -> Optional[np.ndarray]:
        """
        Retrieve `variants_alt`.

        Returns:
            array of shape (n_snps,): An array containing the alternate allele for each SNP.
        """
        return self.__variants_alt

    @variants_alt.setter
    def variants_alt(self, x: np.ndarray):
        """
        Update `variants_alt`.
        """
        self.__variants_alt = x

    @property
    def variants_chrom(self) -> Optional[np.ndarray]:
        """
        Retrieve `variants_chrom`.

        Returns:
            array of shape (n_snps,): An array containing the chromosome for each SNP.
        """
        return self.__variants_chrom

    @variants_chrom.setter
    def variants_chrom(self, x: np.ndarray):
        """
        Update `variants_chrom`.
        """
        self.__variants_chrom = x

    @property
    def variants_cm(self) -> Optional[np.ndarray]:
        """
        Retrieve `variants_cm`.

        Returns:
            array of shape (n_snps,): An array containing the genetic-map position in centimorgans for each SNP.
        """
        return self.__variants_cm

    @variants_cm.setter
    def variants_cm(self, x: np.ndarray):
        """
        Update `variants_cm`.
        """
        self.__variants_cm = x

    @property
    def variants_filter_pass(self) -> Optional[np.ndarray]:
        """
        Retrieve `variants_filter_pass`.

        Returns:
            array of shape (n_snps,): An array indicating whether each SNP passed control checks.
        """
        return self.__variants_filter_pass

    @variants_filter_pass.setter
    def variants_filter_pass(self, x: np.ndarray):
        """
        Update `variants_filter_pass`.
        """
        self.__variants_filter_pass = x

    @property
    def variants_id(self) -> Optional[np.ndarray]:
        """
        Retrieve `variants_id`.

        Returns:
            array of shape (n_snps,): An array containing unique identifiers (IDs) for each SNP.
        """
        return self.__variants_id

    @variants_id.setter
    def variants_id(self, x: np.ndarray):
        """
        Update `variants_id`.
        """
        self.__variants_id = x

    @property
    def variants_pos(self) -> Optional[np.ndarray]:
        """
        Retrieve `variants_pos`.

        Returns:
            array of shape (n_snps,): An array containing the chromosomal positions for each SNP.
        """
        return self.__variants_pos

    @variants_pos.setter
    def variants_pos(self, x: np.ndarray):
        """
        Update `variants_pos`.
        """
        self.__variants_pos = x

    @property
    def variants_qual(self) -> Optional[np.ndarray]:
        """
        Retrieve `variants_qual`.

        Returns:
            array of shape (n_snps,): An array containing the Phred-scaled quality score for each SNP.
        """
        return self.__variants_qual

    @variants_qual.setter
    def variants_qual(self, x: np.ndarray):
        """
        Update `variants_qual`.
        """
        self.__variants_qual = x

    @property
    def variants_info(self) -> Optional[np.ndarray]:
        """
        Retrieve `variants_info`.

        Returns:
            array of shape (n_snps,): An array containing VCF/PVAR INFO column values for each SNP.
        """
        return self.__variants_info

    @variants_info.setter
    def variants_info(self, x: np.ndarray):
        """
        Update `variants_info`.
        """
        self.__variants_info = x

    @property
    def calldata_lai(self) -> Optional[np.ndarray]:
        """
        Retrieve `calldata_lai`.

        Returns:
            array:
                An array containing the ancestry for each SNP. This array can be either 2D with shape
                `(n_snps, n_samples*2)`, or 3D with shape (n_snps, n_samples, 2).
        """
        return self.__calldata_lai

    @calldata_lai.setter
    def calldata_lai(self, x: np.ndarray):
        """
        Update `calldata_lai`.
        """
        self.__calldata_lai = x

    @property
    def calldata_gp(self) -> Optional[np.ndarray]:
        """
        Retrieve `calldata_gp`.

        Returns:
            array:
                Genotype probabilities with shape `(n_snps, n_samples, n_probabilities)`.
        """
        return self.__calldata_gp

    @calldata_gp.setter
    def calldata_gp(self, x: np.ndarray):
        """
        Update `calldata_gp`.
        """
        self.__calldata_gp = x

    @property
    def ancestry_map(self) -> Optional[Dict[str, str]]:
        """
        Retrieve `ancestry_map`.

        Returns:
            dict of str to str: A dictionary mapping ancestry codes to region names.
        """
        return self.__ancestry_map

    @ancestry_map.setter
    def ancestry_map(self, x):
        """
        Update `ancestry_map`.
        """
        self.__ancestry_map = x

    @property
    def n_samples(self) -> int:
        """
        Retrieve `n_samples`.

        Returns:
            int: The total number of samples.
        """
        if self.__samples is not None:
            return len(self.__samples)
        elif self.__genotypes is not None:
            return self.__genotypes.shape[1]
        elif self.__calldata_gp is not None:
            return self.__calldata_gp.shape[1]
        elif self.__calldata_lai is not None:
            if self.__calldata_lai.ndim == 2:
                return self.__calldata_lai.shape[1] // 2
            elif self.__calldata_lai.ndim == 3:
                return self.__calldata_lai.shape[1]
        else:
            raise ValueError("Unable to determine the total number of samples: no relevant data is available.")

    @property
    def n_snps(self) -> int:
        """
        Retrieve `n_snps`.

        Returns:
            int: The total number of SNPs.
        """
        # List of attributes that can indicate the number of SNPs
        potential_attributes = [
            self.__genotypes,
            self.__variants_ref,
            self.__variants_alt,
            self.__variants_chrom,
            self.__variants_cm,
            self.__variants_filter_pass,
            self.__variants_id,
            self.__variants_pos,
            self.__variants_qual,
            self.__variants_info,
            self.__calldata_lai,
            self.__calldata_gp
        ]

        # Check each attribute for its first dimension, which corresponds to `n_snps`
        for attr in potential_attributes:
            if attr is not None:
                return attr.shape[0]

        raise ValueError("Unable to determine the total number of SNPs: no relevant data is available.")

    @property
    def n_chrom(self) -> Optional[int]:
        """
        Retrieve `n_chrom`.

        Returns:
            int: The total number of unique chromosomes in `variants_chrom`.
        """
        if self.variants_chrom is None:
            warnings.warn("Chromosome data `variants_chrom` is None.")
            return None

        return len(self.unique_chrom)

    @property
    def n_ancestries(self) -> int:
        """
        Retrieve `n_ancestries`.

        Returns:
            int: The total number of unique ancestries.
        """
        if self.__calldata_lai is not None:
            return len(np.unique(known_lai_values(self.__calldata_lai)))
        else:
            raise ValueError("Unable to determine the total number of ancestries: no relevant data is available.")

    @property
    def shape(self) -> Tuple[Optional[int], ...]:
        """
        Retrieve the primary data shape.

        Returns:
            tuple: The shape of `genotypes` when present, otherwise the shape
            of `calldata_gp` or `calldata_lai`. If only metadata is available, returns
            `(n_snps, n_samples)` with unknown dimensions represented as None.
        """
        gt_shape = array_shape(self.__genotypes)
        if gt_shape is not None:
            return gt_shape

        gp_shape = array_shape(self.__calldata_gp)
        if gp_shape is not None:
            return gp_shape

        lai_shape = array_shape(self.__calldata_lai)
        if lai_shape is not None:
            return lai_shape

        return (self._n_snps_or_none(), self._n_samples_or_none())

    @property
    def unique_chrom(self) -> Optional[np.ndarray]:
        """
        Retrieve `unique_chrom`.

        Returns:
            array: The unique chromosome names in `variants_chrom`, preserving their order of appearance.
        """
        if self.variants_chrom is None:
            warnings.warn("Chromosome data `variants_chrom` is None.")
            return None

        # Identify unique chromosome names and their first indexes of occurrence
        _, idx = np.unique(self.variants_chrom, return_index=True)
        # Return chromosome names sorted by their first occurrence to maintain original order
        return self.variants_chrom[np.sort(idx)]

    @property
    def is_dosage(self) -> Optional[bool]:
        """
        Report whether ``genotypes`` stores per-sample dosages.

        Returns:
            bool:
                True when ``genotypes`` has shape ``(n_snps, n_samples)`` and
                False when it contains phased calls with shape
                ``(n_snps, n_samples, 2)``. None when genotypes are unavailable.
        """
        if self.genotypes is None:
            warnings.warn("Genotype data `genotypes` is None.")
            return None

        return self.genotypes.ndim == 2

    def copy(self) -> SNPObject:
        """
        Create and return a copy of `self`.

        Returns:
            SNPObject:
                A new instance of the current object.
        """
        return copy.deepcopy(self)

    def keys(self) -> List[str]:
        """
        Retrieve a list of public attribute names for `self`.

        Returns:
            list of str:
                A list of attribute names, with internal name-mangling removed,
                for easier reference to public attributes in the instance.
        """
        return [attr.replace('_SNPObject__', '') for attr in vars(self)]

    def dosage(self, allele: Union[str, int] = "ALT") -> np.ndarray:
        """
        Return expected allele dosages as a 2D ``(n_snps, n_samples)`` array.

        If ``calldata_gp`` is present, this converts BGEN-style genotype
        probabilities to expected alternate-allele dosage. For now this supports
        biallelic variants, which are the common GWAS case. If genotype calls are
        present instead, 3D biallelic calls are summed across the allele axis and
        2D calls are returned as floating-point dosages. Multiallelic hard calls are
        rejected because a single dosage value cannot identify the counted ALT allele.

        Args:
            allele: Allele to dosage. Currently only ``"ALT"`` or ``1`` is
                supported for BGEN probability data.

        Returns:
            A float32 dosage matrix.
        """
        if self.calldata_gp is not None:
            return self._bgen_alt_dosage(allele=allele)

        if self.genotypes is None:
            raise ValueError("SNPObject requires either `calldata_gp` or `genotypes` to compute dosage.")

        gt = np.asarray(self.genotypes)
        validate_biallelic_hard_calls(
            np.empty(0, dtype=np.int8),
            alternate_alleles=self.variants_alt,
        )
        if gt.ndim == 2:
            return gt.astype(np.float32, copy=True)
        if gt.ndim == 3:
            validate_biallelic_hard_calls(gt)
            return sum_diploid_genotypes(gt, dtype=np.float32, missing_value=-1.0)
        raise ValueError("`genotypes` must be a 2D dosage array or 3D allele-call array.")

    def to_dosage(self, inplace: bool = False) -> Optional['SNPObject']:
        """
        Store expected dosages in ``genotypes``.

        This is useful for dosage-based analyses, such as GWAS, after reading a
        BGEN file where probabilities are stored in ``calldata_gp``. The original
        probability array is preserved.

        Args:
            inplace: If True, modifies ``self`` and returns None. If False,
                returns a copy with ``genotypes`` set to dosages.
        """
        dosage = self.dosage()
        if inplace:
            self.genotypes = dosage
            return None

        snpobj = self.copy()
        snpobj.genotypes = dosage
        return snpobj

    def _bgen_alt_dosage(self, allele: Union[str, int] = "ALT") -> np.ndarray:
        if isinstance(allele, str):
            if allele.upper() != "ALT":
                raise NotImplementedError("Only ALT allele dosage is currently supported for BGEN probabilities.")
        elif int(allele) != 1:
            raise NotImplementedError("Only allele index 1 (ALT) dosage is currently supported for BGEN probabilities.")

        if self.calldata_gp is None:
            raise ValueError("Genotype probability data `calldata_gp` is None.")

        if self.variants_alt is not None:
            alt = np.asarray(self.variants_alt, dtype=object).astype(str)
            multiallelic = np.char.find(alt, ",") >= 0
            if np.any(multiallelic):
                first = int(np.flatnonzero(multiallelic)[0])
                raise ValueError(
                    "BGEN dosage conversion currently supports only biallelic variants; "
                    f"variant {first} has ALT={alt[first]!r}."
                )

        gp = np.asarray(self.calldata_gp, dtype=np.float32)
        if gp.ndim != 3:
            raise ValueError("`calldata_gp` must have shape (n_snps, n_samples, n_probabilities).")

        n_snps, n_samples, n_probabilities = gp.shape
        dosage = np.empty((n_snps, n_samples), dtype=np.float32)
        weights = np.arange(n_probabilities, dtype=np.float32)
        probability_target = 64_000_000
        chunk_size = max(1, min(n_snps, probability_target // max(1, n_samples * n_probabilities)))

        for start in range(0, n_snps, chunk_size):
            stop = min(start + chunk_size, n_snps)
            block = gp[start:stop]
            if not np.isnan(block).any():
                if n_probabilities == 3:
                    dosage[start:stop] = block[:, :, 1] + (2.0 * block[:, :, 2])
                    continue
                if n_probabilities == 4:
                    pair0 = block[:, :, 0] + block[:, :, 1]
                    pair1 = block[:, :, 2] + block[:, :, 3]
                    phased = np.all(
                        np.isclose(pair0, 1.0, atol=1e-4, rtol=0)
                        & np.isclose(pair1, 1.0, atol=1e-4, rtol=0),
                        axis=1,
                    )
                    if np.all(phased):
                        dosage[start:stop] = block[:, :, 1] + block[:, :, 3]
                    else:
                        block_dosage = (
                            block[:, :, 1]
                            + (2.0 * block[:, :, 2])
                            + (3.0 * block[:, :, 3])
                        )
                        if np.any(phased):
                            block_dosage[phased] = block[phased, :, 1] + block[phased, :, 3]
                        dosage[start:stop] = block_dosage
                    continue

            finite = np.isfinite(block)
            widths = finite.sum(axis=2)
            expected_finite = np.arange(n_probabilities) < widths[:, :, None]
            if not np.array_equal(finite, expected_finite):
                raise ValueError(
                    "BGEN probability rows may only contain NaN values as all-missing rows "
                    "or as trailing padding for lower-ploidy samples."
                )

            block_dosage = np.nansum(block * weights, axis=2, dtype=np.float32)
            if n_probabilities % 2 == 0:
                phased = np.zeros(block.shape[0], dtype=bool)
                observed = widths > 0
                even_widths = np.all((widths % 2 == 0) | ~observed, axis=1)
                even_indices = np.flatnonzero(even_widths)
                if even_indices.size:
                    pairs = block[even_indices].reshape(
                        even_indices.size,
                        n_samples,
                        n_probabilities // 2,
                        2,
                    )
                    pair_present = np.isfinite(pairs).any(axis=3)
                    pair_sums = np.nansum(pairs, axis=3, dtype=np.float32)
                    phased[even_indices] = np.all(
                        np.isclose(pair_sums, 1.0, atol=1e-4, rtol=0) | ~pair_present,
                        axis=(1, 2),
                    )
                if np.any(phased):
                    block_dosage[phased] = np.nansum(block[phased, :, 1::2], axis=2, dtype=np.float32)
            block_dosage[widths == 0] = np.nan
            dosage[start:stop] = block_dosage
        return dosage

    @staticmethod
    def _bgen_biallelic_rows_look_phased(probabilities: np.ndarray, widths: np.ndarray) -> bool:
        observed_widths = [int(width) for width in widths if int(width) > 0]
        if not observed_widths or any(width % 2 != 0 for width in observed_widths):
            return False

        for sample_probs, width in zip(probabilities, widths):
            width = int(width)
            if width == 0:
                continue
            haplotypes = sample_probs[:width].reshape(width // 2, 2)
            if not np.allclose(haplotypes.sum(axis=1), 1.0, atol=1e-4, rtol=0):
                return False
        return True

    def allele_freq(
        self,
        sample_labels: Optional[Sequence[Any]] = None,
        ancestry: Optional[Union[str, int]] = None,
        laiobj: Optional["LocalAncestryObject"] = None,
        pseudohaploid: Union[bool, int] = False,
        return_counts: bool = False,
        as_dataframe: bool = False,
    ) -> Any:
        """
        Compute per-SNP alternate allele frequencies from `genotypes`.

        Args:
            sample_labels (sequence, optional):
                Population label per sample. If None, computes cohort-level frequencies.
            ancestry (str or int, optional):
                If provided, compute ancestry-masked frequencies using SNP-level LAI.
            laiobj (LocalAncestryObject, optional):
                Optional LAI object used when `self.calldata_lai` is not set.
            pseudohaploid (bool or int, default=False):
                If True, detects pseudo-haploid samples (samples with no heterozygotes in the first 1000 SNPs)
                and treats them as haploid. If an integer `n` is provided, checks the first `n` SNPs.
                If False, treats all samples as diploid.
            return_counts (bool, default=False):
                If True, also return called-allele counts with the same shape as frequencies.
            as_dataframe (bool, default=False):
                If True, return pandas DataFrame output.

        Returns:
            Frequencies as a NumPy array (or DataFrame if `as_dataframe=True`).
            If `return_counts=True`, returns `(freq, counts)`.
        """
        if self.genotypes is None:
            raise ValueError("Genotype data `genotypes` is None.")

        gt = np.asarray(self.genotypes)
        if gt.ndim not in (2, 3):
            raise ValueError("'genotypes' must be 2D or 3D array")

        n_samples = gt.shape[1]

        grouped_output = sample_labels is not None
        if sample_labels is None:
            labels = np.repeat("__all__", n_samples)
        else:
            labels = np.asarray(sample_labels)
            if labels.ndim != 1:
                labels = labels.ravel()
            if labels.shape[0] != n_samples:
                raise ValueError(
                    "'sample_labels' must have length equal to the number of samples in `genotypes`."
                )

        calldata_lai = None
        if ancestry is not None:
            if self.calldata_lai is not None:
                calldata_lai = self.calldata_lai
            elif laiobj is not None:
                try:
                    converted_lai = laiobj.convert_to_snp_level(snpobject=self, lai_format="3D")
                    calldata_lai = getattr(converted_lai, "calldata_lai", None)
                except Exception:
                    calldata_lai = None

            if calldata_lai is None:
                raise ValueError(
                    "Ancestry-specific masking requires SNP-level LAI "
                    "(provide a LocalAncestryObject via 'laiobj' or ensure 'self.calldata_lai' is set)."
                )

        afs, counts, pops = aggregate_pop_allele_freq(
            genotypes=gt,
            sample_labels=labels,
            ancestry=ancestry,
            calldata_lai=calldata_lai,
            pseudohaploid=pseudohaploid,
        )

        if grouped_output:
            freq_out = afs
            count_out = counts
            if as_dataframe:
                import pandas as pd

                freq_out = pd.DataFrame(afs, columns=pops)
                count_out = pd.DataFrame(counts, columns=pops)
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

    @staticmethod
    def _format_allele_stat_output(
        values: np.ndarray,
        sample_labels: Optional[Sequence[Any]],
        cohort_column: str,
        as_dataframe: bool,
    ) -> Any:
        if not as_dataframe:
            return values

        import pandas as pd

        if sample_labels is None:
            return pd.DataFrame({cohort_column: np.asarray(values).ravel()})

        labels = np.asarray(sample_labels)
        if labels.ndim != 1:
            labels = labels.ravel()
        pops = np.unique(labels)
        return pd.DataFrame(values, columns=pops)

    def _called_genotype_mask(self) -> np.ndarray:
        if self.genotypes is None:
            raise ValueError("Genotype data `genotypes` is None.")

        gt = np.asarray(self.genotypes)
        if gt.ndim not in (2, 3):
            raise ValueError("'genotypes' must be a 2D or 3D array.")

        try:
            called_entries = np.isfinite(gt) & (gt >= 0)
        except TypeError as exc:
            raise ValueError("'genotypes' must contain numeric values.") from exc

        if gt.ndim == 3:
            return np.all(called_entries, axis=2)
        return called_entries

    def _called_mask_for_samples(
        self,
        sample_indexes: np.ndarray,
        *,
        context: str,
    ) -> np.ndarray:
        sample_indexes = np.asarray(sample_indexes, dtype=int)
        if self.calldata_gp is not None:
            dosages = self._diploid_dosages(samples=sample_indexes, context=context)
            return np.isfinite(dosages)

        called = self._called_genotype_mask()
        return called[:, sample_indexes]

    def _sample_subset_indices(
        self,
        samples: Optional[Union[str, Sequence[str], np.ndarray]],
    ) -> np.ndarray:
        n_samples = self.n_samples
        if samples is None:
            return np.arange(n_samples, dtype=int)

        selector = np.atleast_1d(np.asarray(samples)).ravel()
        if selector.size == 0:
            return np.array([], dtype=int)

        if np.issubdtype(selector.dtype, np.bool_):
            if selector.shape[0] != n_samples:
                raise ValueError(
                    f"Boolean 'samples' mask must have length equal to the number of samples ({n_samples}); "
                    f"got {selector.shape[0]}."
                )
            return np.flatnonzero(selector)

        if np.issubdtype(selector.dtype, np.integer):
            out_of_bounds = selector[(selector < -n_samples) | (selector >= n_samples)]
            if out_of_bounds.size > 0:
                raise ValueError("One or more sample indexes are out of bounds.")
            indexes = np.mod(selector, n_samples).astype(int, copy=False)
            return np.array(list(dict.fromkeys(indexes.tolist())), dtype=int)

        if self.samples is None:
            raise ValueError("Sample names are required when 'samples' is not an index or boolean mask.")

        sample_names = np.asarray(self.samples)
        name_to_idx = {name: idx for idx, name in enumerate(sample_names)}
        missing = [sample for sample in selector if sample not in name_to_idx]
        if missing:
            raise ValueError(f"The following specified samples were not found: {missing}")
        indexes = [name_to_idx[sample] for sample in selector]
        return np.array(list(dict.fromkeys(indexes)), dtype=int)

    @staticmethod
    def _validate_probability_threshold(name: str, value: float) -> float:
        try:
            threshold = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"'{name}' must be a numeric value between 0 and 1.") from exc
        if not 0 <= threshold <= 1:
            raise ValueError(f"'{name}' must be between 0 and 1.")
        return threshold

    @staticmethod
    def _validate_call_rate_threshold(min_call_rate: float) -> float:
        return SNPObject._validate_probability_threshold("min_call_rate", min_call_rate)

    @staticmethod
    def _validate_differential_missingness_test(test: str) -> str:
        test = str(test).lower().replace("-", "_")
        aliases = {
            "auto": "auto",
            "chi2": "chi2",
            "chi_square": "chi2",
            "chisq": "chi2",
            "fisher": "fisher",
            "fisher_exact": "fisher",
        }
        if test not in aliases:
            raise ValueError("'test' must be one of 'auto', 'chi2', or 'fisher'.")
        return aliases[test]

    @staticmethod
    def _validate_integer_parameter(name: str, value: int, min_value: int) -> int:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"'{name}' must be an integer greater than or equal to {min_value}.") from exc
        if not parsed.is_integer():
            raise ValueError(f"'{name}' must be an integer greater than or equal to {min_value}.")
        parsed_int = int(parsed)
        if parsed_int < min_value:
            raise ValueError(f"'{name}' must be greater than or equal to {min_value}.")
        return parsed_int

    @staticmethod
    def _validate_nonnegative_parameter(name: str, value: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"'{name}' must be a non-negative numeric value.") from exc
        if not np.isfinite(parsed) or parsed < 0:
            raise ValueError(f"'{name}' must be non-negative.")
        return parsed

    @staticmethod
    def _format_call_rate_output(
        values: np.ndarray,
        column: str,
        as_dataframe: bool,
    ) -> Any:
        if not as_dataframe:
            return values

        import pandas as pd

        return pd.DataFrame({column: np.asarray(values, dtype=float).ravel()})

    def _format_sample_stat_output(
        self,
        values: np.ndarray,
        sample_indexes: np.ndarray,
        column: str,
        as_dataframe: bool,
    ) -> Any:
        if not as_dataframe:
            return values

        import pandas as pd

        data = {"sample_index": np.asarray(sample_indexes, dtype=int)}
        if self.samples is not None:
            data["sample"] = np.asarray(self.samples)[sample_indexes]
        data[column] = np.asarray(values, dtype=float).ravel()
        return pd.DataFrame(data)

    @staticmethod
    def _validate_duplicate_keep(keep: Union[str, bool]) -> Union[str, bool]:
        if keep is False:
            return False
        if isinstance(keep, str):
            normalized = keep.lower().replace("-", "_")
            aliases: Dict[str, Union[str, bool]] = {
                "first": "first",
                "last": "last",
                "false": False,
                "all": False,
                "none": False,
            }
            if normalized in aliases:
                return aliases[normalized]
        raise ValueError("'keep' must be 'first', 'last', or False.")

    @staticmethod
    def _duplicate_value_is_missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, np.generic):
            value = value.item()
        try:
            if isinstance(value, float) and np.isnan(value):
                return True
        except TypeError:
            pass
        text = str(value).strip()
        return text == "" or text == "."

    @classmethod
    def _duplicate_key(cls, value: Any) -> Any:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and np.isnan(value):
            return ("__nan__",)
        try:
            hash(value)
            return value
        except TypeError:
            return str(value)

    @classmethod
    def _duplicate_mask_and_groups(
        cls,
        keys: Sequence[Any],
        keep: Union[str, bool],
        valid: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        keep = cls._validate_duplicate_keep(keep)
        n_values = len(keys)
        if valid is None:
            valid = np.ones(n_values, dtype=bool)
        else:
            valid = np.asarray(valid, dtype=bool).ravel()
            if valid.shape[0] != n_values:
                raise ValueError("'valid' must have the same length as duplicate keys.")

        positions_by_key: Dict[Any, List[int]] = {}
        for idx, key in enumerate(keys):
            if not valid[idx]:
                continue
            positions_by_key.setdefault(key, []).append(idx)

        duplicate_mask = np.zeros(n_values, dtype=bool)
        duplicate_group = np.full(n_values, -1, dtype=int)
        group_idx = 0
        duplicate_groups = [
            positions for positions in positions_by_key.values()
            if len(positions) > 1
        ]
        duplicate_groups.sort(key=lambda positions: positions[0])
        for positions in duplicate_groups:
            duplicate_group[positions] = group_idx
            group_idx += 1
            if keep == "first":
                marked = positions[1:]
            elif keep == "last":
                marked = positions[:-1]
            else:
                marked = positions
            duplicate_mask[marked] = True

        return duplicate_mask, duplicate_group

    @classmethod
    def _duplicate_mask_for_values(
        cls,
        values: np.ndarray,
        keep: Union[str, bool],
        ignore_missing: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        values = np.asarray(values, dtype=object).ravel()
        valid = np.ones(values.shape[0], dtype=bool)
        if ignore_missing:
            valid = np.array(
                [not cls._duplicate_value_is_missing(value) for value in values],
                dtype=bool,
            )
        keys = [cls._duplicate_key(value) for value in values]
        return cls._duplicate_mask_and_groups(keys, keep=keep, valid=valid)

    @staticmethod
    def _normalize_variant_coordinate_fields(
        fields: Union[str, Sequence[str]],
    ) -> Tuple[Tuple[str, str], ...]:
        if isinstance(fields, str):
            raw_fields = [
                field.strip()
                for field in fields.replace("+", ",").split(",")
                if field.strip()
            ]
        else:
            raw_fields = [str(field).strip() for field in fields]

        if not raw_fields:
            raise ValueError("'fields' must include at least one variant metadata field.")

        aliases = {
            "chrom": ("variants_chrom", "chrom"),
            "chromosome": ("variants_chrom", "chrom"),
            "#chrom": ("variants_chrom", "chrom"),
            "pos": ("variants_pos", "pos"),
            "position": ("variants_pos", "pos"),
            "ref": ("variants_ref", "ref"),
            "reference": ("variants_ref", "ref"),
            "alt": ("variants_alt", "alt"),
            "alternate": ("variants_alt", "alt"),
            "id": ("variants_id", "id"),
            "variant_id": ("variants_id", "id"),
        }

        normalized: List[Tuple[str, str]] = []
        seen: set[str] = set()
        for field in raw_fields:
            key = field.lower()
            if key not in aliases:
                raise ValueError(
                    "'fields' entries must be among chrom, pos, ref, alt, or id."
                )
            attr, label = aliases[key]
            if attr in seen:
                continue
            seen.add(attr)
            normalized.append((attr, label))

        return tuple(normalized)

    def _variant_coordinate_arrays(
        self,
        fields: Union[str, Sequence[str]],
    ) -> Tuple[Tuple[str, str], List[np.ndarray]]:
        normalized_fields = self._normalize_variant_coordinate_fields(fields)
        arrays: List[np.ndarray] = []
        n_snps = self.n_snps
        for attr, _ in normalized_fields:
            values = getattr(self, attr)
            if values is None:
                raise ValueError(f"'{attr}' is required for duplicate variant coordinate QC.")
            arr = np.asarray(values, dtype=object).ravel()
            if arr.shape[0] != n_snps:
                raise ValueError(
                    f"'{attr}' must have length equal to the number of SNPs ({n_snps}); "
                    f"got {arr.shape[0]}."
                )
            arrays.append(arr)
        return normalized_fields, arrays

    @classmethod
    def _duplicate_mask_for_rows(
        cls,
        arrays: Sequence[np.ndarray],
        keep: Union[str, bool],
        ignore_missing: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not arrays:
            raise ValueError("At least one array is required for duplicate row QC.")

        n_values = arrays[0].shape[0]
        valid = np.ones(n_values, dtype=bool)
        keys: List[Tuple[Any, ...]] = []
        for idx in range(n_values):
            row = tuple(array[idx] for array in arrays)
            if ignore_missing and any(cls._duplicate_value_is_missing(value) for value in row):
                valid[idx] = False
            keys.append(tuple(cls._duplicate_key(value) for value in row))

        return cls._duplicate_mask_and_groups(keys, keep=keep, valid=valid)

    @staticmethod
    def _validate_duplicate_variant_by(by: str) -> str:
        key = str(by).lower().replace("-", "_")
        aliases = {
            "id": "id",
            "ids": "id",
            "variant_id": "id",
            "variant_ids": "id",
            "coordinate": "coordinates",
            "coordinates": "coordinates",
            "coord": "coordinates",
            "coords": "coordinates",
            "position": "coordinates",
            "positions": "coordinates",
        }
        if key not in aliases:
            raise ValueError("'by' must be either 'id' or 'coordinates'.")
        return aliases[key]

    @staticmethod
    def _group_labels_missing_mask(labels: np.ndarray) -> np.ndarray:
        try:
            import pandas as pd

            missing = np.asarray(pd.isna(labels), dtype=bool)
        except Exception:
            missing = np.zeros(labels.shape, dtype=bool)
            for idx, value in enumerate(labels):
                missing[idx] = value is None or (
                    isinstance(value, float) and np.isnan(value)
                )

        string_labels = labels.astype(str)
        missing |= np.char.strip(string_labels) == ""
        return missing

    @staticmethod
    def _hashable_group_label(label: Any) -> Any:
        if isinstance(label, np.generic):
            label = label.item()
        try:
            hash(label)
            return label
        except TypeError:
            return str(label)

    @staticmethod
    def _encode_group_labels(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        group_to_index: Dict[Any, int] = {}
        group_names: List[Any] = []
        group_ids = np.empty(labels.shape[0], dtype=int)

        for idx, label in enumerate(labels):
            key = SNPObject._hashable_group_label(label)
            if key not in group_to_index:
                group_to_index[key] = len(group_names)
                group_names.append(label)
            group_ids[idx] = group_to_index[key]

        if len(group_names) < 2:
            raise ValueError("'groups' must contain at least two non-missing groups.")

        return np.asarray(group_names, dtype=object), group_ids

    def _align_group_labels_by_sample(
        self,
        source_samples: Sequence[Any],
        values: np.ndarray,
        sample_indexes: np.ndarray,
        *,
        source_name: str,
    ) -> np.ndarray:
        if self.samples is None:
            raise ValueError(
                f"Sample names are required to align group labels from {source_name}."
            )

        values = np.asarray(values, dtype=object)
        if values.ndim != 1:
            values = values.ravel()

        source_samples = np.asarray(source_samples, dtype=object).ravel()
        if source_samples.shape[0] != values.shape[0]:
            raise ValueError(
                f"{source_name} sample/value length mismatch: "
                f"{source_samples.shape[0]} samples but {values.shape[0]} values."
            )

        source_names = [str(sample) for sample in source_samples.tolist()]
        if len(set(source_names)) != len(source_names):
            raise ValueError(f"{source_name} sample IDs must be unique.")

        value_by_sample = {
            sample_name: values[idx] for idx, sample_name in enumerate(source_names)
        }
        selected_samples = np.asarray(self.samples, dtype=object)[sample_indexes]
        missing = [
            str(sample) for sample in selected_samples
            if str(sample) not in value_by_sample
        ]
        if missing:
            raise ValueError(
                f"{source_name} is missing group labels for samples: {missing}"
            )

        return np.asarray(
            [value_by_sample[str(sample)] for sample in selected_samples],
            dtype=object,
        )

    def _extract_group_column_values(
        self,
        values: np.ndarray,
        *,
        group_column: Optional[str],
        names: Optional[Sequence[str]],
        source_name: str,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=object)
        if values.ndim == 1:
            if group_column is not None and names is not None and group_column not in names:
                raise ValueError(f"Group column '{group_column}' was not found in {source_name}.")
            return values

        if values.ndim != 2:
            raise ValueError(f"{source_name} group values must be one- or two-dimensional.")

        if group_column is None:
            if values.shape[1] == 1:
                return values[:, 0]
            raise ValueError(
                f"`group_column` is required when {source_name} contains multiple columns."
            )

        if names is None:
            raise ValueError(f"{source_name} does not expose column names for `group_column`.")

        names = [str(name) for name in names]
        if str(group_column) not in names:
            raise ValueError(f"Group column '{group_column}' was not found in {source_name}.")
        return values[:, names.index(str(group_column))]

    def _group_labels_for_samples(
        self,
        groups: Any,
        sample_indexes: np.ndarray,
        *,
        group_column: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        sample_indexes = np.asarray(sample_indexes, dtype=int)
        n_selected = sample_indexes.shape[0]
        if n_selected == 0:
            raise ValueError("At least one sample is required for differential missingness.")

        labels: np.ndarray

        if isinstance(groups, Mapping):
            if self.samples is None:
                raise ValueError("Sample names are required when 'groups' is a mapping.")
            selected_samples = np.asarray(self.samples, dtype=object)[sample_indexes]
            string_mapping = {str(key): value for key, value in groups.items()}
            missing = [
                str(sample) for sample in selected_samples
                if str(sample) not in string_mapping
            ]
            if missing:
                raise ValueError(f"'groups' is missing labels for samples: {missing}")
            labels = np.asarray(
                [string_mapping[str(sample)] for sample in selected_samples],
                dtype=object,
            )
        elif hasattr(groups, "phen_df"):
            frame = groups.phen_df
            names = [str(name) for name in getattr(groups, "phenotype_names", [])]
            if group_column is None:
                if len(names) != 1:
                    raise ValueError(
                        "`group_column` is required when a MultiPhenotypeObject "
                        "contains multiple phenotype columns."
                    )
                group_column = names[0]
            columns_by_name = {str(col): col for col in frame.columns}
            if str(group_column) not in columns_by_name:
                raise ValueError(f"Group column '{group_column}' was not found in 'groups'.")
            resolved_group_column = columns_by_name[str(group_column)]
            labels = self._align_group_labels_by_sample(
                getattr(groups, "samples"),
                np.asarray(frame.loc[:, resolved_group_column], dtype=object),
                sample_indexes,
                source_name="groups",
            )
        elif hasattr(groups, "columns") and hasattr(groups, "loc"):
            if group_column is None:
                raise ValueError("`group_column` is required when 'groups' is a DataFrame.")
            columns_by_name = {str(col): col for col in groups.columns}
            if str(group_column) not in columns_by_name:
                raise ValueError(f"Group column '{group_column}' was not found in 'groups'.")
            resolved_group_column = columns_by_name[str(group_column)]
            values = np.asarray(groups.loc[:, resolved_group_column], dtype=object)
            index_values = np.asarray(groups.index, dtype=object)
            if self.samples is not None:
                selected_samples = np.asarray(self.samples, dtype=object)[sample_indexes]
                index_strings = {str(sample) for sample in index_values.tolist()}
                if all(str(sample) in index_strings for sample in selected_samples):
                    labels = self._align_group_labels_by_sample(
                        index_values,
                        values,
                        sample_indexes,
                        source_name="groups",
                    )
                elif values.shape[0] == self.n_samples:
                    labels = values[sample_indexes]
                else:
                    raise ValueError(
                        "'groups' DataFrame must be indexed by sample ID or have "
                        "one row per SNPObject sample."
                    )
            elif values.shape[0] == self.n_samples:
                labels = values[sample_indexes]
            else:
                raise ValueError(
                    "'groups' DataFrame must have one row per SNPObject sample when "
                    "SNPObject.samples is None."
                )
        elif hasattr(groups, "index") and hasattr(groups, "to_numpy"):
            values = np.asarray(groups.to_numpy(), dtype=object).ravel()
            index_values = np.asarray(groups.index, dtype=object)
            if self.samples is not None:
                selected_samples = np.asarray(self.samples, dtype=object)[sample_indexes]
                index_strings = {str(sample) for sample in index_values.tolist()}
                if all(str(sample) in index_strings for sample in selected_samples):
                    labels = self._align_group_labels_by_sample(
                        index_values,
                        values,
                        sample_indexes,
                        source_name="groups",
                    )
                elif values.shape[0] == self.n_samples:
                    labels = values[sample_indexes]
                elif values.shape[0] == n_selected:
                    labels = values
                else:
                    raise ValueError(
                        "'groups' Series must be indexed by sample ID, have one "
                        "entry per SNPObject sample, or have one entry per selected sample."
                    )
            elif values.shape[0] == self.n_samples:
                labels = values[sample_indexes]
            elif values.shape[0] == n_selected:
                labels = values
            else:
                raise ValueError(
                    "'groups' Series must have one entry per SNPObject sample or "
                    "one entry per selected sample when SNPObject.samples is None."
                )
        elif hasattr(groups, "samples") and hasattr(groups, "values"):
            names = (
                getattr(groups, "phenotype_names", None)
                or getattr(groups, "covariate_names", None)
                or getattr(groups, "names", None)
            )
            values = self._extract_group_column_values(
                np.asarray(getattr(groups, "values"), dtype=object),
                group_column=group_column,
                names=names,
                source_name=type(groups).__name__,
            )
            labels = self._align_group_labels_by_sample(
                getattr(groups, "samples"),
                values,
                sample_indexes,
                source_name=type(groups).__name__,
            )
        else:
            values = np.asarray(groups, dtype=object)
            if values.ndim == 0:
                raise ValueError("'groups' must contain one label per sample.")
            if values.ndim != 1:
                values = values.ravel()
            if values.shape[0] == self.n_samples:
                labels = values[sample_indexes]
            elif values.shape[0] == n_selected:
                labels = values
            else:
                raise ValueError(
                    "'groups' must have length equal to the number of SNPObject "
                    "samples or the number of selected samples."
                )

        missing = self._group_labels_missing_mask(labels)
        if np.any(missing):
            raise ValueError("'groups' contains missing labels for selected samples.")

        group_names, group_ids = self._encode_group_labels(labels)
        return labels, group_names, group_ids

    @staticmethod
    def _chi2_missingness_stats(
        missing_counts: np.ndarray,
        called_counts: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        missing_counts = np.asarray(missing_counts, dtype=float)
        called_counts = np.asarray(called_counts, dtype=float)
        n_variants, n_groups = missing_counts.shape

        p_values = np.full(n_variants, np.nan, dtype=float)
        statistics = np.full(n_variants, np.nan, dtype=float)
        min_expected = np.full(n_variants, np.nan, dtype=float)

        group_totals = missing_counts + called_counts
        row_missing = missing_counts.sum(axis=1)
        row_called = called_counts.sum(axis=1)
        grand_total = row_missing + row_called

        degenerate = (grand_total > 0) & ((row_missing == 0) | (row_called == 0))
        p_values[degenerate] = 1.0
        statistics[degenerate] = 0.0
        min_expected[degenerate] = 0.0

        valid = (
            (grand_total > 0)
            & (row_missing > 0)
            & (row_called > 0)
            & np.all(group_totals > 0, axis=1)
        )
        if not np.any(valid):
            return p_values, statistics, min_expected

        expected_missing = (
            row_missing[valid, None] * group_totals[valid] / grand_total[valid, None]
        )
        expected_called = (
            row_called[valid, None] * group_totals[valid] / grand_total[valid, None]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            stat = (
                ((missing_counts[valid] - expected_missing) ** 2 / expected_missing).sum(axis=1)
                + ((called_counts[valid] - expected_called) ** 2 / expected_called).sum(axis=1)
            )
        statistics[valid] = stat
        p_values[valid] = chi2.sf(stat, df=n_groups - 1)
        min_expected[valid] = np.minimum(
            expected_missing.min(axis=1),
            expected_called.min(axis=1),
        )
        return p_values, statistics, min_expected

    @staticmethod
    def _fisher_missingness_stats(
        missing_counts: np.ndarray,
        called_counts: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if missing_counts.shape[1] != 2:
            raise ValueError("Fisher exact differential missingness requires exactly two groups.")

        n_variants = missing_counts.shape[0]
        p_values = np.full(n_variants, np.nan, dtype=float)
        statistics = np.full(n_variants, np.nan, dtype=float)

        row_missing = missing_counts.sum(axis=1)
        row_called = called_counts.sum(axis=1)
        degenerate = (row_missing == 0) | (row_called == 0)
        p_values[degenerate] = 1.0

        for variant_idx in np.flatnonzero(~degenerate):
            table = np.vstack(
                [missing_counts[variant_idx], called_counts[variant_idx]]
            ).astype(int, copy=False)
            result = fisher_exact(table)
            if hasattr(result, "statistic"):
                statistics[variant_idx] = float(result.statistic)
                p_values[variant_idx] = float(result.pvalue)
            else:
                odds_ratio, p_value = result
                statistics[variant_idx] = float(odds_ratio)
                p_values[variant_idx] = float(p_value)

        return p_values, statistics

    @staticmethod
    def _differential_missingness_column_name(
        prefix: str,
        group_name: Any,
        used: Dict[str, int],
    ) -> str:
        label = str(group_name).strip() or "group"
        column = f"{prefix}_{label}"
        seen = used.get(column, 0)
        used[column] = seen + 1
        if seen:
            return f"{column}_{seen + 1}"
        return column

    def _format_differential_missingness_output(
        self,
        p_values: np.ndarray,
        statistics: np.ndarray,
        method_names: np.ndarray,
        missing_counts: np.ndarray,
        called_counts: np.ndarray,
        group_names: np.ndarray,
        as_dataframe: bool,
    ) -> Any:
        if not as_dataframe:
            return p_values

        import pandas as pd

        total_missing = missing_counts.sum(axis=1)
        total_called = called_counts.sum(axis=1)
        total = total_missing + total_called
        data: Dict[str, Any] = {
            "differential_missingness_pvalue": np.asarray(p_values, dtype=float),
            "statistic": np.asarray(statistics, dtype=float),
            "test": np.asarray(method_names, dtype=object),
            "missing_count": total_missing.astype(int),
            "called_count": total_called.astype(int),
            "n_samples": total.astype(int),
        }

        column_counts = {column: 1 for column in data}
        for group_idx, group_name in enumerate(group_names):
            for prefix, values in (
                ("missing_count", missing_counts[:, group_idx].astype(int)),
                ("called_count", called_counts[:, group_idx].astype(int)),
            ):
                column = self._differential_missingness_column_name(
                    prefix,
                    group_name,
                    column_counts,
                )
                data[column] = values

            group_total = missing_counts[:, group_idx] + called_counts[:, group_idx]
            rate = np.full(group_total.shape, np.nan, dtype=float)
            nonzero = group_total > 0
            rate[nonzero] = missing_counts[nonzero, group_idx] / group_total[nonzero]
            column = self._differential_missingness_column_name(
                "missing_rate",
                group_name,
                column_counts,
            )
            data[column] = rate

        return pd.DataFrame(data)

    def _hard_call_dosages(
        self,
        samples: Optional[Union[str, Sequence[str], np.ndarray]] = None,
        *,
        context: str = "HWE exact test",
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.genotypes is None:
            raise ValueError("Genotype data `genotypes` is None.")

        sample_indexes = self._sample_subset_indices(samples)
        gt = np.asarray(self.genotypes)
        if gt.ndim not in (2, 3):
            raise ValueError("'genotypes' must be a 2D or 3D array.")

        try:
            gt = gt.astype(float, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("'genotypes' must contain numeric hard calls.") from exc

        if gt.ndim == 3:
            subset = gt[:, sample_indexes, :]
            called_entries = np.isfinite(subset) & (subset >= 0)
            called = np.all(called_entries, axis=2)
            called_alleles = subset[called]
            if called_alleles.size and not np.all(np.isclose(called_alleles, 0) | np.isclose(called_alleles, 1)):
                raise ValueError(f"{context} requires biallelic hard calls encoded as 0/1 alleles.")
            dosages = np.full(called.shape, np.nan, dtype=float)
            if called_alleles.size:
                dosages[called] = called_alleles.sum(axis=1)
            return dosages, called

        subset = gt[:, sample_indexes]
        called = np.isfinite(subset) & (subset >= 0)
        called_dosages = subset[called]
        valid_dosages = (
            np.isclose(called_dosages, 0)
            | np.isclose(called_dosages, 1)
            | np.isclose(called_dosages, 2)
        )
        if called_dosages.size and not np.all(valid_dosages):
            raise ValueError(f"{context} requires hard-call dosages encoded as 0, 1, or 2.")
        dosages = np.where(called, np.rint(subset), np.nan)
        return dosages, called

    def _diploid_dosages(
        self,
        samples: Optional[Union[str, Sequence[str], np.ndarray]] = None,
        *,
        context: str = "Diploid dosage calculation",
    ) -> np.ndarray:
        sample_indexes = self._sample_subset_indices(samples)

        if self.calldata_gp is not None:
            dosage = np.asarray(self.dosage(), dtype=float)
            if dosage.ndim != 2:
                raise ValueError("Expected dosage data with shape (n_snps, n_samples).")
            subset = dosage[:, sample_indexes]
            called = np.isfinite(subset) & (subset >= 0)
            called_dosages = subset[called]
            if called_dosages.size and np.any((called_dosages < 0) | (called_dosages > 2)):
                raise ValueError(f"{context} requires dosages between 0 and 2.")
            return np.where(called, subset, np.nan)

        if self.genotypes is None:
            raise ValueError("SNPObject requires either `genotypes` or `calldata_gp` for dosage-based QC.")

        gt = np.asarray(self.genotypes)
        if gt.ndim not in (2, 3):
            raise ValueError("'genotypes' must be a 2D or 3D array.")

        try:
            gt = gt.astype(float, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("'genotypes' must contain numeric values.") from exc

        if gt.ndim == 3:
            subset = gt[:, sample_indexes, :]
            called_entries = np.isfinite(subset) & (subset >= 0)
            called = np.all(called_entries, axis=2)
            called_alleles = subset[called]
            if called_alleles.size and not np.all(np.isclose(called_alleles, 0) | np.isclose(called_alleles, 1)):
                raise ValueError(f"{context} requires biallelic allele calls encoded as 0/1.")
            dosages = np.full(called.shape, np.nan, dtype=float)
            if called_alleles.size:
                dosages[called] = called_alleles.sum(axis=1)
            return dosages

        subset = gt[:, sample_indexes]
        called = np.isfinite(subset) & (subset >= 0)
        called_dosages = subset[called]
        if called_dosages.size and np.any((called_dosages < 0) | (called_dosages > 2)):
            raise ValueError(f"{context} requires dosages between 0 and 2.")
        return np.where(called, subset, np.nan)

    def _sample_heterozygosity_components(
        self,
        samples: Optional[Union[str, Sequence[str], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sample_indexes = self._sample_subset_indices(samples)
        dosages, called = self._hard_call_dosages(
            samples=sample_indexes,
            context="Sample heterozygosity",
        )
        heterozygous = called & (dosages == 1)
        called_counts = called.sum(axis=0).astype(float)
        observed_hets = heterozygous.sum(axis=0).astype(float)
        heterozygosity = np.full(called_counts.shape, np.nan, dtype=float)
        nonzero = called_counts > 0
        heterozygosity[nonzero] = observed_hets[nonzero] / called_counts[nonzero]
        return sample_indexes, dosages, called, observed_hets, called_counts, heterozygosity

    def _sample_inbreeding_components(
        self,
        samples: Optional[Union[str, Sequence[str], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sample_indexes, dosages, called, observed_hets, called_counts, _ = (
            self._sample_heterozygosity_components(samples=samples)
        )
        called_alleles = called.sum(axis=1).astype(float) * 2.0
        alt_counts = np.where(called, dosages, 0.0).sum(axis=1)
        allele_freq = np.full(called_alleles.shape, np.nan, dtype=float)
        variant_called = called_alleles > 0
        allele_freq[variant_called] = alt_counts[variant_called] / called_alleles[variant_called]
        expected_per_variant = 2.0 * allele_freq * (1.0 - allele_freq)
        expected_hets = np.where(
            called,
            np.nan_to_num(expected_per_variant, nan=0.0)[:, None],
            0.0,
        ).sum(axis=0)

        inbreeding = np.full(observed_hets.shape, np.nan, dtype=float)
        informative = expected_hets > 0
        inbreeding[informative] = 1.0 - (observed_hets[informative] / expected_hets[informative])
        return sample_indexes, inbreeding, expected_hets, observed_hets, called_counts

    @staticmethod
    def _hwe_exact_pvalue(obs_hets: int, obs_hom_ref: int, obs_hom_alt: int) -> float:
        genotypes = obs_hets + obs_hom_ref + obs_hom_alt
        if genotypes == 0:
            return np.nan

        obs_hom_rare = min(obs_hom_ref, obs_hom_alt)
        rare_copies = 2 * obs_hom_rare + obs_hets
        probs = np.zeros(rare_copies + 1, dtype=float)

        mid = int(rare_copies * (2 * genotypes - rare_copies) / (2 * genotypes))
        if (rare_copies & 1) != (mid & 1):
            mid += 1

        probs[mid] = 1.0
        total = probs[mid]

        curr_hets = mid
        curr_hom_rare = (rare_copies - curr_hets) // 2
        curr_hom_common = genotypes - curr_hets - curr_hom_rare
        while curr_hets > 1:
            prob = (
                probs[curr_hets]
                * curr_hets
                * (curr_hets - 1)
                / (4.0 * (curr_hom_rare + 1) * (curr_hom_common + 1))
            )
            curr_hets -= 2
            curr_hom_rare += 1
            curr_hom_common += 1
            probs[curr_hets] = prob
            total += prob

        curr_hets = mid
        curr_hom_rare = (rare_copies - curr_hets) // 2
        curr_hom_common = genotypes - curr_hets - curr_hom_rare
        while curr_hets <= rare_copies - 2:
            prob = (
                probs[curr_hets]
                * 4.0
                * curr_hom_rare
                * curr_hom_common
                / ((curr_hets + 2) * (curr_hets + 1))
            )
            curr_hets += 2
            curr_hom_rare -= 1
            curr_hom_common -= 1
            probs[curr_hets] = prob
            total += prob

        probs /= total
        p_value = probs[probs <= probs[obs_hets] + 1e-12].sum()
        return min(1.0, float(p_value))

    def _ld_dosages(
        self,
        samples: Optional[Union[str, Sequence[str], np.ndarray]] = None,
    ) -> np.ndarray:
        return self._diploid_dosages(samples=samples, context="LD pruning")

    @staticmethod
    def _pairwise_r2_ignore_missing(
        first: np.ndarray,
        second: np.ndarray,
        min_samples: int,
    ) -> float:
        valid = np.isfinite(first) & np.isfinite(second)
        if np.count_nonzero(valid) < min_samples:
            return np.nan

        x = first[valid]
        y = second[valid]
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        ssx = np.dot(x_centered, x_centered)
        ssy = np.dot(y_centered, y_centered)
        if ssx <= 0 or ssy <= 0:
            return np.nan

        r = np.dot(x_centered, y_centered) / np.sqrt(ssx * ssy)
        return min(1.0, float(r * r))

    @staticmethod
    def _validate_relatedness_method(method: str) -> str:
        method = str(method).lower().replace("-", "_")
        if method not in {"grm", "ibd"}:
            raise ValueError("'method' must be either 'grm' or 'ibd'.")
        return method

    @staticmethod
    def _validate_relatedness_scale(scale: str) -> str:
        aliases = {
            "relationship": "relationship",
            "grm": "relationship",
            "kinship": "kinship",
        }
        key = str(scale).lower().replace("-", "_")
        if key not in aliases:
            raise ValueError("'scale' must be either 'relationship' or 'kinship'.")
        return aliases[key]

    @staticmethod
    def _validate_positive_parameter(name: str, value: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"'{name}' must be a positive numeric value.") from exc
        if not np.isfinite(parsed) or parsed <= 0:
            raise ValueError(f"'{name}' must be positive.")
        return parsed

    @staticmethod
    def _validate_optional_nonnegative_parameter(name: str, value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        return SNPObject._validate_nonnegative_parameter(name, value)

    def _format_relatedness_output(
        self,
        values: np.ndarray,
        sample_indexes: np.ndarray,
        as_dataframe: bool,
    ) -> Any:
        if not as_dataframe:
            return values

        import pandas as pd

        if self.samples is not None:
            labels = np.asarray(self.samples)[sample_indexes]
        else:
            labels = sample_indexes
        return pd.DataFrame(values, index=labels, columns=labels)

    def _grm_relatedness_components(
        self,
        samples: Optional[Union[str, Sequence[str], np.ndarray]] = None,
        *,
        min_variants: int = 1,
        block_size: int = 10000,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        min_variants = self._validate_integer_parameter("min_variants", min_variants, 1)
        block_size = self._validate_integer_parameter("block_size", block_size, 1)
        sample_indexes = self._sample_subset_indices(samples)
        dosages = self._diploid_dosages(samples=sample_indexes, context="GRM relatedness")
        n_samples = dosages.shape[1]

        numerator = np.zeros((n_samples, n_samples), dtype=float)
        counts = np.zeros((n_samples, n_samples), dtype=np.int64)

        for start in range(0, dosages.shape[0], block_size):
            stop = min(start + block_size, dosages.shape[0])
            block = dosages[start:stop]
            called = np.isfinite(block)
            called_per_variant = called.sum(axis=1)
            informative_input = called_per_variant > 0
            if not np.any(informative_input):
                continue

            allele_sum = np.where(called, block, 0.0).sum(axis=1)
            allele_freq = np.full(block.shape[0], np.nan, dtype=float)
            allele_freq[informative_input] = (
                allele_sum[informative_input] / (2.0 * called_per_variant[informative_input])
            )
            variance = 2.0 * allele_freq * (1.0 - allele_freq)
            informative = np.isfinite(variance) & (variance > 0)
            if not np.any(informative):
                continue

            block = block[informative].astype(float, copy=True)
            called = called[informative]
            allele_freq = allele_freq[informative]
            scale = np.sqrt(2.0 * allele_freq * (1.0 - allele_freq))
            standardized = (block - (2.0 * allele_freq[:, None])) / scale[:, None]
            standardized[~called] = 0.0

            called_int = called.astype(np.int64, copy=False)
            numerator += standardized.T @ standardized
            counts += called_int.T @ called_int

        relatedness = np.full(numerator.shape, np.nan, dtype=float)
        valid = counts >= min_variants
        relatedness[valid] = numerator[valid] / counts[valid]
        return sample_indexes, relatedness, counts

    def _ibd_relatedness_components(
        self,
        ibdobj: Any,
        samples: Optional[Union[str, Sequence[str], np.ndarray]] = None,
        *,
        genome_length_cm: float = 3400.0,
        min_segment_cm: Optional[float] = None,
        segment_types: Optional[Sequence[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        if ibdobj is None:
            raise ValueError("`ibdobj` is required when method='ibd'.")
        if self.samples is None:
            raise ValueError("Sample names are required on `SNPObject.samples` when method='ibd'.")

        genome_length_cm = self._validate_positive_parameter("genome_length_cm", genome_length_cm)
        min_segment_cm = self._validate_optional_nonnegative_parameter("min_segment_cm", min_segment_cm)
        sample_indexes = self._sample_subset_indices(samples)
        sample_names = np.asarray(self.samples)
        sample_to_position = {str(sample_names[idx]): pos for pos, idx in enumerate(sample_indexes)}
        n_samples = sample_indexes.size

        length_cm = getattr(ibdobj, "length_cm", None)
        if length_cm is None:
            raise ValueError("`ibdobj.length_cm` is required for IBD relatedness.")
        length_cm = np.asarray(length_cm, dtype=float)

        sample_id_1 = np.asarray(getattr(ibdobj, "sample_id_1"))
        sample_id_2 = np.asarray(getattr(ibdobj, "sample_id_2"))
        if sample_id_1.shape[0] != length_cm.shape[0] or sample_id_2.shape[0] != length_cm.shape[0]:
            raise ValueError("IBD sample ID arrays must have the same length as `ibdobj.length_cm`.")

        segment_type = getattr(ibdobj, "segment_type", None)
        if segment_type is not None:
            segment_type = np.asarray(segment_type)
            if segment_type.shape[0] != length_cm.shape[0]:
                raise ValueError("`ibdobj.segment_type` must have the same length as `ibdobj.length_cm`.")
        elif segment_types is not None:
            raise ValueError("`ibdobj.segment_type` is required when filtering by `segment_types`.")

        mask = np.isfinite(length_cm) & (length_cm > 0)
        if min_segment_cm is not None:
            mask &= length_cm >= min_segment_cm
        if segment_types is not None:
            allowed_types = np.char.upper(np.atleast_1d(np.asarray(segment_types, dtype=str)))
            observed_types = np.char.upper(segment_type.astype(str))
            mask &= np.isin(observed_types, allowed_types)

        ibd1_cm = np.zeros((n_samples, n_samples), dtype=float)
        ibd2_cm = np.zeros((n_samples, n_samples), dtype=float)
        weighted_ibd_cm = np.zeros((n_samples, n_samples), dtype=float)
        n_segments = np.zeros((n_samples, n_samples), dtype=np.int64)

        for idx in np.flatnonzero(mask):
            first = sample_to_position.get(str(sample_id_1[idx]))
            second = sample_to_position.get(str(sample_id_2[idx]))
            if first is None or second is None or first == second:
                continue

            segment_length = float(length_cm[idx])
            type_label = "" if segment_type is None else str(segment_type[idx]).upper()
            weight = 2.0 if type_label == "IBD2" else 1.0
            i, j = (first, second) if first < second else (second, first)

            if weight == 2.0:
                ibd2_cm[i, j] += segment_length
            else:
                ibd1_cm[i, j] += segment_length
            weighted_ibd_cm[i, j] += segment_length * weight
            n_segments[i, j] += 1

        for matrix in (ibd1_cm, ibd2_cm, weighted_ibd_cm, n_segments):
            matrix += matrix.T

        relationship = weighted_ibd_cm / (2.0 * genome_length_cm)
        np.fill_diagonal(relationship, 1.0)

        metrics = {
            "n_segments": n_segments,
            "ibd1_cm": ibd1_cm,
            "ibd2_cm": ibd2_cm,
            "weighted_ibd_cm": weighted_ibd_cm,
        }
        return sample_indexes, relationship, metrics

    def _sample_call_rate_from_dosages(self, sample_indexes: np.ndarray) -> np.ndarray:
        dosages = self._diploid_dosages(samples=sample_indexes, context="Relatedness pruning")
        if dosages.shape[0] == 0:
            return np.full(dosages.shape[1], np.nan, dtype=float)
        return np.isfinite(dosages).mean(axis=0)

    def _sample_call_rate_for_pruning(self, sample_indexes: np.ndarray) -> np.ndarray:
        try:
            return self._sample_call_rate_from_dosages(sample_indexes)
        except ValueError:
            return np.ones(sample_indexes.shape[0], dtype=float)

    @staticmethod
    def _normalize_imputation_info_keys(
        info_keys: Optional[Union[str, Sequence[str]]],
    ) -> Tuple[str, ...]:
        if info_keys is None:
            return SNPObject._DEFAULT_IMPUTATION_INFO_KEYS
        if isinstance(info_keys, str):
            return (info_keys,)
        return tuple(str(key) for key in info_keys)

    @staticmethod
    def _coerce_info_float(value: Any) -> float:
        if value is None:
            return np.nan
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value, dtype=object).ravel()
            if arr.size == 0:
                return np.nan
            value = arr[0]
        if isinstance(value, str):
            value = value.strip()
            if "," in value:
                value = value.split(",", 1)[0].strip()
            if value in {"", ".", "NA", "NaN", "nan", "None"}:
                return np.nan
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return np.nan
        if not np.isfinite(parsed):
            return np.nan
        return parsed

    @classmethod
    def _parse_imputation_info_value(cls, info: Any, info_keys: Tuple[str, ...]) -> float:
        if info is None:
            return np.nan
        if isinstance(info, bytes):
            info = info.decode("utf-8", errors="ignore")
        if isinstance(info, dict):
            upper_map = {str(key).upper(): value for key, value in info.items()}
            for key in info_keys:
                if key in info:
                    return cls._coerce_info_float(info[key])
                value = upper_map.get(key.upper())
                if value is not None:
                    return cls._coerce_info_float(value)
            return np.nan
        if isinstance(info, (int, float, np.number)):
            return cls._coerce_info_float(info)

        info_str = str(info).strip()
        if info_str in {"", ".", "NA", "NaN", "nan", "None"}:
            return np.nan
        bare_value = cls._coerce_info_float(info_str)
        if np.isfinite(bare_value) and "=" not in info_str and ";" not in info_str:
            return bare_value

        fields = {}
        for item in info_str.split(";"):
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            fields[key.strip()] = value.strip()
        upper_fields = {key.upper(): value for key, value in fields.items()}
        for key in info_keys:
            if key in fields:
                return cls._coerce_info_float(fields[key])
            value = upper_fields.get(key.upper())
            if value is not None:
                return cls._coerce_info_float(value)
        return np.nan

    def _imputation_r2_from_info(
        self,
        info_keys: Optional[Union[str, Sequence[str]]] = None,
    ) -> np.ndarray:
        info = self.variants_info
        values = np.full(self.n_snps, np.nan, dtype=float)
        if info is None:
            return values

        info_array = np.asarray(info, dtype=object).ravel()
        if info_array.size == 0:
            return values
        if info_array.shape[0] != self.n_snps:
            raise ValueError(
                "`variants_info` must have length equal to the number of variants "
                "to extract imputation quality metrics."
            )

        keys = self._normalize_imputation_info_keys(info_keys)
        for idx, info_value in enumerate(info_array):
            values[idx] = self._parse_imputation_info_value(info_value, keys)
        return values

    @staticmethod
    def _validate_bgen_probability_padding(probabilities: np.ndarray) -> np.ndarray:
        finite = np.isfinite(probabilities)
        widths = finite.sum(axis=1)
        for sample_idx, width in enumerate(widths):
            if width == 0:
                continue
            if finite[sample_idx, :width].all() and not finite[sample_idx, width:].any():
                continue
            raise ValueError(
                "BGEN probability rows may only contain NaN values as all-missing rows "
                "or as trailing padding for lower-ploidy samples."
            )
        return widths

    def _imputation_r2_from_gp(self) -> np.ndarray:
        if self.calldata_gp is None:
            return np.full(self.n_snps, np.nan, dtype=float)

        if self.variants_alt is not None:
            alt = np.asarray(self.variants_alt, dtype=object).astype(str)
            multiallelic = np.char.find(alt, ",") >= 0
            if np.any(multiallelic):
                first = int(np.flatnonzero(multiallelic)[0])
                raise ValueError(
                    "BGEN imputation R2 currently supports only biallelic variants; "
                    f"variant {first} has ALT={alt[first]!r}."
                )

        gp = np.asarray(self.calldata_gp, dtype=np.float32)
        if gp.ndim != 3:
            raise ValueError("`calldata_gp` must have shape (n_snps, n_samples, n_probabilities).")

        r2 = np.full(gp.shape[0], np.nan, dtype=float)
        for variant_idx, probabilities in enumerate(gp):
            widths = self._validate_bgen_probability_padding(probabilities)
            phased = self._bgen_biallelic_rows_look_phased(probabilities, widths)
            dosage_sum = 0.0
            ploidy_sum = 0.0
            posterior_var_sum = 0.0
            called = 0

            for sample_idx, width in enumerate(widths):
                width = int(width)
                if width == 0:
                    continue

                sample_probs = probabilities[sample_idx, :width].astype(float, copy=False)

                if phased:
                    if width % 2 != 0:
                        raise ValueError(
                            f"Cannot compute phased BGEN imputation R2 for variant {variant_idx}, "
                            f"sample {sample_idx}: expected an even probability width, got {width}."
                        )
                    haplotypes = sample_probs.reshape(width // 2, 2)
                    haplotype_sums = haplotypes.sum(axis=1)
                    if not np.all(np.isfinite(haplotype_sums)) or np.any(haplotype_sums <= 0):
                        continue
                    haplotypes = haplotypes / haplotype_sums[:, None]
                    alt_probs = haplotypes[:, 1]
                    ploidy = alt_probs.size
                    dosage = float(alt_probs.sum())
                    posterior_var = float(np.sum(alt_probs * (1.0 - alt_probs)))
                else:
                    prob_sum = sample_probs.sum()
                    if not np.isfinite(prob_sum) or prob_sum <= 0:
                        continue
                    sample_probs = sample_probs / prob_sum
                    ploidy = width - 1
                    if ploidy <= 0:
                        continue
                    genotype_values = np.arange(width, dtype=float)
                    dosage = float(np.dot(sample_probs, genotype_values))
                    expected_square = float(np.dot(sample_probs, genotype_values * genotype_values))
                    posterior_var = max(0.0, expected_square - dosage * dosage)

                dosage_sum += dosage
                ploidy_sum += float(ploidy)
                posterior_var_sum += posterior_var
                called += 1

            if called == 0 or ploidy_sum <= 0:
                continue

            allele_freq = dosage_sum / ploidy_sum
            denominator = (ploidy_sum / called) * allele_freq * (1.0 - allele_freq)
            if denominator <= 0:
                continue

            estimate = 1.0 - (posterior_var_sum / called) / denominator
            r2[variant_idx] = min(1.0, max(0.0, float(estimate)))
        return r2

    def _imputation_r2_from_dosage(self) -> np.ndarray:
        dosages = self._ld_dosages()
        r2 = np.full(dosages.shape[0], np.nan, dtype=float)
        for variant_idx, row in enumerate(dosages):
            called = np.isfinite(row)
            if np.count_nonzero(called) < 2:
                continue

            observed = row[called]
            allele_freq = observed.mean() / 2.0
            denominator = 2.0 * allele_freq * (1.0 - allele_freq)
            if denominator <= 0:
                continue

            estimate = np.var(observed, ddof=0) / denominator
            r2[variant_idx] = min(1.0, max(0.0, float(estimate)))
        return r2

    @staticmethod
    def _validate_imputation_r2_source(source: str) -> str:
        aliases = {
            "auto": "auto",
            "info": "info",
            "gp": "gp",
            "bgen": "gp",
            "probability": "gp",
            "probabilities": "gp",
            "calldata_gp": "gp",
            "dosage": "dosage",
            "dosages": "dosage",
            "ds": "dosage",
        }
        key = str(source).lower().replace("-", "_")
        if key not in aliases:
            raise ValueError("'source' must be one of 'auto', 'info', 'gp', or 'dosage'.")
        return aliases[key]

    def allele_counts(
        self,
        sample_labels: Optional[Sequence[Any]] = None,
        ancestry: Optional[Union[str, int]] = None,
        laiobj: Optional["LocalAncestryObject"] = None,
        pseudohaploid: Union[bool, int] = False,
        return_called: bool = False,
        as_dataframe: bool = False,
    ) -> Any:
        """
        Compute per-SNP alternate allele counts from observed calls.

        This uses the same missing-data handling as :meth:`allele_freq`: missing
        calls do not contribute to either the alternate allele count or the
        called-allele denominator. For 2D dosage arrays, alternate allele counts
        may be fractional when dosages are fractional.

        Args:
            sample_labels (sequence, optional):
                Population label per sample. If None, computes cohort-level counts.
            ancestry (str or int, optional):
                If provided, compute ancestry-masked counts using SNP-level LAI.
            laiobj (LocalAncestryObject, optional):
                Optional LAI object used when `self.calldata_lai` is not set.
            pseudohaploid (bool or int, default=False):
                If True, detects pseudo-haploid samples using the same rule as
                :meth:`allele_freq`. If an integer `n` is provided, checks the first
                `n` SNPs.
            return_called (bool, default=False):
                If True, also return called-allele counts with the same shape as
                the alternate allele counts.
            as_dataframe (bool, default=False):
                If True, return pandas DataFrame output.

        Returns:
            Alternate allele counts as a NumPy array (or DataFrame if
            ``as_dataframe=True``). If ``return_called=True``, returns
            ``(alt_counts, called_alleles)``.
        """
        allele_freq, called_alleles = self.allele_freq(
            sample_labels=sample_labels,
            ancestry=ancestry,
            laiobj=laiobj,
            pseudohaploid=pseudohaploid,
            return_counts=True,
            as_dataframe=False,
        )
        allele_freq = np.asarray(allele_freq, dtype=float)
        called_alleles = np.asarray(called_alleles)
        called_float = called_alleles.astype(float, copy=False)

        with np.errstate(invalid="ignore"):
            alt_counts = allele_freq * called_float
        alt_counts = np.where(called_float > 0, alt_counts, np.nan)

        alt_out = self._format_allele_stat_output(
            alt_counts,
            sample_labels=sample_labels,
            cohort_column="alt_allele_count",
            as_dataframe=as_dataframe,
        )
        if not return_called:
            return alt_out

        called_out = self._format_allele_stat_output(
            called_alleles,
            sample_labels=sample_labels,
            cohort_column="called_alleles",
            as_dataframe=as_dataframe,
        )
        return alt_out, called_out

    def maf(
        self,
        sample_labels: Optional[Sequence[Any]] = None,
        ancestry: Optional[Union[str, int]] = None,
        laiobj: Optional["LocalAncestryObject"] = None,
        pseudohaploid: Union[bool, int] = False,
        as_dataframe: bool = False,
    ) -> Any:
        """
        Compute per-SNP minor allele frequency from observed calls.

        Missing calls are excluded from the denominator. Variants with no called
        alleles return ``NaN``.
        """
        allele_freq = self.allele_freq(
            sample_labels=sample_labels,
            ancestry=ancestry,
            laiobj=laiobj,
            pseudohaploid=pseudohaploid,
            return_counts=False,
            as_dataframe=False,
        )
        allele_freq = np.asarray(allele_freq, dtype=float)
        with np.errstate(invalid="ignore"):
            minor_af = np.minimum(allele_freq, 1.0 - allele_freq)
        return self._format_allele_stat_output(
            minor_af,
            sample_labels=sample_labels,
            cohort_column="maf",
            as_dataframe=as_dataframe,
        )

    def mac(
        self,
        sample_labels: Optional[Sequence[Any]] = None,
        ancestry: Optional[Union[str, int]] = None,
        laiobj: Optional["LocalAncestryObject"] = None,
        pseudohaploid: Union[bool, int] = False,
        as_dataframe: bool = False,
    ) -> Any:
        """
        Compute per-SNP minor allele count from observed calls.

        Missing calls are excluded from the denominator. Variants with no called
        alleles return ``NaN``.
        """
        alt_counts, called_alleles = self.allele_counts(
            sample_labels=sample_labels,
            ancestry=ancestry,
            laiobj=laiobj,
            pseudohaploid=pseudohaploid,
            return_called=True,
            as_dataframe=False,
        )
        alt_counts = np.asarray(alt_counts, dtype=float)
        called_alleles = np.asarray(called_alleles, dtype=float)
        with np.errstate(invalid="ignore"):
            minor_ac = np.minimum(alt_counts, called_alleles - alt_counts)
        minor_ac = np.where(called_alleles > 0, minor_ac, np.nan)
        return self._format_allele_stat_output(
            minor_ac,
            sample_labels=sample_labels,
            cohort_column="mac",
            as_dataframe=as_dataframe,
        )

    def variant_call_rate(self, as_dataframe: bool = False) -> Any:
        """
        Compute the fraction of samples with non-missing genotype calls per variant.

        Missing calls are represented as negative values or NaN. For 3D genotype
        arrays, a sample is counted as called only when all allele entries are
        non-missing.
        """
        called = self._called_genotype_mask()
        if called.shape[1] == 0:
            call_rate = np.full(called.shape[0], np.nan, dtype=float)
        else:
            call_rate = called.mean(axis=1)
        return self._format_call_rate_output(
            call_rate,
            column="variant_call_rate",
            as_dataframe=as_dataframe,
        )

    def sample_call_rate(self, as_dataframe: bool = False) -> Any:
        """
        Compute the fraction of variants with non-missing genotype calls per sample.

        Missing calls are represented as negative values or NaN. For 3D genotype
        arrays, a genotype is counted as called only when all allele entries are
        non-missing.
        """
        called = self._called_genotype_mask()
        if called.shape[0] == 0:
            call_rate = np.full(called.shape[1], np.nan, dtype=float)
        else:
            call_rate = called.mean(axis=0)
        return self._format_call_rate_output(
            call_rate,
            column="sample_call_rate",
            as_dataframe=as_dataframe,
        )

    def duplicate_sample_ids(
        self,
        keep: Union[str, bool] = "first",
        ignore_missing: bool = True,
        as_dataframe: bool = False,
    ) -> Any:
        """
        Identify duplicate sample IDs.

        Args:
            keep ({"first", "last", False}, default="first"):
                Which duplicate entry to keep unmarked. ``"first"`` marks all
                but the first entry in each duplicate group, ``"last"`` marks
                all but the last, and False marks all members of duplicate
                groups.
            ignore_missing (bool, default=True):
                If True, empty strings and ``"."`` are not considered duplicate
                IDs.
            as_dataframe (bool, default=False):
                If True, return a report with sample indexes, IDs, duplicate
                flags, and duplicate group IDs.

        Returns:
            Boolean duplicate mask over samples, or a pandas DataFrame if
            ``as_dataframe=True``.
        """
        if self.samples is None:
            raise ValueError("Sample IDs `samples` are required for duplicate sample ID QC.")

        samples = np.asarray(self.samples, dtype=object).ravel()
        duplicate_mask, duplicate_group = self._duplicate_mask_for_values(
            samples,
            keep=keep,
            ignore_missing=ignore_missing,
        )
        if not as_dataframe:
            return duplicate_mask

        import pandas as pd

        return pd.DataFrame(
            {
                "sample_index": np.arange(samples.shape[0], dtype=int),
                "sample": samples,
                "is_duplicate": duplicate_mask,
                "duplicate_group": duplicate_group,
            }
        )

    def duplicate_variant_ids(
        self,
        keep: Union[str, bool] = "first",
        ignore_missing: bool = True,
        as_dataframe: bool = False,
    ) -> Any:
        """
        Identify duplicate variant IDs.

        Args:
            keep ({"first", "last", False}, default="first"):
                Which duplicate entry to keep unmarked. ``"first"`` marks all
                but the first entry in each duplicate group, ``"last"`` marks
                all but the last, and False marks all members of duplicate
                groups.
            ignore_missing (bool, default=True):
                If True, empty strings and ``"."`` are not considered duplicate
                IDs.
            as_dataframe (bool, default=False):
                If True, return a report with variant indexes, IDs, duplicate
                flags, and duplicate group IDs.

        Returns:
            Boolean duplicate mask over variants, or a pandas DataFrame if
            ``as_dataframe=True``.
        """
        if self.variants_id is None:
            raise ValueError("Variant IDs `variants_id` are required for duplicate variant ID QC.")

        variant_ids = np.asarray(self.variants_id, dtype=object).ravel()
        n_snps = self.n_snps
        if variant_ids.shape[0] != n_snps:
            raise ValueError(
                f"'variants_id' must have length equal to the number of SNPs ({n_snps}); "
                f"got {variant_ids.shape[0]}."
            )
        duplicate_mask, duplicate_group = self._duplicate_mask_for_values(
            variant_ids,
            keep=keep,
            ignore_missing=ignore_missing,
        )
        if not as_dataframe:
            return duplicate_mask

        import pandas as pd

        return pd.DataFrame(
            {
                "variant_index": np.arange(variant_ids.shape[0], dtype=int),
                "variant_id": variant_ids,
                "is_duplicate": duplicate_mask,
                "duplicate_group": duplicate_group,
            }
        )

    def duplicate_variant_coordinates(
        self,
        fields: Union[str, Sequence[str]] = ("chrom", "pos", "ref", "alt"),
        keep: Union[str, bool] = "first",
        ignore_missing: bool = True,
        as_dataframe: bool = False,
    ) -> Any:
        """
        Identify duplicate variants by metadata coordinates.

        By default, variants are compared by chromosome, position, reference
        allele, and alternate allele. Pass a narrower ``fields`` value such as
        ``("chrom", "pos")`` to flag variants sharing only a genomic position.

        Args:
            fields (str or sequence of str, default=("chrom", "pos", "ref", "alt")):
                Variant metadata fields used as the duplicate key. Supported
                values are ``"chrom"``, ``"pos"``, ``"ref"``, ``"alt"``, and
                ``"id"``.
            keep ({"first", "last", False}, default="first"):
                Which duplicate entry to keep unmarked. ``"first"`` marks all
                but the first entry in each duplicate group, ``"last"`` marks
                all but the last, and False marks all members of duplicate
                groups.
            ignore_missing (bool, default=True):
                If True, rows with missing key fields are not considered
                duplicates.
            as_dataframe (bool, default=False):
                If True, return a report with variant indexes, key fields,
                duplicate flags, and duplicate group IDs.

        Returns:
            Boolean duplicate mask over variants, or a pandas DataFrame if
            ``as_dataframe=True``.
        """
        normalized_fields, arrays = self._variant_coordinate_arrays(fields)
        duplicate_mask, duplicate_group = self._duplicate_mask_for_rows(
            arrays,
            keep=keep,
            ignore_missing=ignore_missing,
        )
        if not as_dataframe:
            return duplicate_mask

        import pandas as pd

        data: Dict[str, Any] = {
            "variant_index": np.arange(arrays[0].shape[0], dtype=int)
        }
        for (_, label), array in zip(normalized_fields, arrays):
            data[label] = array
        data["is_duplicate"] = duplicate_mask
        data["duplicate_group"] = duplicate_group
        return pd.DataFrame(data)

    def differential_missingness(
        self,
        groups: Any,
        samples: Optional[Union[str, Sequence[str], np.ndarray]] = None,
        test: str = "auto",
        min_expected: float = 5.0,
        group_column: Optional[str] = None,
        as_dataframe: bool = False,
    ) -> Any:
        """
        Test whether genotype missingness differs across sample groups per variant.

        Missingness is computed from genotype calls or BGEN genotype
        probabilities. For 3D genotype arrays, a sample is counted as called
        only when all allele entries are non-missing. ``groups`` may be a
        sequence aligned to samples, a mapping keyed by sample ID, a pandas
        Series/DataFrame, or a snputils phenotype/covariate-style object with
        ``samples`` and ``values`` attributes.

        Args:
            groups:
                Group labels, such as case/control status or genotyping batch.
            samples (str or array_like, optional):
                Optional sample IDs, sample indexes, or boolean mask selecting
                samples used for the test.
            test (str, default="auto"):
                Statistical test. ``"chi2"`` uses the chi-square test of
                independence for 2xK tables. ``"fisher"`` uses Fisher's exact
                test and requires exactly two groups. ``"auto"`` uses chi-square
                except for two-group variants with any expected cell below
                ``min_expected``, where it uses Fisher's exact test.
            min_expected (float, default=5.0):
                Expected cell-count cutoff used by ``test="auto"``.
            group_column (str, optional):
                Column name to use when ``groups`` contains multiple columns.
            as_dataframe (bool, default=False):
                If True, return p-values, test names, statistics, and per-group
                missing/called counts as a pandas DataFrame.

        Returns:
            NumPy array of p-values, or a DataFrame if ``as_dataframe=True``.
        """
        test = self._validate_differential_missingness_test(test)
        min_expected = self._validate_nonnegative_parameter("min_expected", min_expected)
        sample_indexes = self._sample_subset_indices(samples)
        _, group_names, group_ids = self._group_labels_for_samples(
            groups,
            sample_indexes,
            group_column=group_column,
        )
        n_groups = group_names.shape[0]
        if test == "fisher" and n_groups != 2:
            raise ValueError("Fisher exact differential missingness requires exactly two groups.")

        called = self._called_mask_for_samples(
            sample_indexes,
            context="Differential missingness",
        )
        if called.shape[1] != group_ids.shape[0]:
            raise ValueError("Internal error: group labels and selected samples are misaligned.")

        group_one_hot = np.zeros((group_ids.shape[0], n_groups), dtype=np.int64)
        group_one_hot[np.arange(group_ids.shape[0]), group_ids] = 1
        missing_counts = (~called).astype(np.int64, copy=False) @ group_one_hot
        called_counts = called.astype(np.int64, copy=False) @ group_one_hot

        chi2_p, chi2_stat, expected_min = self._chi2_missingness_stats(
            missing_counts,
            called_counts,
        )
        method_names = np.full(called.shape[0], "chi2", dtype=object)

        if test == "chi2" or (test == "auto" and n_groups > 2):
            p_values = chi2_p
            statistics = chi2_stat
        elif test == "fisher":
            p_values, statistics = self._fisher_missingness_stats(
                missing_counts,
                called_counts,
            )
            method_names[:] = "fisher"
        else:
            p_values = chi2_p.copy()
            statistics = chi2_stat.copy()
            row_missing = missing_counts.sum(axis=1)
            row_called = called_counts.sum(axis=1)
            use_fisher = (
                np.isfinite(expected_min)
                & (expected_min < min_expected)
                & (row_missing > 0)
                & (row_called > 0)
            )
            if np.any(use_fisher):
                fisher_p, fisher_stat = self._fisher_missingness_stats(
                    missing_counts,
                    called_counts,
                )
                p_values[use_fisher] = fisher_p[use_fisher]
                statistics[use_fisher] = fisher_stat[use_fisher]
                method_names[use_fisher] = "fisher"

        return self._format_differential_missingness_output(
            p_values,
            statistics,
            method_names,
            missing_counts,
            called_counts,
            group_names,
            as_dataframe=as_dataframe,
        )

    def relatedness(
        self,
        method: str = "grm",
        samples: Optional[Union[str, Sequence[str], np.ndarray]] = None,
        scale: str = "relationship",
        min_variants: int = 1,
        block_size: int = 10000,
        ibdobj: Any = None,
        genome_length_cm: float = 3400.0,
        min_segment_cm: Optional[float] = None,
        segment_types: Optional[Sequence[str]] = None,
        as_dataframe: bool = False,
    ) -> Any:
        """
        Compute sample relatedness.

        ``method="grm"`` computes genotype-derived relatedness from diploid
        biallelic dosages. Genotypes are standardized per variant as
        ``(g - 2p) / sqrt(2p(1-p))`` and pairwise products are averaged over
        variants where both samples are called. ``method="ibd"`` summarizes
        already-called IBD segments from an ``IBDObject`` as
        ``(IBD1 cM + 2 * IBD2 cM) / (2 * genome_length_cm)``.

        Args:
            method (str, default="grm"):
                Relatedness estimator. Supported values are ``"grm"`` and ``"ibd"``.
            samples (str or array_like, optional):
                Sample IDs, sample indexes, or boolean mask selecting samples.
                If None, all samples are used.
            scale (str, default="relationship"):
                ``"relationship"`` returns the relationship coefficient.
                ``"kinship"`` returns half the relationship coefficient.
            min_variants (int, default=1):
                Minimum number of pairwise non-missing informative variants
                required to report a finite GRM value.
            block_size (int, default=10000):
                Number of variants per block used while accumulating the GRM.
            ibdobj (IBDObject, optional):
                IBD segments used when ``method="ibd"``.
            genome_length_cm (float, default=3400.0):
                Diploid genome length denominator for IBD normalization.
            min_segment_cm (float, optional):
                Minimum IBD segment length to include.
            segment_types (sequence of str, optional):
                IBD segment types to include, such as ``["IBD1", "IBD2"]``.
            as_dataframe (bool, default=False):
                If True, return a pandas DataFrame with sample labels.

        Returns:
            Square NumPy array of relatedness values, or a DataFrame if
            ``as_dataframe=True``.
        """
        method = self._validate_relatedness_method(method)
        scale = self._validate_relatedness_scale(scale)
        if method == "grm":
            sample_indexes, relationship, _ = self._grm_relatedness_components(
                samples=samples,
                min_variants=min_variants,
                block_size=block_size,
            )
        else:
            sample_indexes, relationship, _ = self._ibd_relatedness_components(
                ibdobj=ibdobj,
                samples=samples,
                genome_length_cm=genome_length_cm,
                min_segment_cm=min_segment_cm,
                segment_types=segment_types,
            )
        values = relationship if scale == "relationship" else relationship / 2.0
        return self._format_relatedness_output(
            values,
            sample_indexes=sample_indexes,
            as_dataframe=as_dataframe,
        )

    def flag_related_pairs(
        self,
        threshold: float = 0.0884,
        samples: Optional[Union[str, Sequence[str], np.ndarray]] = None,
        min_variants: int = 1,
        block_size: int = 10000,
        method: str = "grm",
        ibdobj: Any = None,
        genome_length_cm: float = 3400.0,
        min_segment_cm: Optional[float] = None,
        segment_types: Optional[Sequence[str]] = None,
    ) -> Any:
        """
        Return sample pairs with estimated kinship at or above ``threshold``.

        The default threshold, 0.0884, is a commonly used second-degree kinship
        cutoff. For ``method="grm"``, input variants should already be
        appropriate for relatedness QC (autosomal, reasonably common, and
        preferably LD-pruned). For ``method="ibd"``, pass an ``IBDObject`` via
        ``ibdobj``.
        """
        threshold = self._validate_nonnegative_parameter("threshold", threshold)
        method = self._validate_relatedness_method(method)
        if method == "grm":
            sample_indexes, relationship, counts = self._grm_relatedness_components(
                samples=samples,
                min_variants=min_variants,
                block_size=block_size,
            )
            extra_metrics = {"n_variants": counts}
        else:
            sample_indexes, relationship, extra_metrics = self._ibd_relatedness_components(
                ibdobj=ibdobj,
                samples=samples,
                genome_length_cm=genome_length_cm,
                min_segment_cm=min_segment_cm,
                segment_types=segment_types,
            )
        kinship = relationship / 2.0

        columns = ["sample_index_1", "sample_index_2"]
        include_names = self.samples is not None
        if include_names:
            columns.extend(["sample_1", "sample_2"])
            sample_names = np.asarray(self.samples)
        metric_columns = (
            ["n_variants"]
            if method == "grm"
            else ["n_segments", "ibd1_cm", "ibd2_cm", "weighted_ibd_cm"]
        )
        columns.extend(["relationship", "kinship", *metric_columns])

        records = []
        n_samples = sample_indexes.size
        for first in range(n_samples):
            for second in range(first + 1, n_samples):
                value = kinship[first, second]
                if not np.isfinite(value) or value < threshold:
                    continue
                record = {
                    "sample_index_1": int(sample_indexes[first]),
                    "sample_index_2": int(sample_indexes[second]),
                    "relationship": float(relationship[first, second]),
                    "kinship": float(value),
                }
                if include_names:
                    record["sample_1"] = sample_names[sample_indexes[first]]
                    record["sample_2"] = sample_names[sample_indexes[second]]
                for column in metric_columns:
                    metric_value = extra_metrics[column][first, second]
                    if column in {"n_variants", "n_segments"}:
                        metric_value = int(metric_value)
                    else:
                        metric_value = float(metric_value)
                    record[column] = metric_value
                records.append(record)

        import pandas as pd

        return pd.DataFrame.from_records(records, columns=columns)

    def prune_related_samples(
        self,
        threshold: float = 0.0884,
        samples: Optional[Union[str, Sequence[str], np.ndarray]] = None,
        inplace: bool = False,
        method: str = "grm",
        min_variants: int = 1,
        block_size: int = 10000,
        ibdobj: Any = None,
        genome_length_cm: float = 3400.0,
        min_segment_cm: Optional[float] = None,
        segment_types: Optional[Sequence[str]] = None,
    ) -> Optional['SNPObject']:
        """
        Greedily remove samples until no selected pair exceeds a kinship threshold.

        At each step, the removed sample is chosen by: more flagged
        relationships, then lower sample call rate, then later sample index.
        Samples outside the optional ``samples`` subset are kept.
        """
        threshold = self._validate_nonnegative_parameter("threshold", threshold)
        method = self._validate_relatedness_method(method)
        if method == "grm":
            sample_indexes, relationship, _ = self._grm_relatedness_components(
                samples=samples,
                min_variants=min_variants,
                block_size=block_size,
            )
        else:
            sample_indexes, relationship, _ = self._ibd_relatedness_components(
                ibdobj=ibdobj,
                samples=samples,
                genome_length_cm=genome_length_cm,
                min_segment_cm=min_segment_cm,
                segment_types=segment_types,
            )
        kinship = relationship / 2.0
        keep = np.ones(sample_indexes.size, dtype=bool)
        call_rate = self._sample_call_rate_for_pruning(sample_indexes)

        while np.count_nonzero(keep) > 1:
            active = np.flatnonzero(keep)
            active_kinship = kinship[np.ix_(active, active)]
            related = np.triu(np.isfinite(active_kinship) & (active_kinship >= threshold), k=1)
            if not np.any(related):
                break

            pair_positions = np.argwhere(related)
            first_positions = active[pair_positions[:, 0]]
            second_positions = active[pair_positions[:, 1]]
            degrees = np.zeros(sample_indexes.size, dtype=int)
            np.add.at(degrees, first_positions, 1)
            np.add.at(degrees, second_positions, 1)
            candidates = np.unique(np.concatenate([first_positions, second_positions]))

            def removal_key(position: int) -> Tuple[int, float, int]:
                rate = call_rate[position]
                rate_key = float(rate) if np.isfinite(rate) else -np.inf
                return degrees[position], -rate_key, int(sample_indexes[position])

            remove_position = max(candidates, key=removal_key)
            keep[remove_position] = False

        removed_indexes = sample_indexes[~keep]
        return self.filter_samples(indexes=removed_indexes, include=False, inplace=inplace)

    def imputation_r2(
        self,
        source: str = "auto",
        info_keys: Optional[Union[str, Sequence[str]]] = None,
        as_dataframe: bool = False,
    ) -> Any:
        """
        Compute per-variant imputation quality R2/INFO values.

        ``source="info"`` extracts scalar quality values from ``variants_info``
        using common keys such as ``INFO``, ``R2``, ``DR2``, and ``IMP``.
        ``source="gp"`` estimates an INFO-like value from genotype
        probabilities as ``1 - mean(posterior variance) / expected genotype
        variance``. ``source="dosage"`` computes empirical dosage R2 as
        ``Var(DS) / expected genotype variance``. ``source="auto"`` uses INFO
        values when available and fills missing values from genotype
        probabilities, then dosages.

        Args:
            source (str, default="auto"):
                One of ``"auto"``, ``"info"``, ``"gp"``, or ``"dosage"``.
                Aliases such as ``"bgen"`` and ``"ds"`` are accepted.
            info_keys (str or sequence of str, optional):
                INFO field keys to search, in priority order. Defaults to common
                imputation quality keys.
            as_dataframe (bool, default=False):
                If True, return a pandas DataFrame.

        Returns:
            NumPy array of imputation quality values, or a DataFrame if
            ``as_dataframe=True``. Values unavailable from the selected source
            are returned as NaN.
        """
        source = self._validate_imputation_r2_source(source)

        if source == "info":
            values = self._imputation_r2_from_info(info_keys=info_keys)
        elif source == "gp":
            values = self._imputation_r2_from_gp()
        elif source == "dosage":
            values = self._imputation_r2_from_dosage()
        else:
            values = np.full(self.n_snps, np.nan, dtype=float)

            info_values = self._imputation_r2_from_info(info_keys=info_keys)
            info_mask = np.isfinite(info_values)
            values[info_mask] = info_values[info_mask]

            missing = ~np.isfinite(values)
            if np.any(missing) and self.calldata_gp is not None:
                gp_values = self._imputation_r2_from_gp()
                gp_mask = missing & np.isfinite(gp_values)
                values[gp_mask] = gp_values[gp_mask]

            missing = ~np.isfinite(values)
            if np.any(missing) and self.genotypes is not None:
                dosage_values = self._imputation_r2_from_dosage()
                dosage_mask = missing & np.isfinite(dosage_values)
                values[dosage_mask] = dosage_values[dosage_mask]

        return self._format_call_rate_output(
            np.asarray(values, dtype=float),
            column="imputation_r2",
            as_dataframe=as_dataframe,
        )

    def sample_heterozygosity(
        self,
        samples: Optional[Union[str, Sequence[str], np.ndarray]] = None,
        as_dataframe: bool = False,
    ) -> Any:
        """
        Compute observed heterozygosity per sample.

        The value is the fraction of called diploid biallelic genotypes that are
        heterozygous. Missing calls are ignored. This QC statistic is typically
        most useful after restricting to autosomal, reasonably common, LD-pruned
        variants.

        Args:
            samples (str or array_like, optional):
                Sample IDs, sample indexes, or boolean mask selecting samples.
                If None, all samples are used.
            as_dataframe (bool, default=False):
                If True, return a pandas DataFrame.

        Returns:
            NumPy array of per-sample heterozygosity, or a DataFrame if
            ``as_dataframe=True``.
        """
        sample_indexes, _, _, _, _, heterozygosity = self._sample_heterozygosity_components(
            samples=samples,
        )
        return self._format_sample_stat_output(
            heterozygosity,
            sample_indexes=sample_indexes,
            column="heterozygosity",
            as_dataframe=as_dataframe,
        )

    def sample_inbreeding_coefficient(
        self,
        samples: Optional[Union[str, Sequence[str], np.ndarray]] = None,
        as_dataframe: bool = False,
    ) -> Any:
        """
        Compute per-sample inbreeding coefficients from observed and expected heterozygosity.

        The coefficient is ``1 - observed_hets / expected_hets``, where expected
        heterozygosity is summed from cohort allele frequencies estimated across
        the selected samples. Missing calls are ignored per sample. This statistic
        is typically most useful after restricting to autosomal, reasonably
        common, LD-pruned variants.

        Args:
            samples (str or array_like, optional):
                Sample IDs, sample indexes, or boolean mask selecting samples.
                If None, all samples are used.
            as_dataframe (bool, default=False):
                If True, return a pandas DataFrame.

        Returns:
            NumPy array of per-sample inbreeding coefficients, or a DataFrame if
            ``as_dataframe=True``.
        """
        sample_indexes, inbreeding, _, _, _ = self._sample_inbreeding_components(
            samples=samples,
        )
        return self._format_sample_stat_output(
            inbreeding,
            sample_indexes=sample_indexes,
            column="inbreeding_coefficient",
            as_dataframe=as_dataframe,
        )

    def flag_heterozygosity_outliers(
        self,
        n_sd: float = 3,
        samples: Optional[Union[str, Sequence[str], np.ndarray]] = None,
    ) -> Any:
        """
        Build a per-sample heterozygosity outlier report.

        Samples are flagged when their observed heterozygosity is more than
        ``n_sd`` standard deviations from the mean among finite selected samples.
        This is best run on autosomal, reasonably common, LD-pruned variants.

        Args:
            n_sd (float, default=3):
                Number of standard deviations from the mean used for flagging.
                Must be non-negative.
            samples (str or array_like, optional):
                Sample IDs, sample indexes, or boolean mask selecting samples.
                If None, all samples are used.

        Returns:
            pandas.DataFrame:
                Sample-level QC report with heterozygosity, inbreeding coefficient,
                z-score, and outlier flag.
        """
        threshold = self._validate_nonnegative_parameter("n_sd", n_sd)
        sample_indexes, _, _, observed_hets, called_counts, heterozygosity = (
            self._sample_heterozygosity_components(samples=samples)
        )
        _, inbreeding, expected_hets, _, _ = self._sample_inbreeding_components(
            samples=sample_indexes,
        )

        finite = np.isfinite(heterozygosity)
        z_scores = np.full(heterozygosity.shape, np.nan, dtype=float)
        outliers = np.zeros(heterozygosity.shape, dtype=bool)
        if np.count_nonzero(finite) > 0:
            mean = heterozygosity[finite].mean()
            sd = heterozygosity[finite].std(ddof=0)
            if sd > 0:
                z_scores[finite] = (heterozygosity[finite] - mean) / sd
                outliers[finite] = np.abs(z_scores[finite]) > threshold
            else:
                z_scores[finite] = 0.0

        import pandas as pd

        data = {
            "sample_index": np.asarray(sample_indexes, dtype=int),
            "called_genotypes": called_counts.astype(int),
            "observed_heterozygotes": observed_hets.astype(int),
            "expected_heterozygotes": expected_hets,
            "heterozygosity": heterozygosity,
            "inbreeding_coefficient": inbreeding,
            "heterozygosity_z": z_scores,
            "heterozygosity_outlier": outliers,
        }
        if self.samples is not None:
            data = {
                "sample_index": data["sample_index"],
                "sample": np.asarray(self.samples)[sample_indexes],
                **{key: value for key, value in data.items() if key != "sample_index"},
            }
        return pd.DataFrame(data)

    def hwe_pvalue(
        self,
        samples: Optional[Union[str, Sequence[str], np.ndarray]] = None,
        as_dataframe: bool = False,
    ) -> Any:
        """
        Compute exact Hardy-Weinberg equilibrium p-values per variant.

        HWE is computed from hard-called diploid biallelic genotypes only.
        Missing calls are ignored. The optional ``samples`` argument can be used
        to restrict the calculation to a control-only subset, using sample IDs,
        sample indexes, or a boolean sample mask.

        Args:
            samples (str or array_like, optional):
                Sample IDs, sample indexes, or boolean mask selecting the samples
                used for the HWE test. If None, all samples are used.
            as_dataframe (bool, default=False):
                If True, return a pandas DataFrame.

        Returns:
            NumPy array of HWE p-values, or a DataFrame if ``as_dataframe=True``.
            Variants with no called genotypes in the selected samples return NaN.
        """
        dosages, called = self._hard_call_dosages(samples=samples)
        p_values = np.full(dosages.shape[0], np.nan, dtype=float)

        for variant_idx in range(dosages.shape[0]):
            observed = dosages[variant_idx, called[variant_idx]]
            if observed.size == 0:
                continue

            obs_hom_ref = int(np.count_nonzero(observed == 0))
            obs_hets = int(np.count_nonzero(observed == 1))
            obs_hom_alt = int(np.count_nonzero(observed == 2))
            p_values[variant_idx] = self._hwe_exact_pvalue(
                obs_hets=obs_hets,
                obs_hom_ref=obs_hom_ref,
                obs_hom_alt=obs_hom_alt,
            )

        return self._format_call_rate_output(
            p_values,
            column="hwe_pvalue",
            as_dataframe=as_dataframe,
        )

    def ld_prune_mask(
        self,
        window_size: int = 50,
        step_size: int = 5,
        r2_threshold: float = 0.2,
        samples: Optional[Union[str, Sequence[str], np.ndarray]] = None,
        min_samples: int = 3,
    ) -> np.ndarray:
        """
        Build a greedy LD-pruning mask using sliding windows and pairwise r-squared.

        Variants are processed in their current order. Within each window, the
        earlier variant is kept and later variants with pairwise r-squared greater
        than ``r2_threshold`` are removed. Windows advance by ``step_size`` and
        previously removed variants stay removed.

        Args:
            window_size (int, default=50):
                Number of variants per sliding window. Must be at least 2.
            step_size (int, default=5):
                Number of variants to advance the window each step. Must be at least 1.
            r2_threshold (float, default=0.2):
                Pairwise LD r-squared threshold. Must be between 0 and 1.
            samples (str or array_like, optional):
                Sample IDs, sample indexes, or boolean mask selecting samples used
                to estimate LD. If None, all samples are used.
            min_samples (int, default=3):
                Minimum number of pairwise non-missing samples needed to compute r-squared.

        Returns:
            np.ndarray:
                Boolean mask aligned to variants, where True indicates retained variants.
        """
        window_size = self._validate_integer_parameter("window_size", window_size, 2)
        step_size = self._validate_integer_parameter("step_size", step_size, 1)
        min_samples = self._validate_integer_parameter("min_samples", min_samples, 2)
        threshold = self._validate_probability_threshold("r2_threshold", r2_threshold)

        dosages = self._ld_dosages(samples=samples)
        n_snps = dosages.shape[0]
        keep = np.ones(n_snps, dtype=bool)

        for start in range(0, n_snps, step_size):
            stop = min(start + window_size, n_snps)
            if stop - start < 2:
                continue

            window_indexes = np.arange(start, stop)
            for offset, first_idx in enumerate(window_indexes[:-1]):
                if not keep[first_idx]:
                    continue
                for second_idx in window_indexes[offset + 1:]:
                    if not keep[second_idx]:
                        continue
                    r2 = self._pairwise_r2_ignore_missing(
                        dosages[first_idx],
                        dosages[second_idx],
                        min_samples=min_samples,
                    )
                    if np.isfinite(r2) and r2 > threshold:
                        keep[second_idx] = False

        return keep

    def filter_variants(
            self,
            chrom: Optional[Union[str, Sequence[str], np.ndarray, None]] = None,
            pos: Optional[Union[int, Sequence[int], np.ndarray, None]] = None,
            indexes: Optional[Union[int, Sequence[int], np.ndarray, None]] = None,
            mask: Optional[Union[Sequence[bool], np.ndarray, None]] = None,
            include: bool = True,
            inplace: bool = False
        ) -> Optional['SNPObject']:
        """
        Filter variants based on chromosome names, variant positions, indexes, or a boolean mask.

        This method updates the `genotypes`, `variants_ref`, `variants_alt`,
        `variants_chrom`, `variants_filter_pass`, `variants_id`, `variants_pos`,
        `variants_qual`, and `lai` attributes to include or exclude the specified variants. The filtering
        criteria can be based on chromosome names, variant positions, or indexes. If multiple
        criteria are provided, their union is used for filtering. The order of the variants is preserved.

        Negative indexes are supported and follow
        [NumPy's indexing conventions](https://numpy.org/doc/stable/user/basics.indexing.html).

        Args:
            chrom (str or array_like of str, optional):
                Chromosome(s) to filter variants by. Can be a single chromosome as a string or a sequence
                of chromosomes. If both `chrom` and `pos` are provided, they must either have matching lengths
                (pairing each chromosome with a position) or `chrom` should be a single value that applies to
                all positions in `pos`. Default is None.
            pos (int or array_like of int, optional):
                Position(s) to filter variants by. Can be a single position as an integer or a sequence of positions.
                If `chrom` is also provided, `pos` should either match `chrom` in length or `chrom` should be a
                single value. Default is None.
            indexes (int or array_like of int, optional):
                Index(es) of the variants to include or exclude. Can be a single index or a sequence
                of indexes. Negative indexes are supported. Default is None.
            mask (array_like of bool, optional):
                Boolean mask aligned to the SNP axis. If provided with other criteria, the union of all
                selected variants is used before applying ``include``.
            include (bool, default=True):
                If True, includes only the specified variants. If False, excludes the specified
                variants. Default is True.
            inplace (bool, default=False):
                If True, modifies `self` in place. If False, returns a new `SNPObject` with the variants
                filtered. Default is False.

        Returns:
            Optional[SNPObject]:
                A new `SNPObject` with the specified variants filtered if `inplace=False`.
                If `inplace=True`, modifies `self` in place and returns None.
        """
        if chrom is None and pos is None and indexes is None and mask is None:
            raise ValueError("At least one of 'chrom', 'pos', 'indexes', or 'mask' must be provided.")

        n_snps = self.n_snps

        # Convert inputs to arrays for consistency
        chrom = np.atleast_1d(chrom) if chrom is not None else None
        pos = np.atleast_1d(pos) if pos is not None else None
        indexes = np.atleast_1d(indexes) if indexes is not None else None

        # Validate chrom and pos lengths if both are provided
        if chrom is not None and pos is not None:
            if len(chrom) != len(pos) and len(chrom) > 1:
                raise ValueError(
                    "When both 'chrom' and 'pos' are provided, they must either be of the same length "
                    "or 'chrom' must be a single value."
                )

        # Create a mask for chromosome and position filtering
        mask_combined = np.zeros(n_snps, dtype=bool)
        if chrom is not None and pos is not None:
            if len(chrom) == 1:
                # Apply single chromosome to all positions in `pos`
                mask_combined = (self['variants_chrom'] == chrom[0]) & np.isin(self['variants_pos'], pos)
            else:
                # Vectorized pair matching for chrom and pos
                query_pairs = np.array(
                    list(zip(chrom, pos)),
                    dtype=[
                        ('chrom', self['variants_chrom'].dtype),
                        ('pos', self['variants_pos'].dtype)
                    ]
                )
                data_pairs = np.array(
                    list(zip(self['variants_chrom'], self['variants_pos'])),
                    dtype=[
                        ('chrom', self['variants_chrom'].dtype),
                        ('pos', self['variants_pos'].dtype)
                    ]
                )
                mask_combined = np.isin(data_pairs, query_pairs)

        elif chrom is not None:
            # Only chromosome filtering
            mask_combined = np.isin(self['variants_chrom'], chrom)
        elif pos is not None:
            # Only position filtering
            mask_combined = np.isin(self['variants_pos'], pos)

        # Create mask based on indexes if provided
        if indexes is not None:
            # Validate indexes, allowing negative indexes
            out_of_bounds_indexes = indexes[(indexes < -n_snps) | (indexes >= n_snps)]
            if out_of_bounds_indexes.size > 0:
                raise ValueError(f"One or more sample indexes are out of bounds.")

            # Handle negative indexes and check for out-of-bounds indexes
            adjusted_indexes = np.mod(indexes, n_snps)

            # Create mask for specified indexes
            mask_indexes = np.zeros(n_snps, dtype=bool)
            mask_indexes[adjusted_indexes] = True

            # Combine with `chrom` and `pos` mask using logical OR (union of all specified criteria)
            mask_combined = mask_combined | mask_indexes

        if mask is not None:
            mask_array = np.asarray(mask, dtype=bool).ravel()
            if mask_array.shape[0] != n_snps:
                raise ValueError(
                    f"'mask' must have length equal to the number of SNPs ({n_snps}); "
                    f"got {mask_array.shape[0]}."
                )
            mask_combined = mask_combined | mask_array

        # Invert mask if `include` is False
        if not include:
            mask_combined = ~mask_combined

        # Define keys to filter
        keys = [
            'genotypes', 'variants_ref', 'variants_alt', 'variants_chrom', 'variants_cm', 'variants_filter_pass',
            'variants_id', 'variants_pos', 'variants_qual', 'variants_info', 'calldata_lai', 'calldata_gp'
        ]

        # Apply filtering based on inplace parameter
        def _apply_mask(snpobj: SNPObject) -> None:
            for key in keys:
                val = snpobj[key]
                if val is None:
                    continue
                arr = np.asarray(val)
                if arr.shape[0] != n_snps:
                    # VCF (and similar) readers use length-0 arrays as placeholders when a column
                    # was not loaded; those are not aligned to the SNP axis.
                    if arr.shape[0] == 0 and n_snps > 0:
                        snpobj[key] = None
                        continue
                    raise ValueError(
                        f"Cannot filter '{key}': length along SNP axis is {arr.shape[0]}, expected {n_snps}."
                    )
                if arr.ndim > 1:
                    snpobj[key] = arr[mask_combined, ...]
                else:
                    snpobj[key] = arr[mask_combined]

        if inplace:
            _apply_mask(self)
            return None

        snpobj = self.copy()
        _apply_mask(snpobj)
        return snpobj

    def filter_biallelic_variants(
            self,
            snv_only: bool = True,
            inplace: bool = False,
        ) -> Optional['SNPObject']:
        """
        Keep variants with exactly one alternate allele.

        Args:
            snv_only (bool, default=True):
                If True, also require REF and ALT to be single-base alleles.
            inplace (bool, default=False):
                If True, modifies ``self`` in place. If False, returns a filtered copy.

        Returns:
            Optional[SNPObject]:
                A filtered SNPObject if ``inplace=False``; otherwise modifies ``self`` and returns None.
        """
        if self.variants_ref is None or self.variants_alt is None:
            raise ValueError("'variants_ref' and 'variants_alt' are required to filter biallelic variants.")

        ref = np.asarray(self.variants_ref).astype(str)
        alt = np.asarray(self.variants_alt).astype(str)
        mask = (
            (ref != "")
            & (alt != "")
            & (ref != ".")
            & (alt != ".")
            & (np.char.find(alt, ",") < 0)
        )
        if snv_only:
            mask = mask & (np.char.str_len(ref) == 1) & (np.char.str_len(alt) == 1)
        return self.filter_variants(mask=mask, include=True, inplace=inplace)

    def filter_complete_genotypes(self, inplace: bool = False) -> Optional['SNPObject']:
        """
        Keep variants with no missing genotype calls across all samples.

        Missing calls are represented as negative values in ``genotypes``.
        """
        if self.genotypes is None:
            raise ValueError("Genotype data `genotypes` is None.")
        gt = np.asarray(self.genotypes)
        if gt.ndim == 2:
            mask = np.all(gt >= 0, axis=1)
        elif gt.ndim == 3:
            mask = np.all(gt >= 0, axis=(1, 2))
        else:
            raise ValueError("'genotypes' must be a 2D or 3D array.")
        return self.filter_variants(mask=mask, include=True, inplace=inplace)

    def filter_variable_genotypes(self, inplace: bool = False) -> Optional['SNPObject']:
        """
        Keep variants with at least two observed genotype values among called samples.

        This filters on genotype variability, not allele polymorphism. For example,
        a site where every called sample is heterozygous has one observed genotype
        value and is therefore removed. Phase order is ignored when comparing
        phased allele calls.
        """
        if self.genotypes is None:
            raise ValueError("Genotype data `genotypes` is None.")
        gt = np.asarray(self.genotypes)
        if gt.ndim == 3:
            called = np.all(gt >= 0, axis=2)
        elif gt.ndim == 2:
            called = gt >= 0
        else:
            raise ValueError("'genotypes' must be a 2D or 3D array.")

        mask = np.zeros(gt.shape[0], dtype=bool)
        for i in range(gt.shape[0]):
            observed = gt[i, called[i]]
            if gt.ndim == 3:
                observed = np.sort(observed, axis=1)
                mask[i] = np.unique(observed, axis=0).shape[0] >= 2
            else:
                mask[i] = np.unique(observed).size >= 2
        return self.filter_variants(mask=mask, include=True, inplace=inplace)

    def filter_variants_by_call_rate(
            self,
            min_call_rate: float = 0.98,
            include: bool = True,
            inplace: bool = False,
        ) -> Optional['SNPObject']:
        """
        Filter variants by genotype call rate.

        Args:
            min_call_rate (float, default=0.98):
                Minimum fraction of samples with non-missing genotype calls.
                Must be between 0 and 1.
            include (bool, default=True):
                If True, keeps variants with call rate greater than or equal to
                ``min_call_rate``. If False, excludes those variants.
            inplace (bool, default=False):
                If True, modifies ``self`` in place. If False, returns a filtered copy.

        Returns:
            Optional[SNPObject]:
                A filtered SNPObject if ``inplace=False``; otherwise modifies ``self`` and returns None.
        """
        threshold = self._validate_call_rate_threshold(min_call_rate)
        call_rate = np.asarray(self.variant_call_rate(), dtype=float).ravel()
        mask = np.isfinite(call_rate) & (call_rate >= threshold)
        return self.filter_variants(mask=mask, include=include, inplace=inplace)

    def filter_samples_by_call_rate(
            self,
            min_call_rate: float = 0.98,
            include: bool = True,
            inplace: bool = False,
        ) -> Optional['SNPObject']:
        """
        Filter samples by genotype call rate.

        Args:
            min_call_rate (float, default=0.98):
                Minimum fraction of variants with non-missing genotype calls.
                Must be between 0 and 1.
            include (bool, default=True):
                If True, keeps samples with call rate greater than or equal to
                ``min_call_rate``. If False, excludes those samples.
            inplace (bool, default=False):
                If True, modifies ``self`` in place. If False, returns a filtered copy.

        Returns:
            Optional[SNPObject]:
                A filtered SNPObject if ``inplace=False``; otherwise modifies ``self`` and returns None.
        """
        threshold = self._validate_call_rate_threshold(min_call_rate)
        call_rate = np.asarray(self.sample_call_rate(), dtype=float).ravel()
        mask = np.isfinite(call_rate) & (call_rate >= threshold)
        indexes = np.where(mask)[0]
        return self.filter_samples(indexes=indexes, include=include, inplace=inplace)

    def filter_duplicate_samples(
            self,
            keep: Union[str, bool] = "first",
            ignore_missing: bool = True,
            inplace: bool = False,
        ) -> Optional['SNPObject']:
        """
        Remove duplicate sample IDs.

        Args:
            keep ({"first", "last", False}, default="first"):
                Which sample in each duplicate ID group to keep. False removes
                every sample belonging to a duplicate group.
            ignore_missing (bool, default=True):
                If True, empty strings and ``"."`` are not considered duplicate
                sample IDs.
            inplace (bool, default=False):
                If True, modifies ``self`` in place. If False, returns a
                filtered copy.

        Returns:
            Optional[SNPObject]:
                A filtered SNPObject if ``inplace=False``; otherwise modifies
                ``self`` and returns None.
        """
        duplicate_mask = np.asarray(
            self.duplicate_sample_ids(
                keep=keep,
                ignore_missing=ignore_missing,
            ),
            dtype=bool,
        )
        return self.filter_samples(
            indexes=np.flatnonzero(duplicate_mask),
            include=False,
            inplace=inplace,
        )

    def filter_duplicate_variants(
            self,
            by: str = "id",
            fields: Union[str, Sequence[str]] = ("chrom", "pos", "ref", "alt"),
            keep: Union[str, bool] = "first",
            ignore_missing: bool = True,
            inplace: bool = False,
        ) -> Optional['SNPObject']:
        """
        Remove duplicate variants by ID or coordinate key.

        Args:
            by (str, default="id"):
                Duplicate key to use: ``"id"`` or ``"coordinates"``.
            fields (str or sequence of str, default=("chrom", "pos", "ref", "alt")):
                Coordinate fields used when ``by="coordinates"``.
            keep ({"first", "last", False}, default="first"):
                Which variant in each duplicate group to keep. False removes
                every variant belonging to a duplicate group.
            ignore_missing (bool, default=True):
                If True, rows with missing key fields are not considered
                duplicates.
            inplace (bool, default=False):
                If True, modifies ``self`` in place. If False, returns a
                filtered copy.

        Returns:
            Optional[SNPObject]:
                A filtered SNPObject if ``inplace=False``; otherwise modifies
                ``self`` and returns None.
        """
        by = self._validate_duplicate_variant_by(by)
        if by == "id":
            duplicate_mask = self.duplicate_variant_ids(
                keep=keep,
                ignore_missing=ignore_missing,
            )
        else:
            duplicate_mask = self.duplicate_variant_coordinates(
                fields=fields,
                keep=keep,
                ignore_missing=ignore_missing,
            )
        return self.filter_variants(
            indexes=np.flatnonzero(np.asarray(duplicate_mask, dtype=bool)),
            include=False,
            inplace=inplace,
        )

    def filter_differential_missingness(
            self,
            groups: Any,
            min_p: float = 1e-5,
            samples: Optional[Union[str, Sequence[str], np.ndarray]] = None,
            test: str = "auto",
            min_expected: float = 5.0,
            group_column: Optional[str] = None,
            include: bool = True,
            inplace: bool = False,
        ) -> Optional['SNPObject']:
        """
        Filter variants by differential missingness p-value.

        Args:
            groups:
                Group labels passed to :meth:`differential_missingness`.
            min_p (float, default=1e-5):
                Minimum differential missingness p-value. Must be between 0
                and 1. Variants below this threshold show evidence that
                missingness differs by group.
            samples (str or array_like, optional):
                Optional sample IDs, sample indexes, or boolean mask selecting
                samples used for the test.
            test (str, default="auto"):
                Statistical test passed to :meth:`differential_missingness`.
            min_expected (float, default=5.0):
                Expected cell-count cutoff used by ``test="auto"``.
            group_column (str, optional):
                Column name to use when ``groups`` contains multiple columns.
            include (bool, default=True):
                If True, keeps variants with p-value greater than or equal to
                ``min_p``. If False, excludes those variants.
            inplace (bool, default=False):
                If True, modifies ``self`` in place. If False, returns a
                filtered copy.

        Returns:
            Optional[SNPObject]:
                A filtered SNPObject if ``inplace=False``; otherwise modifies
                ``self`` and returns None.
        """
        threshold = self._validate_probability_threshold("min_p", min_p)
        p_values = np.asarray(
            self.differential_missingness(
                groups,
                samples=samples,
                test=test,
                min_expected=min_expected,
                group_column=group_column,
            ),
            dtype=float,
        ).ravel()
        mask = np.isfinite(p_values) & (p_values >= threshold)
        return self.filter_variants(mask=mask, include=include, inplace=inplace)

    def filter_imputation_quality(
            self,
            min_r2: float = 0.8,
            source: str = "auto",
            info_keys: Optional[Union[str, Sequence[str]]] = None,
            include: bool = True,
            inplace: bool = False,
        ) -> Optional['SNPObject']:
        """
        Filter variants by imputation quality R2/INFO.

        Args:
            min_r2 (float, default=0.8):
                Minimum imputation quality value. Must be between 0 and 1.
            source (str, default="auto"):
                Source used by :meth:`imputation_r2`: ``"auto"``, ``"info"``,
                ``"gp"``, or ``"dosage"``.
            info_keys (str or sequence of str, optional):
                INFO field keys to search when extracting values from
                ``variants_info``.
            include (bool, default=True):
                If True, keeps variants with imputation quality greater than or
                equal to ``min_r2``. If False, excludes those variants.
            inplace (bool, default=False):
                If True, modifies ``self`` in place. If False, returns a filtered copy.

        Returns:
            Optional[SNPObject]:
                A filtered SNPObject if ``inplace=False``; otherwise modifies ``self`` and returns None.
        """
        threshold = self._validate_probability_threshold("min_r2", min_r2)
        r2 = np.asarray(
            self.imputation_r2(source=source, info_keys=info_keys),
            dtype=float,
        ).ravel()
        mask = np.isfinite(r2) & (r2 >= threshold)
        return self.filter_variants(mask=mask, include=include, inplace=inplace)

    def filter_hwe(
            self,
            min_p: float = 1e-6,
            samples: Optional[Union[str, Sequence[str], np.ndarray]] = None,
            include: bool = True,
            inplace: bool = False,
        ) -> Optional['SNPObject']:
        """
        Filter variants by exact Hardy-Weinberg equilibrium p-value.

        Args:
            min_p (float, default=1e-6):
                Minimum HWE p-value. Must be between 0 and 1.
            samples (str or array_like, optional):
                Sample IDs, sample indexes, or boolean mask selecting the samples
                used for the HWE test. This is intended for control-only HWE QC
                in case-control GWAS.
            include (bool, default=True):
                If True, keeps variants with HWE p-value greater than or equal to
                ``min_p``. If False, excludes those variants.
            inplace (bool, default=False):
                If True, modifies ``self`` in place. If False, returns a filtered copy.

        Returns:
            Optional[SNPObject]:
                A filtered SNPObject if ``inplace=False``; otherwise modifies ``self`` and returns None.
        """
        threshold = self._validate_probability_threshold("min_p", min_p)
        p_values = np.asarray(self.hwe_pvalue(samples=samples), dtype=float).ravel()
        mask = np.isfinite(p_values) & (p_values >= threshold)
        return self.filter_variants(mask=mask, include=include, inplace=inplace)

    def filter_ld_pruned(
            self,
            window_size: int = 50,
            step_size: int = 5,
            r2_threshold: float = 0.2,
            samples: Optional[Union[str, Sequence[str], np.ndarray]] = None,
            min_samples: int = 3,
            include: bool = True,
            inplace: bool = False,
        ) -> Optional['SNPObject']:
        """
        Filter variants using greedy sliding-window LD pruning.

        Args:
            window_size (int, default=50):
                Number of variants per sliding window. Must be at least 2.
            step_size (int, default=5):
                Number of variants to advance the window each step. Must be at least 1.
            r2_threshold (float, default=0.2):
                Pairwise LD r-squared threshold. Must be between 0 and 1.
            samples (str or array_like, optional):
                Sample IDs, sample indexes, or boolean mask selecting samples used
                to estimate LD. If None, all samples are used.
            min_samples (int, default=3):
                Minimum number of pairwise non-missing samples needed to compute r-squared.
            include (bool, default=True):
                If True, keeps retained variants. If False, excludes retained variants.
            inplace (bool, default=False):
                If True, modifies ``self`` in place. If False, returns a filtered copy.

        Returns:
            Optional[SNPObject]:
                A filtered SNPObject if ``inplace=False``; otherwise modifies ``self`` and returns None.
        """
        mask = self.ld_prune_mask(
            window_size=window_size,
            step_size=step_size,
            r2_threshold=r2_threshold,
            samples=samples,
            min_samples=min_samples,
        )
        return self.filter_variants(mask=mask, include=include, inplace=inplace)

    def ld_prune(
            self,
            window_size: int = 50,
            step_size: int = 5,
            r2_threshold: float = 0.2,
            samples: Optional[Union[str, Sequence[str], np.ndarray]] = None,
            min_samples: int = 3,
            inplace: bool = False,
        ) -> Optional['SNPObject']:
        """
        Alias for :meth:`filter_ld_pruned`.
        """
        return self.filter_ld_pruned(
            window_size=window_size,
            step_size=step_size,
            r2_threshold=r2_threshold,
            samples=samples,
            min_samples=min_samples,
            include=True,
            inplace=inplace,
        )

    def filter_maf(
            self,
            maf: float = 0.01,
            include: bool = True,
            inplace: bool = False,
        ) -> Optional['SNPObject']:
        """
        Filter variants by minor allele frequency.

        The frequency is computed from observed allele calls only; missing calls
        do not contribute to the denominator.

        Args:
            maf (float, default=0.01):
                Minor allele frequency threshold. Must be between 0 and 0.5.
            include (bool, default=True):
                If True, keeps variants with minor allele frequency greater than
                or equal to ``maf``. If False, excludes those variants.
            inplace (bool, default=False):
                If True, modifies ``self`` in place. If False, returns a filtered copy.

        Returns:
            Optional[SNPObject]:
                A filtered SNPObject if ``inplace=False``; otherwise modifies ``self`` and returns None.
        """
        try:
            threshold = float(maf)
        except (TypeError, ValueError) as exc:
            raise ValueError("'maf' must be a numeric value between 0 and 0.5.") from exc
        if not 0 <= threshold <= 0.5:
            raise ValueError("'maf' must be between 0 and 0.5.")

        minor_af = np.asarray(self.maf(), dtype=float).ravel()
        mask = np.isfinite(minor_af) & (minor_af >= threshold)
        return self.filter_variants(mask=mask, include=include, inplace=inplace)

    def filter_mac(
            self,
            mac: Union[int, float] = 20,
            include: bool = True,
            inplace: bool = False,
        ) -> Optional['SNPObject']:
        """
        Filter variants by minor allele count.

        The count is computed from observed allele calls only; missing calls do
        not contribute to the denominator.

        Args:
            mac (int or float, default=20):
                Minor allele count threshold. Must be non-negative.
            include (bool, default=True):
                If True, keeps variants with minor allele count greater than or
                equal to ``mac``. If False, excludes those variants.
            inplace (bool, default=False):
                If True, modifies ``self`` in place. If False, returns a filtered copy.

        Returns:
            Optional[SNPObject]:
                A filtered SNPObject if ``inplace=False``; otherwise modifies ``self`` and returns None.
        """
        try:
            threshold = float(mac)
        except (TypeError, ValueError) as exc:
            raise ValueError("'mac' must be a non-negative numeric value.") from exc
        if threshold < 0:
            raise ValueError("'mac' must be non-negative.")

        minor_ac = np.asarray(self.mac(), dtype=float).ravel()
        mask = np.isfinite(minor_ac) & (minor_ac >= threshold)
        return self.filter_variants(mask=mask, include=include, inplace=inplace)

    def filter_samples(
            self,
            samples: Optional[Union[str, Sequence[str], np.ndarray, None]] = None,
            indexes: Optional[Union[int, Sequence[int], np.ndarray, None]] = None,
            include: bool = True,
            reorder: bool = False,
            inplace: bool = False
        ) -> Optional['SNPObject']:
        """
        Filter samples based on specified names or indexes.

        This method updates the `samples` and `genotypes` attributes to include or exclude the specified
        samples. The order of the samples is preserved. Set `reorder=True` to match the ordering of the
        provided `samples` and/or `indexes` lists when including.

        If both samples and indexes are provided, any sample matching either a name in samples or an index in
        indexes will be included or excluded.

        This method allows inclusion or exclusion of specific samples by their names or
        indexes. When both sample names and indexes are provided, the union of the specified samples
        is used. Negative indexes are supported and follow
        [NumPy's indexing conventions](https://numpy.org/doc/stable/user/basics.indexing.html).

        Args:
            samples (str or array_like of str, optional):
                 Name(s) of the samples to include or exclude. Can be a single sample name or a
                 sequence of sample names. Default is None.
            indexes (int or array_like of int, optional):
                Index(es) of the samples to include or exclude. Can be a single index or a sequence
                of indexes. Negative indexes are supported. Default is None.
            include (bool, default=True):
                If True, includes only the specified samples. If False, excludes the specified
                samples. Default is True.
            inplace (bool, default=False):
                If True, modifies `self` in place. If False, returns a new `SNPObject` with the samples
                filtered. Default is False.

        Returns:
            Optional[SNPObject]:
                A new `SNPObject` with the specified samples filtered if `inplace=False`.
                If `inplace=True`, modifies `self` in place and returns None.
        """
        if samples is None and indexes is None:
            raise ValueError("At least one of 'samples' or 'indexes' must be provided.")

        n_samples = self.n_samples
        sample_names = np.array(self['samples'])

        # Create mask based on sample names
        if samples is not None:
            samples = np.asarray(samples).ravel()
            mask_samples = np.isin(sample_names, samples)
            missing_samples = samples[~np.isin(samples, sample_names)]
            if missing_samples.size > 0:
                raise ValueError(f"The following specified samples were not found: {missing_samples.tolist()}")
        else:
            mask_samples = np.zeros(n_samples, dtype=bool)

        # Create mask based on sample indexes
        if indexes is not None:
            indexes = np.asarray(indexes).ravel()

            # Validate indexes, allowing negative indexes
            out_of_bounds_indexes = indexes[(indexes < -n_samples) | (indexes >= n_samples)]
            if out_of_bounds_indexes.size > 0:
                raise ValueError(f"One or more sample indexes are out of bounds.")

            # Handle negative indexes
            adjusted_indexes = np.mod(indexes, n_samples)

            mask_indexes = np.zeros(n_samples, dtype=bool)
            mask_indexes[adjusted_indexes] = True
        else:
            mask_indexes = np.zeros(n_samples, dtype=bool)

        # Combine masks using logical OR (union of samples)
        mask_combined = mask_samples | mask_indexes

        if not include:
            mask_combined = ~mask_combined

        # If requested, compute an ordering of selected samples that follows the provided lists.
        ordered_indices = None
        if include and reorder:
            sel_indices = np.where(mask_combined)[0]
            ordered_list: List[int] = []
            added = np.zeros(n_samples, dtype=bool)

            # Prioritize the order in `samples`
            if samples is not None:
                name_to_idx = {name: idx for idx, name in enumerate(sample_names)}
                for s in samples:
                    idx = name_to_idx.get(s)
                    if idx is not None and mask_combined[idx] and not added[idx]:
                        ordered_list.append(idx)
                        added[idx] = True

            # Then respect the order in `indexes`
            if indexes is not None:
                adj_idx = np.mod(np.atleast_1d(indexes), n_samples)
                for idx in adj_idx:
                    if mask_combined[idx] and not added[idx]:
                        ordered_list.append(int(idx))
                        added[idx] = True

            # Finally, append any remaining selected samples in their original order
            for idx in sel_indices:
                if not added[idx]:
                    ordered_list.append(int(idx))

            ordered_indices = np.asarray(ordered_list, dtype=int)

        def subset_sample_axis(key: str, value: np.ndarray) -> np.ndarray:
            arr = np.asarray(value)
            if ordered_indices is not None:
                if key == "calldata_lai" and arr.ndim == 2:
                    hap_idx = np.ravel(
                        np.column_stack([2 * ordered_indices, 2 * ordered_indices + 1])
                    )
                    return arr[:, hap_idx]
                if arr.ndim > 1:
                    return arr[:, ordered_indices, ...]
                return arr[ordered_indices]

            if arr.ndim > 1:
                return arr[:, mask_combined, ...]
            return arr[mask_combined]

        new_samples = (
            subset_sample_axis('samples', self.samples)
            if self.samples is not None
            else None
        )
        new_sample_fid = (
            subset_sample_axis('sample_fid', self.sample_fid)
            if self.sample_fid is not None
            else None
        )
        new_sample_sex = (
            subset_sample_axis('sample_sex', self.sample_sex)
            if self.sample_sex is not None
            else None
        )
        data_updates = {
            key: subset_sample_axis(key, self[key])
            for key in ['genotypes', 'calldata_lai', 'calldata_gp']
            if self[key] is not None
        }
        sample_metadata = getattr(self, "sample_metadata", None)
        new_sample_metadata = None
        if sample_metadata is not None:
            if len(sample_metadata) != n_samples:
                raise ValueError(
                    f"Cannot filter 'sample_metadata': length is {len(sample_metadata)}, "
                    f"expected {n_samples}."
                )
            sample_indices = (
                ordered_indices
                if ordered_indices is not None
                else np.where(mask_combined)[0]
            )
            new_sample_metadata = sample_metadata.iloc[sample_indices].reset_index(drop=True)

        # Apply filtering based on inplace parameter
        if inplace:
            self.sample_fid = None
            self.sample_sex = None
            if new_samples is not None:
                self.samples = new_samples
            self.sample_fid = new_sample_fid
            self.sample_sex = new_sample_sex
            for key, value in data_updates.items():
                self[key] = value
            if sample_metadata is not None:
                self.sample_metadata = new_sample_metadata
            return None
        else:
            # Create A new `SNPObject` with filtered data
            snpobj = self.copy()
            snpobj.sample_fid = None
            snpobj.sample_sex = None
            if new_samples is not None:
                snpobj.samples = new_samples
            snpobj.sample_fid = new_sample_fid
            snpobj.sample_sex = new_sample_sex
            for key, value in data_updates.items():
                snpobj[key] = value
            if sample_metadata is not None:
                snpobj.sample_metadata = new_sample_metadata
            return snpobj

    def detect_chromosome_format(self) -> str:
        """
        Detect the chromosome naming convention in `variants_chrom` based on the prefix
        of the first chromosome identifier in `unique_chrom`.

        Recognized formats:

        - `'chr'`: Format with 'chr' prefix, e.g., 'chr1', 'chr2', ..., 'chrX', 'chrY', 'chrM'.
        - `'chm'`: Format with 'chm' prefix, e.g., 'chm1', 'chm2', ..., 'chmX', 'chmY', 'chmM'.
        - `'chrom'`: Format with 'chrom' prefix, e.g., 'chrom1', 'chrom2', ..., 'chromX', 'chromY', 'chromM'.
        - `'plain'`: Plain format without a prefix, e.g., '1', '2', ..., 'X', 'Y', 'M'.

        If the format does not match any recognized pattern, `'Unknown format'` is returned.

        Returns:
            str:
                A string indicating the detected chromosome format (`'chr'`, `'chm'`, `'chrom'`, or `'plain'`).
                If no recognized format is matched, returns `'Unknown format'`.
        """
        # Select the first unique chromosome identifier for format detection
        chromosome_str = self.unique_chrom[0]

        # Define regular expressions to match each recognized chromosome format
        patterns = {
            'chr': r'^chr(\d+|X|Y|M)$',    # Matches 'chr' prefixed format
            'chm': r'^chm(\d+|X|Y|M)$',    # Matches 'chm' prefixed format
            'chrom': r'^chrom(\d+|X|Y|M)$', # Matches 'chrom' prefixed format
            'plain': r'^(\d+|X|Y|M)$'       # Matches plain format without prefix
        }

        # Iterate through the patterns to identify the chromosome format
        for prefix, pattern in patterns.items():
            if re.match(pattern, chromosome_str):
                return prefix  # Return the recognized format prefix

        # If no pattern matches, return 'Unknown format'
        return 'Unknown format'

    def convert_chromosome_format(
        self,
        from_format: str,
        to_format: str,
        inplace: bool = False
    ) -> Optional['SNPObject']:
        """
        Convert the chromosome format from one naming convention to another in `variants_chrom`.

        Supported formats:

        - `'chr'`: Format with 'chr' prefix, e.g., 'chr1', 'chr2', ..., 'chrX', 'chrY', 'chrM'.
        - `'chm'`: Format with 'chm' prefix, e.g., 'chm1', 'chm2', ..., 'chmX', 'chmY', 'chmM'.
        - `'chrom'`: Format with 'chrom' prefix, e.g., 'chrom1', 'chrom2', ..., 'chromX', 'chromY', 'chromM'.
        - `'plain'`: Plain format without a prefix, e.g., '1', '2', ..., 'X', 'Y', 'M'.

        Args:
            from_format (str):
                The current chromosome format. Acceptable values are `'chr'`, `'chm'`, `'chrom'`, or `'plain'`.
            to_format (str):
                The target format for chromosome data conversion. Acceptable values match `from_format` options.
            inplace (bool, default=False):
                If True, modifies `self` in place. If False, returns a new `SNPObject` with the converted format.
                Default is False.

        Returns:
            Optional[SNPObject]: A new `SNPObject` with the converted chromosome format if `inplace=False`.
            If `inplace=True`, modifies `self` in place and returns None.
        """
        # Define the list of standard chromosome identifiers
        chrom_list = [*map(str, range(1, 23)), 'X', 'Y', 'M']  # M for mitochondrial chromosomes

        # Format mappings for different chromosome naming conventions
        format_mappings = {
            'chr': [f'chr{i}' for i in chrom_list],
            'chm': [f'chm{i}' for i in chrom_list],
            'chrom': [f'chrom{i}' for i in chrom_list],
            'plain': chrom_list,
        }

        # Verify that from_format and to_format are valid naming conventions
        if from_format not in format_mappings or to_format not in format_mappings:
            raise ValueError(f"Invalid format: {from_format} or {to_format}. Must be one of {list(format_mappings.keys())}.")

        # Convert chromosomes to string for consistent comparison
        variants_chrom = self['variants_chrom'].astype(str)

        # Verify that all chromosomes in the object follow the specified `from_format`
        expected_chroms = set(format_mappings[from_format])
        mismatched_chroms = set(variants_chrom) - expected_chroms

        if mismatched_chroms:
            raise ValueError(f"The following chromosomes do not match the `from_format` '{from_format}': {mismatched_chroms}.")

        # Create conditions for selecting based on current `from_format` names
        conditions = [variants_chrom == chrom for chrom in format_mappings[from_format]]

        # Rename chromosomes based on inplace flag
        if inplace:
            self['variants_chrom'] = np.select(conditions, format_mappings[to_format], default='unknown')
            return None
        else:
            snpobject = self.copy()
            snpobject['variants_chrom'] = np.select(conditions, format_mappings[to_format], default='unknown')
            return snpobject

    def match_chromosome_format(self, snpobj: 'SNPObject', inplace: bool = False) -> Optional['SNPObject']:
        """
        Convert the chromosome format in `variants_chrom` from `self` to match the format of a reference `snpobj`.

        Recognized formats:

        - `'chr'`: Format with 'chr' prefix, e.g., 'chr1', 'chr2', ..., 'chrX', 'chrY', 'chrM'.
        - `'chm'`: Format with 'chm' prefix, e.g., 'chm1', 'chm2', ..., 'chmX', 'chmY', 'chmM'.
        - `'chrom'`: Format with 'chrom' prefix, e.g., 'chrom1', 'chrom2', ..., 'chromX', 'chromY', 'chromM'.
        - `'plain'`: Plain format without a prefix, e.g., '1', '2', ..., 'X', 'Y', 'M'.

        Args:
            snpobj (SNPObject):
                The reference SNPObject whose chromosome format will be matched.
            inplace (bool, default=False):
                If True, modifies `self` in place. If False, returns a new `SNPObject` with the
                chromosome format matching that of `snpobj`. Default is False.

        Returns:
            Optional[SNPObject]:
                A new `SNPObject` with matched chromosome format if `inplace=False`.
                If `inplace=True`, modifies `self` in place and returns None.
        """
        # Detect the chromosome naming format of the current SNPObject
        fmt1 = self.detect_chromosome_format()
        if fmt1 == 'Unknown format':
            raise ValueError("The chromosome format of the current SNPObject is unrecognized.")

        # Detect the chromosome naming format of the reference SNPObject
        fmt2 = snpobj.detect_chromosome_format()
        if fmt2 == 'Unknown format':
            raise ValueError("The chromosome format of the reference SNPObject is unrecognized.")

        # Convert the current SNPObject's chromosome format to match the reference format
        return self.convert_chromosome_format(fmt1, fmt2, inplace=inplace)

    def rename_chrom(
        self,
        to_replace: Union[Dict[str, str], str, List[str]] = {'^([0-9]+)$': r'chr\1', r'^chr([0-9]+)$': r'\1'},
        value: Optional[Union[str, List[str]]] = None,
        regex: bool = True,
        inplace: bool = False
    ) -> Optional['SNPObject']:
        """
        Replace chromosome values in `variants_chrom` using patterns or exact matches.

        This method allows flexible chromosome replacements, using regex or exact matches, useful
        for non-standard chromosome formats. For standard conversions (e.g., 'chr1' to '1'),
        consider `convert_chromosome_format`.

        Args:
            to_replace (dict, str, or list of str):
                Pattern(s) or exact value(s) to be replaced in chromosome names. Default behavior
                transforms `<chrom_num>` to `chr<chrom_num>` or vice versa. Non-matching values
                remain unchanged.
                - If str or list of str: Matches will be replaced with `value`.
                - If regex (bool), then any regex matches will be replaced with `value`.
                - If dict: Keys defines values to replace, with corresponding replacements as values.
            value (str or list of str, optional):
                Replacement value(s) if `to_replace` is a string or list. Ignored if `to_replace`
                is a dictionary.
            regex (bool, default=True):
                If True, interprets `to_replace` keys as regex patterns.
            inplace (bool, default=False):
                If True, modifies `self` in place. If False, returns a new `SNPObject` with the chromosomes
                renamed. Default is False.

        Returns:
            Optional[SNPObject]: A new `SNPObject` with the renamed chromosome format if `inplace=False`.
            If `inplace=True`, modifies `self` in place and returns None.
        """
        # Standardize input format: convert `to_replace` and `value` to a dictionary if needed
        if isinstance(to_replace, (str, int)):
            to_replace = [to_replace]
        if isinstance(value, (str, int)):
            value = [value]
        if isinstance(to_replace, list) and isinstance(value, list):
            dictionary = dict(zip(to_replace, value))
        elif isinstance(to_replace, dict) and value is None:
            dictionary = to_replace
        else:
            raise ValueError(
            "Invalid input: `to_replace` and `value` must be compatible types (both str, list of str, or dict)."
        )

        # Vectorized function for replacing values in chromosome array
        vec_replace_values = np.vectorize(self._match_to_replace)

        # Rename chromosomes based on inplace flag
        if inplace:
            self.variants_chrom = vec_replace_values(self.variants_chrom, dictionary, regex)
            return None
        else:
            snpobj = self.copy()
            snpobj.variants_chrom = vec_replace_values(self.variants_chrom, dictionary, regex)
            return snpobj

    def rename_missings(
        self,
        before: Union[int, float, str] = -1,
        after: Union[int, float, str] = '.',
        inplace: bool = False
    ) -> Optional['SNPObject']:
        """
        Replace missing values in the `genotypes` attribute.

        This method identifies missing values in 'genotypes' and replaces them with a specified
        value. By default, it replaces occurrences of `-1` (often used to signify missing data) with `'.'`.

        Args:
            before (int, float, or str, default=-1):
                The current representation of missing values in `genotypes`. Common values might be -1, '.', or NaN.
                Default is -1.
            after (int, float, or str, default='.'):
                The value that will replace `before`. Default is '.'.
            inplace (bool, default=False):
                If True, modifies `self` in place. If False, returns a new `SNPObject` with the applied
                replacements. Default is False.

        Returns:
            Optional[SNPObject]:
                A new `SNPObject` with the renamed missing values if `inplace=False`.
                If `inplace=True`, modifies `self` in place and returns None.
        """
        # Rename missing values in the `genotypes` attribute based on inplace flag
        if inplace:
            self['genotypes'] = np.where(self['genotypes'] == before, after, self['genotypes'])
            return None
        else:
            snpobj = self.copy()
            snpobj['genotypes'] = np.where(snpobj['genotypes'] == before, after, snpobj['genotypes'])
            return snpobj

    def get_common_variants_intersection(
        self,
        snpobj: 'SNPObject',
        index_by: str = 'pos'
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Identify common variants between `self` and the `snpobj` instance based on the specified `index_by` criterion,
        which may match based on chromosome and position (`variants_chrom`, `variants_pos`), ID (`variants_id`), or both.

        This method returns the identifiers of common variants and their corresponding indices in both objects.

        Args:
            snpobj (SNPObject):
                The reference SNPObject to compare against.
            index_by (str, default='pos'):
                Criteria for matching variants. Options:
                - `'pos'`: Matches by chromosome and position (`variants_chrom`, `variants_pos`), e.g., 'chr1-12345'.
                - `'id'`: Matches by variant ID alone (`variants_id`), e.g., 'rs123'.
                - `'pos+id'`: Matches by chromosome, position, and ID (`variants_chrom`, `variants_pos`, `variants_id`), e.g., 'chr1-12345-rs123'.
                Default is 'pos'.

        Returns:
            Tuple containing:
            - list of str: A list of common variant identifiers (as strings).
            - array: An array of indices in `self` where common variants are located.
            - array: An array of indices in `snpobj` where common variants are located.
        """
        # Create unique identifiers for each variant in both SNPObjects based on the specified criterion
        if index_by == 'pos':
            query_identifiers = [f"{chrom}-{pos}" for chrom, pos in zip(self['variants_chrom'], self['variants_pos'])]
            reference_identifiers = [f"{chrom}-{pos}" for chrom, pos in zip(snpobj['variants_chrom'], snpobj['variants_pos'])]
        elif index_by == 'id':
            query_identifiers = self['variants_id'].tolist()
            reference_identifiers = snpobj['variants_id'].tolist()
        elif index_by == 'pos+id':
            query_identifiers = [
                f"{chrom}-{pos}-{ids}" for chrom, pos, ids in zip(self['variants_chrom'], self['variants_pos'], self['variants_id'])
            ]
            reference_identifiers = [
                f"{chrom}-{pos}-{ids}" for chrom, pos, ids in zip(snpobj['variants_chrom'], snpobj['variants_pos'], snpobj['variants_id'])
            ]
        else:
            raise ValueError("`index_by` must be one of 'pos', 'id', or 'pos+id'.")

        return _paired_common_indices(query_identifiers, reference_identifiers)

    def get_common_markers_intersection(
        self,
        snpobj: 'SNPObject'
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Identify common markers between between `self` and the `snpobj` instance. Common markers are identified
        based on matching chromosome (`variants_chrom`), position (`variants_pos`), reference (`variants_ref`),
        and alternate (`variants_alt`) alleles.

        This method returns the identifiers of common markers and their corresponding indices in both objects.

        Args:
            snpobj (SNPObject):
                The reference SNPObject to compare against.

        Returns:
            Tuple containing:
            - list of str: A list of common variant identifiers (as strings).
            - array: An array of indices in `self` where common variants are located.
            - array: An array of indices in `snpobj` where common variants are located.
        """
        # Generate unique identifiers based on chrom, pos, ref, and alt alleles
        query_identifiers = [
            f"{chrom}-{pos}-{ref}-{alt}" for chrom, pos, ref, alt in
            zip(self['variants_chrom'], self['variants_pos'], self['variants_ref'], self['variants_alt'])
        ]
        reference_identifiers = [
            f"{chrom}-{pos}-{ref}-{alt}" for chrom, pos, ref, alt in
            zip(snpobj['variants_chrom'], snpobj['variants_pos'], snpobj['variants_ref'], snpobj['variants_alt'])
        ]

        return _paired_common_indices(query_identifiers, reference_identifiers)

    def subset_to_common_variants(
        self,
        snpobj: 'SNPObject',
        index_by: str = 'pos',
        common_variants_intersection: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        inplace: bool = False
    ) -> Optional['SNPObject']:
        """
        Subset `self` to include only the common variants with a reference `snpobj` based on
        the specified `index_by` criterion, which may match based on chromosome and position
        (`variants_chrom`, `variants_pos`), ID (`variants_id`), or both.

        Args:
            snpobj (SNPObject):
                The reference SNPObject to compare against.
            index_by (str, default='pos'):
                Criteria for matching variants. Options:
                - `'pos'`: Matches by chromosome and position (`variants_chrom`, `variants_pos`), e.g., 'chr1-12345'.
                - `'id'`: Matches by variant ID alone (`variants_id`), e.g., 'rs123'.
                - `'pos+id'`: Matches by chromosome, position, and ID (`variants_chrom`, `variants_pos`, `variants_id`), e.g., 'chr1-12345-rs123'.
                Default is 'pos'.
            common_variants_intersection (Tuple[np.ndarray, np.ndarray], optional):
                Precomputed indices of common variants between `self` and `snpobj`. If None, intersection is
                computed within the function.
            inplace (bool, default=False):
                If True, modifies `self` in place. If False, returns a new `SNPObject` with the common variants
                subsetted. Default is False.

        Returns:
            Optional[SNPObject]:
                A new `SNPObject` with the common variants subsetted if `inplace=False`.
                If `inplace=True`, modifies `self` in place and returns None.
        """
        # Get indices of common variants if not provided
        if common_variants_intersection is None:
            _, query_idx, _ = self.get_common_variants_intersection(snpobj, index_by=index_by)
        else:
            query_idx, _ = common_variants_intersection

        # Use filter_variants method with the identified indices, applying `inplace` as specified
        return self.filter_variants(indexes=query_idx, include=True, inplace=inplace)

    def subset_to_common_markers(
        self,
        snpobj: 'SNPObject',
        common_markers_intersection: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        inplace: bool = False
    ) -> Optional['SNPObject']:
        """
        Subset `self` to include only the common markers with a reference `snpobj`. Common markers are identified
        based on matching chromosome (`variants_chrom`), position (`variants_pos`), reference (`variants_ref`),
        and alternate (`variants_alt`) alleles.

        Args:
            snpobj (SNPObject):
                The reference SNPObject to compare against.
            common_markers_intersection (tuple of arrays, optional):
                Precomputed indices of common markers between `self` and `snpobj`. If None, intersection is
                computed within the function.
            inplace (bool, default=False):
                If True, modifies `self` in place. If False, returns a new `SNPObject` with the common markers
                subsetted. Default is False.

        Returns:
            Optional[SNPObject]:
                A new `SNPObject` with the common markers subsetted if `inplace=False`.
                If `inplace=True`, modifies `self` in place and returns None.
        """
        # Get indices of common markers if not provided
        if common_markers_intersection is None:
            _, query_idx, _ = self.get_common_markers_intersection(snpobj)
        else:
            query_idx, _ = common_markers_intersection

        # Use filter_variants method with the identified indices, applying `inplace` as specified
        return self.filter_variants(indexes=query_idx, include=True, inplace=inplace)

    def merge(
            self,
            snpobj: 'SNPObject',
            force_samples: bool = False,
            prefix: str = '2',
            inplace: bool = False
        ) -> Optional['SNPObject']:
        """
        Merge `self` with `snpobj` along the sample axis.

        This method expects both SNPObjects to contain the same set of SNPs in the same order,
        then combines their genotype (`genotypes`) and LAI (`calldata_lai`) arrays by
        concatenating the sample dimension. Samples from `snpobj` are appended to those in `self`.

        Args:
            snpobj (SNPObject):
                The SNPObject to merge samples with.
            force_samples (bool, default=False):
                If True, duplicate sample names are resolved by prepending the `prefix` to duplicate sample names in
                `snpobj`. Otherwise, merging fails when duplicate sample names are found. Default is False.
            prefix (str, default='2'):
                A string prepended to duplicate sample names in `snpobj` when `force_samples=True`.
                Duplicates are renamed from `<sample_name>` to `<prefix>:<sample_name>`. For instance,
                if `prefix='2'` and there is a conflict with a sample called "sample_1", it becomes "2:sample_1".
            inplace (bool, default=False):
                If True, modifies `self` in place. If False, returns a new `SNPObject` with the merged samples.
                Default is False.

        Returns:
            Optional[SNPObject]: A new SNPObject containing the merged sample data.
        """
        # Merge genotypes if present and compatible
        if self.genotypes is not None and snpobj.genotypes is not None:
            if self.genotypes.shape[0] != snpobj.genotypes.shape[0]:
                raise ValueError(
                    f"Cannot merge SNPObjects: Mismatch in the number of SNPs in `genotypes`.\n"
                    f"`self.genotypes` has {self.genotypes.shape[0]} SNPs, "
                    f"while `snpobj.genotypes` has {snpobj.genotypes.shape[0]} SNPs."
                )
            if self.is_dosage and not snpobj.is_dosage:
                raise ValueError(
                    "Cannot merge SNPObjects: `self` stores dosages, but `snpobj` stores phased calls.\n"
                    "Ensure both objects have the same genotype representation before merging."
                )
            if not self.is_dosage and snpobj.is_dosage:
                raise ValueError(
                    "Cannot merge SNPObjects: `snpobj` stores dosages, but `self` stores phased calls.\n"
                    "Ensure both objects have the same genotype representation before merging."
                )
            genotypes = np.concatenate([self.genotypes, snpobj.genotypes], axis=1)
        else:
            genotypes = None

        # Merge samples if present and compatible, handling duplicates if `force_samples=True`
        merged_fid: Optional[np.ndarray] = None
        merged_sex: Optional[np.ndarray] = None
        if self.samples is not None and snpobj.samples is not None:
            overlapping_samples = set(self.samples).intersection(set(snpobj.samples))
            if overlapping_samples:
                if not force_samples:
                    raise ValueError(
                        f"Cannot merge SNPObjects: Found overlapping sample names {overlapping_samples}.\n"
                        "Samples must be strictly non-overlapping. To allow merging with renaming, set `force_samples=True`."
                    )
                else:
                    # Rename duplicate samples by prepending the file index
                    renamed_samples = [f"{prefix}:{sample}" if sample in overlapping_samples else sample for sample in snpobj.samples]
                    samples = np.concatenate([self.samples, renamed_samples], axis=0)
            else:
                samples = np.concatenate([self.samples, snpobj.samples], axis=0)

            fid_left = self.sample_fid if self.sample_fid is not None else self.samples
            fid_right = snpobj.sample_fid if snpobj.sample_fid is not None else snpobj.samples
            merged_fid = np.concatenate([np.asarray(fid_left), np.asarray(fid_right)], axis=0)
            if self.sample_sex is not None or snpobj.sample_sex is not None:
                sex_left = (
                    self.sample_sex
                    if self.sample_sex is not None
                    else np.repeat("0", len(self.samples))
                )
                sex_right = (
                    snpobj.sample_sex
                    if snpobj.sample_sex is not None
                    else np.repeat("0", len(snpobj.samples))
                )
                merged_sex = np.concatenate([np.asarray(sex_left), np.asarray(sex_right)], axis=0)
        else:
            samples = None

        # Merge LAI data if present and compatible
        if self.calldata_lai is not None and snpobj.calldata_lai is not None:
            if self.calldata_lai.ndim != snpobj.calldata_lai.ndim:
                raise ValueError(
                    f"Cannot merge SNPObjects: Mismatch in `calldata_lai` dimensions.\n"
                    f"`self.calldata_lai` has {self.calldata_lai.ndim} dimensions, "
                    f"while `snpobj.calldata_lai` has {snpobj.calldata_lai.ndim} dimensions."
                )
            if self.calldata_lai.shape[0] != snpobj.calldata_lai.shape[0]:
                raise ValueError(
                    f"Cannot merge SNPObjects: Mismatch in the number of SNPs in `calldata_lai`.\n"
                    f"`self.calldata_lai` has {self.calldata_lai.shape[0]} SNPs, "
                    f"while `snpobj.calldata_lai` has {snpobj.calldata_lai.shape[0]} SNPs."
                )
            calldata_lai = np.concatenate([self.calldata_lai, snpobj.calldata_lai], axis=1)
        else:
            calldata_lai = None

        if self.calldata_gp is not None and snpobj.calldata_gp is not None:
            if self.calldata_gp.ndim != snpobj.calldata_gp.ndim:
                raise ValueError(
                    f"Cannot merge SNPObjects: Mismatch in `calldata_gp` dimensions.\n"
                    f"`self.calldata_gp` has {self.calldata_gp.ndim} dimensions, "
                    f"while `snpobj.calldata_gp` has {snpobj.calldata_gp.ndim} dimensions."
                )
            if self.calldata_gp.shape[0] != snpobj.calldata_gp.shape[0]:
                raise ValueError(
                    f"Cannot merge SNPObjects: Mismatch in the number of SNPs in `calldata_gp`.\n"
                    f"`self.calldata_gp` has {self.calldata_gp.shape[0]} SNPs, "
                    f"while `snpobj.calldata_gp` has {snpobj.calldata_gp.shape[0]} SNPs."
                )
            if self.calldata_gp.shape[2:] != snpobj.calldata_gp.shape[2:]:
                raise ValueError(
                    "Cannot merge SNPObjects: genotype probability columns differ."
                )
            calldata_gp = np.concatenate([self.calldata_gp, snpobj.calldata_gp], axis=1)
        else:
            calldata_gp = None

        if inplace:
            self.genotypes = genotypes
            self.calldata_lai = calldata_lai
            self.calldata_gp = calldata_gp
            self.sample_fid = None
            self.sample_sex = None
            self.samples = samples
            self.sample_fid = merged_fid
            self.sample_sex = merged_sex
            return self

        # Create and return a new SNPObject containing the merged samples
        return SNPObject(
            genotypes=genotypes,
            samples=samples,
            sample_fid=merged_fid,
            sample_sex=merged_sex,
            variants_ref=self.variants_ref,
            variants_alt=self.variants_alt,
            variants_chrom=self.variants_chrom,
            variants_cm=self.variants_cm,
            variants_filter_pass=self.variants_filter_pass,
            variants_id=self.variants_id,
            variants_pos=self.variants_pos,
            variants_qual=self.variants_qual,
            variants_info=self.variants_info,
            calldata_lai=calldata_lai,
            calldata_gp=calldata_gp,
            ancestry_map=self.ancestry_map
        )

    def concat(
        self,
        snpobj: 'SNPObject',
        inplace: bool = False
    ) -> Optional['SNPObject']:
        """
        Concatenate self with snpobj along the SNP axis.

        This method expects both SNPObjects to contain the same set of samples in the same order,
        and that the chromosome(s) in snpobj follow (i.e. have higher numeric identifiers than)
        those in self.

        Args:
            snpobj (SNPObject):
                The SNPObject to concatenate SNPs with.
            inplace (bool, default=False):
                If True, modifies `self` in place. If False, returns a new `SNPObject` with the concatenated SNPs.
                Default is False.

        Returns:
            Optional[SNPObject]: A new SNPObject containing the concatenated SNP data.
        """
        # Merge genotypes if present and compatible
        if self.genotypes is not None and snpobj.genotypes is not None:
            if self.genotypes.shape[1] != snpobj.genotypes.shape[1]:
                raise ValueError(
                    f"Cannot merge SNPObjects: Mismatch in the number of samples in `genotypes`.\n"
                    f"`self.genotypes` has {self.genotypes.shape[1]} samples, "
                    f"while `snpobj.genotypes` has {snpobj.genotypes.shape[1]} samples."
                )
            if self.is_dosage and not snpobj.is_dosage:
                raise ValueError(
                    "Cannot merge SNPObjects: `self` stores dosages, but `snpobj` stores phased calls.\n"
                    "Ensure both objects have the same genotype representation before merging."
                )
            if not self.is_dosage and snpobj.is_dosage:
                raise ValueError(
                    "Cannot merge SNPObjects: `snpobj` stores dosages, but `self` stores phased calls.\n"
                    "Ensure both objects have the same genotype representation before merging."
                )
            genotypes = np.concatenate([self.genotypes, snpobj.genotypes], axis=0)
        else:
            genotypes = None

        if self.samples is not None and snpobj.samples is not None:
            if not np.array_equal(self.samples, snpobj.samples):
                raise ValueError("Cannot concatenate SNPObjects: sample IDs differ or are not in the same order.")
        elif self.samples is not None or snpobj.samples is not None:
            raise ValueError("Cannot concatenate SNPObjects: samples metadata is present in only one object.")

        if self.sample_fid is not None and snpobj.sample_fid is not None:
            if not np.array_equal(self.sample_fid, snpobj.sample_fid):
                raise ValueError("Cannot concatenate SNPObjects: sample_fid values differ.")
        elif self.sample_fid is not None or snpobj.sample_fid is not None:
            raise ValueError("Cannot concatenate SNPObjects: sample_fid metadata is present in only one object.")

        merged_sample_sex = self.sample_sex if self.sample_sex is not None else snpobj.sample_sex
        if self.sample_sex is not None and snpobj.sample_sex is not None:
            if not np.array_equal(self.sample_sex, snpobj.sample_sex):
                raise ValueError("Cannot concatenate SNPObjects: sample_sex values differ.")

        # Merge SNP-related attributes if present
        attributes = [
            'variants_ref', 'variants_alt', 'variants_chrom', 'variants_cm', 'variants_filter_pass',
            'variants_id', 'variants_pos', 'variants_qual', 'variants_info'
        ]
        merged_attrs = {}
        for attr in attributes:
            self_attr = getattr(self, attr, None)
            obj_attr = getattr(snpobj, attr, None)

            # Concatenate if both present
            if self_attr is not None and obj_attr is not None:
                merged_attrs[attr] = np.concatenate([self_attr, obj_attr], axis=0)
            else:
                # If either is None, store None
                merged_attrs[attr] = None

        # Merge LAI data if present and compatible
        if self.calldata_lai is not None and snpobj.calldata_lai is not None:
            if self.calldata_lai.ndim != snpobj.calldata_lai.ndim:
                raise ValueError(
                    f"Cannot merge SNPObjects: Mismatch in `calldata_lai` dimensions.\n"
                    f"`self.calldata_lai` has {self.calldata_lai.ndim} dimensions, "
                    f"while `snpobj.calldata_lai` has {snpobj.calldata_lai.ndim} dimensions."
                )
            if self.calldata_lai.shape[1] != snpobj.calldata_lai.shape[1]:
                raise ValueError(
                    f"Cannot merge SNPObjects: Mismatch in the number of samples in `calldata_lai`.\n"
                    f"`self.calldata_lai` has {self.calldata_lai.shape[1]} samples, "
                    f"while `snpobj.calldata_lai` has {snpobj.calldata_lai.shape[1]} samples."
                )
            calldata_lai = np.concatenate([self.calldata_lai, snpobj.calldata_lai], axis=0)
        else:
            calldata_lai = None

        if self.calldata_gp is not None and snpobj.calldata_gp is not None:
            if self.calldata_gp.ndim != snpobj.calldata_gp.ndim:
                raise ValueError(
                    f"Cannot concatenate SNPObjects: Mismatch in `calldata_gp` dimensions.\n"
                    f"`self.calldata_gp` has {self.calldata_gp.ndim} dimensions, "
                    f"while `snpobj.calldata_gp` has {snpobj.calldata_gp.ndim} dimensions."
                )
            if self.calldata_gp.shape[1:] != snpobj.calldata_gp.shape[1:]:
                raise ValueError(
                    "Cannot concatenate SNPObjects: genotype probability sample/probability dimensions differ."
                )
            calldata_gp = np.concatenate([self.calldata_gp, snpobj.calldata_gp], axis=0)
        else:
            calldata_gp = None

        if inplace:
            self.genotypes = genotypes
            self.calldata_lai = calldata_lai
            self.calldata_gp = calldata_gp
            self.sample_sex = merged_sample_sex
            for attr in attributes:
                self[attr] = merged_attrs[attr]
            return self

        # Create and return a new SNPObject containing the concatenated SNPs
        return SNPObject(
            genotypes=genotypes,
            calldata_lai=calldata_lai,
            calldata_gp=calldata_gp,
            samples=self.samples,
            sample_fid=self.sample_fid,
            sample_sex=merged_sample_sex,
            variants_ref=merged_attrs['variants_ref'],
            variants_alt=merged_attrs['variants_alt'],
            variants_chrom=merged_attrs['variants_chrom'],
            variants_cm=merged_attrs['variants_cm'],
            variants_id=merged_attrs['variants_id'],
            variants_pos=merged_attrs['variants_pos'],
            variants_qual=merged_attrs['variants_qual'],
            variants_info=merged_attrs['variants_info'],
            variants_filter_pass=merged_attrs['variants_filter_pass'],
            ancestry_map=self.ancestry_map
        )

    @classmethod
    def concat_variants(_cls, snpobjs: Sequence['SNPObject']) -> 'SNPObject':
        """
        Concatenate multiple SNPObjects along the SNP axis.

        All objects must have the same sample order and genotype representation.
        """
        if len(snpobjs) == 0:
            raise ValueError("concat_variants requires at least one SNPObject.")
        merged = snpobjs[0].copy()
        for snpobj in snpobjs[1:]:
            merged = merged.concat(snpobj, inplace=False)
        return merged

    def remove_strand_ambiguous_variants(self, inplace: bool = False) -> Optional['SNPObject']:
        """
        A strand-ambiguous variant has reference (`variants_ref`) and alternate (`variants_alt`) alleles
        in the pairs A/T, T/A, C/G, or G/C, where both alleles are complementary and thus indistinguishable
        in terms of strand orientation.

        Args:
            inplace (bool, default=False):
                If True, modifies `self` in place. If False, returns a new `SNPObject` with the
                strand-ambiguous variants removed. Default is False.

        Returns:
            Optional[SNPObject]: A new `SNPObject` with non-ambiguous variants only if `inplace=False`.
            If `inplace=True`, modifies `self` in place and returns None.
        """
        # Identify strand-ambiguous SNPs using vectorized comparisons
        is_AT = (self['variants_ref'] == 'A') & (self['variants_alt'] == 'T')
        is_TA = (self['variants_ref'] == 'T') & (self['variants_alt'] == 'A')
        is_CG = (self['variants_ref'] == 'C') & (self['variants_alt'] == 'G')
        is_GC = (self['variants_ref'] == 'G') & (self['variants_alt'] == 'C')

        # Create a combined mask for all ambiguous variants
        ambiguous_mask = is_AT | is_TA | is_CG | is_GC
        non_ambiguous_idx = np.where(~ambiguous_mask)[0]

        # Count each type of ambiguity using numpy's sum on boolean arrays
        A_T_count = np.sum(is_AT)
        T_A_count = np.sum(is_TA)
        C_G_count = np.sum(is_CG)
        G_C_count = np.sum(is_GC)

        # Log the counts of each type of strand-ambiguous variants
        total_ambiguous = A_T_count + T_A_count + C_G_count + G_C_count
        log.info(f'{A_T_count} ambiguities of A-T type.')
        log.info(f'{T_A_count} ambiguities of T-A type.')
        log.info(f'{C_G_count} ambiguities of C-G type.')
        log.info(f'{G_C_count} ambiguities of G-C type.')

        # Filter out ambiguous variants and keep non-ambiguous ones
        log.debug(f'Removing {total_ambiguous} strand-ambiguous variants...')
        return self.filter_variants(indexes=non_ambiguous_idx, include=True, inplace=inplace)

    def correct_flipped_variants(
        self,
        snpobj: 'SNPObject',
        check_complement: bool = True,
        index_by: str = 'pos',
        common_variants_intersection: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        log_stats: bool = True,
        inplace: bool = False
    ) -> Optional['SNPObject']:
        """
        Correct flipped variants between between `self` and a reference `snpobj`, where reference (`variants_ref`)
        and alternate (`variants_alt`) alleles are swapped.

        Flip Detection Based on `check_complement`:

        - If `check_complement=False`, only direct allele swaps are considered:
            1. Direct Swap: `self.variants_ref == snpobj.variants_alt` and `self.variants_alt == snpobj.variants_ref`.

        - If `check_complement=True`, both direct and complementary swaps are considered, with four possible cases:
            1. Direct Swap: `self.variants_ref == snpobj.variants_alt` and `self.variants_alt == snpobj.variants_ref`.
            2. Complement Swap of Ref: `complement(self.variants_ref) == snpobj.variants_alt` and `self.variants_alt == snpobj.variants_ref`.
            3. Complement Swap of Alt: `self.variants_ref == snpobj.variants_alt` and `complement(self.variants_alt) == snpobj.variants_ref`.
            4. Complement Swap of both Ref and Alt: `complement(self.variants_ref) == snpobj.variants_alt` and `complement(self.variants_alt) == snpobj.variants_ref`.

        Note: Variants where `self.variants_ref == self.variants_alt` are ignored as they are ambiguous.

        Correction Process:
        - Swaps `variants_ref` and `variants_alt` alleles in `self` to align with `snpobj`.
        - Flips called 2D diploid dosages as `2 - dosage` and called 3D allele indexes as
          `1 - allele`, while preserving missing values.

        Args:
            snpobj (SNPObject):
                The reference SNPObject to compare against.
            check_complement (bool, default=True):
                If True, also checks for complementary base pairs (A/T, T/A, C/G, and G/C) when identifying swapped variants.
                Default is True.
            index_by (str, default='pos'):
                Criteria for matching variants. Options:
                - `'pos'`: Matches by chromosome and position (`variants_chrom`, `variants_pos`), e.g., 'chr1-12345'.
                - `'id'`: Matches by variant ID alone (`variants_id`), e.g., 'rs123'.
                - `'pos+id'`: Matches by chromosome, position, and ID (`variants_chrom`, `variants_pos`, `variants_id`), e.g., 'chr1-12345-rs123'.
                Default is 'pos'.
            common_variants_intersection (tuple of arrays, optional):
                Precomputed indices of common variants between `self` and `snpobj`. If None, intersection is
                computed within the function.
            log_stats (bool, default=True):
                If True, logs statistical information about matching and ambiguous alleles. Default is True.
            inplace (bool, default=False):
                If True, modifies `self` in place. If False, returns a new `SNPObject` with corrected
                flips. Default is False.

        Returns:
            Optional[SNPObject]:
                A new `SNPObject` with corrected flips if `inplace=False`.
                If `inplace=True`, modifies `self` in place and returns None.
        """
        # Define complement mappings for nucleotides
        complement_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

        # Helper function to get the complement of a base
        def get_complement(base: str) -> str:
            return complement_map.get(base, base)

        # Get common variant indices if not provided
        if common_variants_intersection != None:
            query_idx, reference_idx = common_variants_intersection
        else:
            _, query_idx, reference_idx = self.get_common_variants_intersection(snpobj, index_by=index_by)

        # Log statistics on matching alleles if enabled
        if log_stats:
            matching_ref = np.sum(self['variants_ref'][query_idx] == snpobj['variants_ref'][reference_idx])
            matching_alt = np.sum(self['variants_alt'][query_idx] == snpobj['variants_alt'][reference_idx])
            ambiguous = np.sum(self['variants_ref'][query_idx] == self['variants_alt'][query_idx])
            log.info(f"Matching reference alleles (ref=ref'): {matching_ref}, Matching alternate alleles (alt=alt'): {matching_alt}.")
            log.info(f"Number of ambiguous alleles (ref=alt): {ambiguous}.")

        # Identify indices where `ref` and `alt` alleles are swapped
        if not check_complement:
            # Simple exact match for swapped alleles
            swapped_ref = (self['variants_ref'][query_idx] == snpobj['variants_alt'][reference_idx])
            swapped_alt = (self['variants_alt'][query_idx] == snpobj['variants_ref'][reference_idx])
        else:
            # Check for swapped or complementary-swapped alleles
            swapped_ref = (
                (self['variants_ref'][query_idx] == snpobj['variants_alt'][reference_idx]) |
                (np.vectorize(get_complement)(self['variants_ref'][query_idx]) == snpobj['variants_alt'][reference_idx])
            )
            swapped_alt = (
                (self['variants_alt'][query_idx] == snpobj['variants_ref'][reference_idx]) |
                (np.vectorize(get_complement)(self['variants_alt'][query_idx]) == snpobj['variants_ref'][reference_idx])
            )

        # Filter out ambiguous variants where `ref` and `alt` alleles match (ref=alt)
        not_ambiguous = (self['variants_ref'][query_idx] != self['variants_alt'][query_idx])

        # Indices in `self` of flipped variants
        flip_idx_query = query_idx[swapped_ref & swapped_alt & not_ambiguous]

        # Correct the identified variant flips
        if len(flip_idx_query) > 0:
            log.info(f'Correcting {len(flip_idx_query)} variant flips...')

            temp_alts = self['variants_alt'][flip_idx_query]
            temp_refs = self['variants_ref'][flip_idx_query]

            def flip_genotypes(target: 'SNPObject') -> None:
                if target.genotypes is None:
                    return
                genotypes = np.asarray(target.genotypes)
                selected = genotypes[flip_idx_query]
                called = np.isfinite(selected) & (selected >= 0)
                corrected = selected.copy()
                if genotypes.ndim == 2:
                    corrected[called] = 2 - selected[called]
                elif genotypes.ndim == 3:
                    corrected[called] = 1 - selected[called]
                else:
                    raise ValueError("`genotypes` must be a 2D dosage or 3D allele-call array.")
                target.genotypes[flip_idx_query] = corrected

            # Correct the variant flips based on whether the operation is in-place or not
            if inplace:
                self['variants_alt'][flip_idx_query] = temp_refs
                self['variants_ref'][flip_idx_query] = temp_alts
                flip_genotypes(self)
                return None
            else:
                snpobj = self.copy()
                snpobj['variants_alt'][flip_idx_query] = temp_refs
                snpobj['variants_ref'][flip_idx_query] = temp_alts
                flip_genotypes(snpobj)
                return snpobj
        else:
            log.info('No variant flips found to correct.')
            return self if not inplace else None

    def remove_mismatching_variants(
        self,
        snpobj: 'SNPObject',
        index_by: str = 'pos',
        common_variants_intersection: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        inplace: bool = False
    ) -> Optional['SNPObject']:
        """
        Remove variants from `self`, where reference (`variants_ref`) and/or alternate (`variants_alt`) alleles
        do not match with a reference `snpobj`.

        Args:
            snpobj (SNPObject):
                The reference SNPObject to compare against.
            index_by (str, default='pos'):
                Criteria for matching variants. Options:
                - `'pos'`: Matches by chromosome and position (`variants_chrom`, `variants_pos`), e.g., 'chr1-12345'.
                - `'id'`: Matches by variant ID alone (`variants_id`), e.g., 'rs123'.
                - `'pos+id'`: Matches by chromosome, position, and ID (`variants_chrom`, `variants_pos`, `variants_id`), e.g., 'chr1-12345-rs123'.
                Default is 'pos'.
            common_variants_intersection (tuple of arrays, optional):
                Precomputed indices of common variants between `self` and the reference `snpobj`.
                If None, the intersection is computed within the function.
            inplace (bool, default=False):
                If True, modifies `self` in place. If False, returns a new `SNPObject` without
                mismatching variants. Default is False.

        Returns:
            Optional[SNPObject]:
                A new `SNPObject` without mismatching variants if `inplace=False`.
                If `inplace=True`, modifies `self` in place and returns None.
        """
        # Get common variant indices if not provided
        if common_variants_intersection is not None:
            query_idx, reference_idx = common_variants_intersection
        else:
            _, query_idx, reference_idx = self.get_common_variants_intersection(snpobj, index_by=index_by)

        # Vectorized comparison of `ref` and `alt` alleles
        ref_mismatch = self['variants_ref'][query_idx] != snpobj['variants_ref'][reference_idx]
        alt_mismatch = self['variants_alt'][query_idx] != snpobj['variants_alt'][reference_idx]
        mismatch_mask = ref_mismatch | alt_mismatch

        # Identify indices in `self` of mismatching variants
        mismatch_idx = query_idx[mismatch_mask]

        # Compute total number of variant mismatches
        total_mismatches = np.sum(mismatch_mask)

        # Filter out mismatching variants
        log.debug(f'Removing {total_mismatches} mismatching variants...')
        return self.filter_variants(indexes=mismatch_idx, include=False, inplace=inplace)

    def shuffle_variants(self, inplace: bool = False) -> Optional['SNPObject']:
        """
        Randomly shuffle the positions of variants in the SNPObject, ensuring that all associated
        data (e.g., `genotypes` and variant-specific attributes) remain aligned.

        Args:
            inplace (bool, default=False):
                If True, modifies `self` in place. If False, returns a new `SNPObject` with
                shuffled variants. Default is False.

        Returns:
            Optional[SNPObject]:
                A new `SNPObject` without shuffled variant positions if `inplace=False`.
                If `inplace=True`, modifies `self` in place and returns None.
        """
        # Generate a random permutation index for shuffling variant positions
        shuffle_index = np.random.permutation(self.n_snps)

        # Apply shuffling to all relevant attributes using the class's dictionary-like interface
        if inplace:
            for key in self.keys():
                if self[key] is not None:
                    if key in ('genotypes', 'calldata_lai', 'calldata_gp'):
                        # Genotype-like arrays are shuffled along the SNP axis.
                        self[key] = self[key][shuffle_index, ...]
                    elif 'variant' in key:
                        # snpobj attributes are 1D arrays
                        self[key] = np.asarray(self[key])[shuffle_index]
            return None
        else:
            shuffled_snpobj = self.copy()
            for key in shuffled_snpobj.keys():
                if shuffled_snpobj[key] is not None:
                    if key in ('genotypes', 'calldata_lai', 'calldata_gp'):
                        shuffled_snpobj[key] = shuffled_snpobj[key][shuffle_index, ...]
                    elif 'variant' in key:
                        shuffled_snpobj[key] = np.asarray(shuffled_snpobj[key])[shuffle_index]
            return shuffled_snpobj

    def set_empty_to_missing(self, inplace: bool = False) -> Optional['SNPObject']:
        """
        Replace empty strings `''` with missing values `'.'` in attributes of `self`.

        Args:
            inplace (bool, default=False):
                If True, modifies `self` in place. If False, returns a new `SNPObject` with empty
                strings `''` replaced by missing values `'.'`. Default is False.

        Returns:
            Optional[SNPObject]:
                A new `SNPObject` with empty strings replaced if `inplace=False`.
                If `inplace=True`, modifies `self` in place and returns None.
        """
        if inplace:
            if self.variants_alt is not None:
                self.variants_alt[self.variants_alt == ''] = '.'
            if self.variants_ref is not None:
                self.variants_ref[self.variants_ref == ''] = '.'
            if self.variants_qual is not None:
                self.variants_qual = self.variants_qual.astype(str)
                self.variants_qual[(self.variants_qual == '') | (self.variants_qual == 'nan')] = '.'
            if self.variants_chrom is not None:
                self.variants_chrom = self.variants_chrom.astype(str)
                self.variants_chrom[self.variants_chrom == ''] = '.'
            if self.variants_filter_pass is not None:
                self.variants_filter_pass[self.variants_filter_pass == ''] = '.'
            if self.variants_id is not None:
                self.variants_id[self.variants_id == ''] = '.'
            if self.variants_info is not None:
                self.variants_info = self.variants_info.astype(str)
                self.variants_info[(self.variants_info == '') | (self.variants_info == 'nan')] = '.'
            return self
        else:
            snpobj = self.copy()
            if snpobj.variants_alt is not None:
                snpobj.variants_alt[snpobj.variants_alt == ''] = '.'
            if snpobj.variants_ref is not None:
                snpobj.variants_ref[snpobj.variants_ref == ''] = '.'
            if snpobj.variants_qual is not None:
                snpobj.variants_qual = snpobj.variants_qual.astype(str)
                snpobj.variants_qual[(snpobj.variants_qual == '') | (snpobj.variants_qual == 'nan')] = '.'
            if snpobj.variants_chrom is not None:
                snpobj.variants_chrom[snpobj.variants_chrom == ''] = '.'
            if snpobj.variants_filter_pass is not None:
                snpobj.variants_filter_pass[snpobj.variants_filter_pass == ''] = '.'
            if snpobj.variants_id is not None:
                snpobj.variants_id[snpobj.variants_id == ''] = '.'
            if snpobj.variants_info is not None:
                snpobj.variants_info = snpobj.variants_info.astype(str)
                snpobj.variants_info[(snpobj.variants_info == '') | (snpobj.variants_info == 'nan')] = '.'
            return snpobj

    def convert_to_window_level(
        self,
        window_size: Optional[int] = None,
        physical_pos: Optional[np.ndarray] = None,
        chromosomes: Optional[np.ndarray] = None,
        window_sizes: Optional[np.ndarray] = None,
        laiobj: Optional['LocalAncestryObject'] = None
    ) -> 'LocalAncestryObject':
        """
        Aggregate the `calldata_lai` attribute into genomic windows within a
        `snputils.ancestry.genobj.LocalAncestryObject`.

        Window definitions are resolved in this precedence order: fixed window
        size via ``window_size``; explicit boundaries via ``physical_pos``
        (optionally with ``chromosomes`` and ``window_sizes``); or reuse of
        existing window metadata from ``laiobj``.

        Args:
            window_size (int, optional):
                Number of SNPs in each window if defining fixed-size windows. If the total number of
                SNPs in a chromosome is not evenly divisible by the window size, the last window on that
                chromosome will include all remaining SNPs and therefore be larger than the specified size.
            physical_pos (array of shape (n_windows, 2), optional):
                A 2D array containing the start and end physical positions for each window.
            chromosomes (array of shape (n_windows,), optional):
                An array with chromosome numbers corresponding to each genomic window.
            window_sizes (array of shape (n_windows,), optional):
                An array specifying the number of SNPs in each genomic window.
            laiobj (LocalAncestryObject, optional):
                A reference `LocalAncestryObject` from which to copy existing window definitions.

        Returns:
            LocalAncestryObject:
                A LocalAncestryObject containing window-level ancestry data.
        """
        from snputils.ancestry.genobj.local import LocalAncestryObject

        if window_size is None and physical_pos is None and laiobj is None:
            raise ValueError("One of `window_size`, `physical_pos`, or `laiobj` must be provided.")

        # Fixed window size
        if window_size is not None:
            physical_pos = []   # Boundaries [start, end] of each window
            chromosomes = []    # Chromosome for each window
            window_sizes = []   # Number of SNPs for each window
            for chrom in self.unique_chrom:
                # Extract indices corresponding to this chromosome
                mask_chrom = (self.variants_chrom == chrom)
                # Subset to this chromosome
                pos_chrom = self.variants_pos[mask_chrom]
                # Number of SNPs for this chromosome
                n_snps_chrom = pos_chrom.size

                # Initialize the start of the first window with the position of the first SNP
                current_start = self.variants_pos[0]

                # Number of full windows with exactly `window_size` SNPs
                n_full_windows = n_snps_chrom // window_size

                # Build all but the last window
                for i in range(n_full_windows-1):
                    current_end = self.variants_pos[(i+1) * window_size - 1]
                    physical_pos.append([current_start, current_end])
                    chromosomes.append(chrom)
                    window_sizes.append(window_size)
                    current_start = self.variants_pos[(i+1) * window_size]

                # Build the last window
                current_end = self.variants_pos[-1]
                physical_pos.append([current_start, current_end])
                chromosomes.append(chrom)
                window_sizes.append(n_snps_chrom - ((n_full_windows - 1) * window_size))

            physical_pos = np.array(physical_pos)
            chromosomes = np.array(chromosomes)
            window_sizes = np.array(window_sizes)

        # Custom start and end positions
        elif physical_pos is not None:
            # Check if there is exactly one chromosome
            if chromosomes is None:
                unique_chrom = self.unique_chrom
                if len(unique_chrom) == 1:
                    # We assume all windows belong to this single chromosome
                    single_chrom = unique_chrom[0]
                    chromosomes = np.array([single_chrom] * physical_pos.shape[0])
                else:
                    raise ValueError("Multiple chromosomes detected, but `chromosomes` was not provided.")

        # Match existing windows to a reference laiobj
        elif laiobj is not None:
            physical_pos = laiobj.physical_pos
            chromosomes = laiobj.chromosomes
            window_sizes = laiobj.window_sizes

        # Allocate an output LAI array
        n_windows = physical_pos.shape[0]
        n_samples = self.n_samples
        if self.calldata_lai.ndim == 3:
            lai = np.zeros((n_windows, n_samples, 2))
        else:
            lai = np.zeros((n_windows, n_samples*2))

        # For each window, find the relevant SNPs and compute the mode of the ancestries
        for i, ((start, end), chrom) in enumerate(zip(physical_pos, chromosomes)):
            snps_mask = (
                (self.variants_chrom == chrom) &
                (self.variants_pos >= start) &
                (self.variants_pos <= end)
            )
            if np.any(snps_mask):
                lai_mask = self.calldata_lai[snps_mask, ...]
                mode_ancestries = mode(lai_mask, axis=0, nan_policy='omit').mode
                lai[i] = mode_ancestries
            else:
                lai[i] = np.nan

        # Generate haplotype labels, e.g. "Sample1.0", "Sample1.1"
        haplotypes = [f"{sample}.{i}" for sample in self.samples for i in range(2)]

        # If original data was (n_snps, n_samples, 2), flatten to (n_windows, n_samples*2)
        if self.calldata_lai.ndim == 3:
            lai = lai.reshape(n_windows, -1)

        # Aggregate into a LocalAncestryObject
        return LocalAncestryObject(
            haplotypes=haplotypes,
            lai=lai,
            samples=self.samples,
            ancestry_map=self.ancestry_map,
            window_sizes=window_sizes,
            physical_pos=physical_pos,
            chromosomes=chromosomes
        )

    def save(self, file: Union[str, Path]) -> None:
        """
        Save the data stored in `self` to a specified file.

        The format of the saved file is determined by the file extension provided in the `file`
        argument.

        Supported formats:

        - `.bed`: Binary PED (Plink) format.
        - `.bgen`: BGEN genotype probability format.
        - `.pgen`: Plink2 binary genotype format.
        - `.vcf`: Variant Call Format.
        - `.bcf`: Binary Call Format.
        - `.pkl`: Pickle format for saving `self` in serialized form.

        Args:
            file (str or pathlib.Path):
                Path to the file where the data will be saved. The extension of the file determines the save format.
                Supported extensions: `.bed`, `.bgen`, `.pgen`, `.vcf`, `.bcf`, `.pkl`.
        """
        ext = Path(file).suffix.lower()
        if ext == '.bed':
            self.save_bed(file)
        elif ext == '.bgen':
            self.save_bgen(file)
        elif ext == '.pgen':
            self.save_pgen(file)
        elif ext == '.vcf':
            self.save_vcf(file)
        elif ext == '.bcf':
            self.save_bcf(file)
        elif ext == '.pkl':
            self.save_pickle(file)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def save_bed(
        self,
        file: Union[str, Path],
        rename_missing_values: bool = True,
        before: Union[int, float, str] = -1,
        after: Union[int, float, str] = '.',
        sample_phenotype: Optional[Union[np.ndarray, Sequence[Union[str, int, float]], str, int, float]] = None,
    ) -> None:
        """
        Save the data stored in `self` to a `.bed` file.

        Args:
            file (str or pathlib.Path):
                Path to the file where the data will be saved. It should end with `.bed`.
                If the provided path does not have this extension, it will be appended.
            rename_missing_values (bool, optional):
                If True, renames potential missing values in `genotypes` before writing.
            before (int, float, or str, default=-1):
                The current representation of missing values in `genotypes`.
            after (int, float, or str, default='.'):
                The value that will replace `before`.
            sample_phenotype (optional):
                PLINK phenotype value per sample, or a scalar used for all samples.
        """
        from snputils.snp.io.write.bed import BEDWriter
        writer = BEDWriter(snpobj=self, filename=file)
        writer.write(
            rename_missing_values=rename_missing_values,
            before=before,
            after=after,
            sample_phenotype=sample_phenotype,
        )

    def save_pgen(self, file: Union[str, Path]) -> None:
        """
        Save the data stored in `self` to a `.pgen` file.

        Args:
            file (str or pathlib.Path):
                Path to the file where the data will be saved. It should end with `.pgen`.
                If the provided path does not have this extension, it will be appended.
        """
        from snputils.snp.io.write.pgen import PGENWriter
        writer = PGENWriter(snpobj=self, filename=file)
        writer.write()

    def save_bgen(self, file: Union[str, Path]) -> None:
        """
        Save the data stored in `self` to a `.bgen` file.

        Args:
            file (str or pathlib.Path):
                Path to the file where the data will be saved. It should end with `.bgen`.
                If the provided path does not have this extension, it will be appended.
        """
        from snputils.snp.io.write.bgen import BGENWriter
        writer = BGENWriter(snpobj=self, filename=file)
        writer.write()

    def save_vcf(self, file: Union[str, Path]) -> None:
        """
        Save the data stored in `self` to a `.vcf` file.

        Args:
            file (str or pathlib.Path):
                Path to the file where the data will be saved. It should end with `.vcf`.
                If the provided path does not have this extension, it will be appended.
        """
        from snputils.snp.io.write.vcf import VCFWriter
        writer = VCFWriter(snpobj=self, filename=file)
        writer.write()

    def save_bcf(self, file: Union[str, Path], phased: bool = False) -> None:
        """
        Save the data stored in `self` to a `.bcf` file.

        Args:
            file (str or pathlib.Path):
                Path to the file where the data will be saved. It should end with `.bcf`.
                If the provided path does not have this extension, it will be appended.
            phased (bool, optional):
                If True, genotype data is written in phased format.
                If False, genotype data is written in unphased format. Defaults to False.
        """
        from snputils.snp.io.write.bcf import BCFWriter
        writer = BCFWriter(snpobj=self, filename=file, phased=phased)
        writer.write()

    def save_pickle(self, file: Union[str, Path]) -> None:
        """
        Save `self` in serialized form to a `.pkl` file.

        Args:
            file (str or pathlib.Path):
                Path to the file where the data will be saved. It should end with `.pkl`.
                If the provided path does not have this extension, it will be appended.
        """
        import pickle
        with open(file, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def _match_to_replace(val: Union[str, int, float], dictionary: Dict[Any, Any], regex: bool = True) -> Union[str, int, float]:
        """
        Find a matching key in the provided dictionary for the given value `val`
        and replace it with the corresponding value.

        Args:
            val (str, int, or float):
                The value to be matched and potentially replaced.
            dictionary (Dict):
                A dictionary containing keys and values for matching and replacement.
                The keys should match the data type of `val`.
            regex (bool):
                If True, interprets keys in `dictionary` as regular expressions.
                Default is True.

        Returns:
            str, int, or float:
                The replacement value from `dictionary` if a match is found; otherwise, the original `val`.
        """
        if regex:
            # Use regular expression matching to find replacements
            for key, value in dictionary.items():
                if isinstance(key, str):
                    match = re.match(key, val)
                    if match:
                        # Replace using the first matching regex pattern
                        return re.sub(key, value, val)
            # Return the original value if no regex match is found
            return val
        else:
            # Return the value for `val` if present in `dictionary`; otherwise, return `val`
            return dictionary.get(val, val)

    @staticmethod
    def _get_chromosome_number(chrom_string: str) -> Union[int, str]:
        """
        Extracts the chromosome number from the given chromosome string.

        Args:
            chrom_string (str):
                The chromosome identifier.

        Returns:
            int or str:
                The numeric representation of the chromosome if detected.
                Returns 10001 for 'X' or 'chrX', 10002 for 'Y' or 'chrY',
                and the original `chrom_string` if unrecognized.
        """
        if chrom_string.isdigit():
            return int(chrom_string)
        else:
            chrom_num = re.search(r'\d+', chrom_string)
            if chrom_num:
                return int(chrom_num.group())
            elif chrom_string.lower() in ['x', 'chrx']:
                return 10001
            elif chrom_string.lower() in ['y', 'chry']:
                return 10002
            else:
                log.warning(f"Chromosome nomenclature not standard. Chromosome: {chrom_string}")
                return chrom_string

    def _sanity_check(self) -> None:
        """
        Perform sanity checks to ensure LAI and ancestry map consistency.

        This method checks that all unique ancestries in the LAI data are represented
        in the ancestry map if it is provided.
        """
        if self.__calldata_lai is not None and self.__ancestry_map is not None:
            unique_ancestries = np.unique(self.__calldata_lai)
            missing_ancestries = [anc for anc in unique_ancestries if str(anc) not in self.__ancestry_map]
            if missing_ancestries:
                warnings.warn(f"Missing ancestries in ancestry_map: {missing_ancestries}")
        if self.__sample_fid is not None:
            if self.__samples is None:
                raise ValueError("'sample_fid' is set but 'samples' is None; both are required together.")
            if len(self.__sample_fid) != len(self.__samples):
                raise ValueError(
                    f"Length mismatch: sample_fid has {len(self.__sample_fid)} entries "
                    f"but samples has {len(self.__samples)}."
                )
        if self.__sample_sex is not None:
            if self.__samples is None:
                raise ValueError("'sample_sex' is set but 'samples' is None; both are required together.")
            if len(self.__sample_sex) != len(self.__samples):
                raise ValueError(
                    f"Length mismatch: sample_sex has {len(self.__sample_sex)} entries "
                    f"but samples has {len(self.__samples)}."
                )
