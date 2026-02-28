from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import copy
from typing import Any, Union, Tuple, List, Sequence, Dict, Optional
import pygrgl as pyg
import subprocess
log = logging.getLogger(__name__)
import tempfile

GRGType = Union[pyg.GRG, pyg.MutableGRG]


class GRGObject:
    """
    A class for Single Nucleotide Polymorphism (SNP) data.
    """
    def __init__(
        self,
        calldata_gt: Optional[GRGType] = None,
        filename: Optional[str] = None,
        mutable: Optional[bool] = None
    ) -> None:
        """
        Args:
            calldata_gt (GRG | MutableGRG, optional): 
                A Genotype Representation Graph containing genotype data for each sample. 
            filename (str, optional)
                File storing the GRG.
        """
        self.__calldata_gt = calldata_gt
        self.__filename = filename
        self.__mutable = mutable
        self.__latest   = False

    def __getitem__(self, key: str) -> Any:
        """
        To access an attribute of the class using the square bracket notation,
        similar to a dictionary.
        """
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f'Invalid key: {key}.')

    def __setitem__(self, key: str, value: Any):
        """
        To set an attribute of the class using the square bracket notation,
        similar to a dictionary.
        """
        try:
            setattr(self, key, value)
        except AttributeError:
            raise KeyError(f'Invalid key: {key}.')

    @property
    def calldata_gt(self) -> np.ndarray:
        """
        Retrieve `calldata_gt`.

        Returns:
            **GRG | MutableGRG:** 
                An GRG containing genotype data for all samples.
        """
        return self.__calldata_gt

    @calldata_gt.setter
    def calldata_gt(self, x: GRGType):
        """
        Update `calldata_gt`.
        """
        self.__calldata_gt = x


    @property
    def filename(self) -> str:
        """
        Retrieve `filename`.

        Returns:
            **str** 
                A string containing the file name.
        """
        return self.__filename

    @filename.setter
    def filename(self, x: str):
        """
        Update `filename`.
        """
        self.__filename = x
    
    @property
    def mutable(self) -> Optional[bool]:
        return self.__mutable

    def allele_freq(self) -> np.ndarray:
        # allele frequency array
        al_freq = np.ones(self.calldata_gt.num_samples) / self.calldata_gt.num_samples
        return pyg.dot_product(self.calldata_gt, al_freq, pyg.TraversalDirection.UP)

    def dot_product(self, array: np.ndarray, traversal_direction: pyg.TraversalDirection):
        return pyg.dot_product(self.calldata_gt, array, traversal_direction)
    
    # TODO: consider moving this elsewhere.
    def allele_freq_from_file(self, filename: Optional[str] = None) -> pd.DataFrame:
        newfile = filename if filename is not None else self.__filename
        if newfile is None:
            raise ValueError("Either pass in a filename, or store an existing GRG filename.")

        with tempfile.NamedTemporaryFile() as fp:
            subprocess.run(["grg", "process", "freq", f"{newfile}"], stdout=fp, check=True)
            fp.seek(0) # set the file cursor
            return pd.read_csv(fp.name, sep="\t")
        
        
    def gwas(self, phenotype_file: str, filename: Optional[str] = None) -> pd.DataFrame:
        grg_file = filename if filename is not None else self.__filename
        if grg_file is None:
            raise ValueError("Either pass in a GRG filename, or store an existing GRG filename.")

        with tempfile.NamedTemporaryFile(suffix=".tsv") as fp:
            try:
                subprocess.run(
                    ["grapp", "assoc", "-p", f"{phenotype_file}", "-o", fp.name, f"{grg_file}"],
                    check=True,
                )
            except FileNotFoundError as exc:
                raise ImportError(
                    "GWAS support requires the optional dependency 'grapp'. "
                    "Install it with: pip install grapp"
                ) from exc
            return pd.read_csv(fp.name, sep="\t")
    
    def merge(self, combineNodes : bool = False, *args) -> Optional[GRGType]:
        # assert self.__mutable and isinstance(self.calldata_gt, pyg.MutableGRG), "GRG must be mutable"
        for arg in args:
            if not isinstance(arg, str):
                raise TypeError("All merge inputs must be strings.")
        # list of files, and combineNodes 
        self.__calldata_gt.merge(list(args), combineNodes)
        #pep8 be damned
        # if inplace: self.__calldata_gt = merged_data
        # else      : return merged_data

    def n_samples(self, ploidy = 2) -> int:
        """
        Get number of samples from GRG. Diploid by default. 
        """
        return int(self.__calldata_gt.num_samples / ploidy)
    
    def n_snps(self) -> int:
        return self.__calldata_gt.num_mutations

    def _sample_ids(self, n_samples: int, sample_prefix: str) -> np.ndarray:
        default_ids = [f"{sample_prefix}_{idx}" for idx in range(n_samples)]
        if self.__calldata_gt is None:
            return np.asarray(default_ids, dtype=object)

        has_individual_ids = bool(getattr(self.__calldata_gt, "has_individual_ids", False))
        num_individuals = int(getattr(self.__calldata_gt, "num_individuals", 0))
        if has_individual_ids and n_samples == num_individuals:
            ids = []
            for idx in range(n_samples):
                try:
                    sample_id = str(self.__calldata_gt.get_individual_id(idx))
                except RuntimeError:
                    sample_id = ""
                ids.append(sample_id if sample_id else default_ids[idx])
        else:
            ids = default_ids

        # Keep IDs unique for downstream writers.
        seen = {}
        unique_ids = []
        for idx, sample_id in enumerate(ids):
            count = seen.get(sample_id, 0)
            unique_ids.append(sample_id if count == 0 else f"{sample_id}_{count}")
            seen[sample_id] = count + 1

        return np.asarray(unique_ids, dtype=object)

    def to_snpobject(
        self,
        sum_strands: bool = False,
        chrom: str = ".",
        sample_prefix: str = "sample",
    ):
        """
        Convert the GRG to a dense SNPObject.

        Notes:
            - This materializes the full genotype matrix, so memory usage scales with
              `num_mutations * num_samples`.
            - For diploid GRGs and `sum_strands=False`, output has shape
              `(n_snps, n_samples, 2)`.
            - For `sum_strands=True`, output has shape `(n_snps, n_samples)` with
              per-individual allele counts.
        """
        from snputils.snp.genobj.snpobj import SNPObject

        if self.__calldata_gt is None:
            raise ValueError("Cannot convert to SNPObject: `calldata_gt` is None.")

        grg = self.__calldata_gt
        n_mutations = int(grg.num_mutations)
        n_haplotypes = int(grg.num_samples)
        ploidy = int(getattr(grg, "ploidy", 2))

        if ploidy <= 0:
            raise ValueError(f"Invalid ploidy in GRG: {ploidy}")
        if n_haplotypes % ploidy != 0:
            raise ValueError(
                f"GRG has {n_haplotypes} haplotypes, not divisible by ploidy {ploidy}."
            )

        n_individuals = n_haplotypes // ploidy
        chrom = str(chrom)

        def _empty(shape):
            return np.empty(shape, dtype=np.int8)

        if sum_strands:
            if n_mutations == 0:
                calldata_gt = _empty((0, n_individuals))
            elif ploidy == 1:
                mutation_eye = np.eye(n_mutations, dtype=np.float64)
                hap_matrix = pyg.matmul(grg, mutation_eye, pyg.TraversalDirection.DOWN)
                calldata_gt = np.rint(hap_matrix).astype(np.int8, copy=False)
            else:
                mutation_eye = np.eye(n_mutations, dtype=np.float64)
                diploid_matrix = pyg.matmul(
                    grg, mutation_eye, pyg.TraversalDirection.DOWN, by_individual=True
                )
                calldata_gt = np.rint(diploid_matrix).astype(np.int8, copy=False)
            sample_ids = self._sample_ids(n_individuals, sample_prefix)
        else:
            if ploidy != 2:
                raise ValueError(
                    "Phased SNPObject output requires diploid GRGs. "
                    "Use `sum_strands=True` for non-diploid data."
                )
            if n_mutations == 0:
                calldata_gt = _empty((0, n_individuals, ploidy))
            else:
                mutation_eye = np.eye(n_mutations, dtype=np.float64)
                hap_matrix = pyg.matmul(grg, mutation_eye, pyg.TraversalDirection.DOWN)
                hap_matrix = np.rint(hap_matrix).astype(np.int8, copy=False)
                calldata_gt = hap_matrix.reshape(n_mutations, n_individuals, ploidy)
            sample_ids = self._sample_ids(n_individuals, sample_prefix)

        variants_ref = np.empty(n_mutations, dtype=object)
        variants_alt = np.empty(n_mutations, dtype=object)
        variants_pos = np.empty(n_mutations, dtype=np.int64)
        variants_id = np.empty(n_mutations, dtype=object)

        for mut_id in range(n_mutations):
            mutation = grg.get_mutation_by_id(mut_id)
            position = int(round(float(mutation.position)))
            ref = str(mutation.ref_allele) if str(mutation.ref_allele) else "."
            alt = str(mutation.allele) if str(mutation.allele) else "."
            variants_pos[mut_id] = position
            variants_ref[mut_id] = ref
            variants_alt[mut_id] = alt
            variants_id[mut_id] = f"{chrom}:{position}"

        variants_chrom = np.full(n_mutations, chrom, dtype=object)
        variants_filter_pass = np.full(n_mutations, "PASS", dtype=object)
        variants_qual = np.full(n_mutations, np.nan, dtype=np.float32)

        return SNPObject(
            calldata_gt=calldata_gt,
            samples=sample_ids,
            variants_ref=variants_ref,
            variants_alt=variants_alt,
            variants_chrom=variants_chrom,
            variants_filter_pass=variants_filter_pass,
            variants_id=variants_id,
            variants_pos=variants_pos,
            variants_qual=variants_qual,
        )

    def copy(self) -> GRGObject:
        """
        Create and return a copy of `self`.

        Returns:
            **GRGObject:** 
                A new instance of the current object.
        """
        return copy.deepcopy(self)

    def keys(self) -> List[str]:
        """
        Retrieve a list of public attribute names for `self`.

        Returns:
            **list of str:** 
                A list of attribute names, with internal name-mangling removed, 
                for easier reference to public attributes in the instance.
        """
        return [attr.replace('_GRGObject__', '') for attr in vars(self)]

    def to_grg(self, filename: str, 
                     allow_simplify: bool = True):
        pyg.save_grg(self.__calldata_gt, filename, allow_simplify)
