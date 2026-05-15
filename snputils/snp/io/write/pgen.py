import logging
import numpy as np
import polars as pl
import pgenlib as pg
from pathlib import Path
import zstandard as zstd
from typing import Optional, Sequence, Union

from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.write._genotype_encoding import phased_to_flat_alleles, phased_to_hardcalls
from snputils.snp.io.write._plink import coerce_sex_codes

log = logging.getLogger(__name__)


class PGENWriter:
    """
    Writes a genotype object in PGEN format (.pgen, .psam, and .pvar files) in the specified output path.
    """

    def __init__(self, snpobj: SNPObject, filename: str):
        """
        Initializes the PGENWriter instance.

        Args:
            snpobj (SNPObject): The SNPObject containing genotype data to be written.
            filename (str): Base path for the output files (excluding extension).
        """
        self.__snpobj = snpobj
        self.__filename = Path(filename)

    def write(
            self, 
            vzs: bool = False,
            rename_missing_values: bool = True, 
            before: Union[int, float, str] = -1, 
            after: Union[int, float, str] = '.'
        ):
        """
        Writes the SNPObject data to .pgen, .psam, and .pvar files.

        Args:
            vzs (bool, optional): 
                If True, compresses the .pvar file using zstd and saves it as .pvar.zst. Defaults to False.
            rename_missing_values (bool, optional):
                If True, renames potential missing values in `snpobj.calldata_gt` before writing. 
                Defaults to True.
            before (int, float, or str, default=-1): 
                The current representation of missing values in `calldata_gt`. Common values might be -1, '.', or NaN.
                Default is -1.
            after (int, float, or str, default='.'): 
                The value that will replace `before`. Default is '.'.
        """
        file_extensions = (".pgen", ".psam", ".pvar", ".pvar.zst")
        if self.__filename.suffix in file_extensions:
            self.__filename = self.__filename.with_suffix('')

        self.__rename_missing_values = rename_missing_values
        self.__missing_before = before
        self.__missing_after = after

        self.write_pvar(vzs=vzs)
        self.write_psam()
        self.write_pgen()

    def write_pvar(self, vzs: bool = False):
        """
        Writes variant data to the .pvar file.

        Args:
            vzs (bool, optional): If True, compresses the .pvar file using zstd and saves it as .pvar.zst. Defaults to False.
        """
        output_filename = f"{self.__filename}.pvar"
        if vzs:
            output_filename += ".zst"
            log.info(f"Writing to {output_filename} (compressed)")
        else:
            log.info(f"Writing to {output_filename}")

        df = pl.DataFrame(
            {
                "#CHROM": self.__snpobj.variants_chrom,
                "POS": self.__snpobj.variants_pos,
                "ID": self.__snpobj.variants_id,
                "REF": self.__snpobj.variants_ref,
                "ALT": self.__snpobj.variants_alt,
                "QUAL": self._coerce_variant_column(self.__snpobj.variants_qual, self.__snpobj.n_snps),
                "FILTER": self._coerce_variant_column(self.__snpobj.variants_filter_pass, self.__snpobj.n_snps),
                "INFO": self._coerce_variant_column(self.__snpobj.variants_info, self.__snpobj.n_snps),
            }
        )

        # Write the DataFrame to a CSV string
        csv_data = "##fileformat=VCFv4.2\n##source=snputils\n" + df.write_csv(None, separator="\t")

        if vzs:
            # Compress the CSV data using zstd
            cctx = zstd.ZstdCompressor()
            compressed_data = cctx.compress(csv_data.encode('utf-8'))
            with open(output_filename, 'wb') as f:
                f.write(compressed_data)
        else:
            with open(output_filename, 'w') as f:
                f.write(csv_data)

    def write_psam(self):
        """
        Writes sample metadata to the .psam file.
        """
        log.info(f"Writing {self.__filename}.psam")
        columns = {}
        if self.__snpobj.sample_fid is not None:
            columns["#FID"] = self._coerce_sample_column(
                self.__snpobj.sample_fid,
                self.__snpobj.n_samples,
                column_name="sample_fid",
            )
            columns["IID"] = self.__snpobj.samples
        else:
            columns["#IID"] = self.__snpobj.samples
        columns["SEX"] = coerce_sex_codes(self.__snpobj.sample_sex, self.__snpobj.n_samples, missing_code="NA")
        df = pl.DataFrame(columns)
        df.write_csv(f"{self.__filename}.psam", separator="\t")

    @staticmethod
    def _coerce_variant_column(
        values: Optional[Union[np.ndarray, Sequence[Union[str, int, float]]]],
        n_variants: int,
        default: str = ".",
    ) -> np.ndarray:
        if values is None:
            return np.repeat(default, n_variants)
        arr = np.asarray(values)
        if arr.shape[0] != n_variants:
            raise ValueError(f"variant metadata length ({arr.shape[0]}) must match number of variants ({n_variants}).")
        return np.asarray([PGENWriter._missing_to_default(value, default) for value in arr], dtype=object)

    @staticmethod
    def _coerce_sample_column(
        values: Union[np.ndarray, Sequence[Union[str, int, float]]],
        n_samples: int,
        *,
        column_name: str,
    ) -> np.ndarray:
        arr = np.asarray(values)
        if arr.shape[0] != n_samples:
            raise ValueError(f"{column_name} length ({arr.shape[0]}) must match number of samples ({n_samples}).")
        return arr

    @staticmethod
    def _missing_to_default(value, default: str) -> str:
        if value is None:
            return default
        text = str(value)
        if text == "" or text.lower() in {"nan", "none"}:
            return default
        return text

    def write_pgen(self):
        """
        Writes the genotype data to a .pgen file.
        """
        log.info(f"Writing to {self.__filename}.pgen")
        summed_strands = False if self.__snpobj.calldata_gt.ndim == 3 else True
        if not summed_strands:
            num_variants, num_samples, num_alleles = self.__snpobj.calldata_gt.shape
            flat_genotypes = phased_to_flat_alleles(
                self.__snpobj.calldata_gt,
                rename_missing_values=self.__rename_missing_values,
                before=self.__missing_before,
                after=self.__missing_after,
            )
            with pg.PgenWriter(
                filename=f"{self.__filename}.pgen".encode('utf-8'),
                sample_ct=num_samples,
                variant_ct=num_variants,
                hardcall_phase_present=True,
            ) as writer:
                for variant_index in range(num_variants):
                    writer.append_alleles(
                        flat_genotypes[variant_index], all_phased=True
                    )
        else:
            genotypes = phased_to_hardcalls(
                self.__snpobj.calldata_gt,
                rename_missing_values=self.__rename_missing_values,
                before=self.__missing_before,
                after=self.__missing_after,
            )
            num_variants, num_samples = genotypes.shape
            with pg.PgenWriter(
                filename=f"{self.__filename}.pgen".encode('utf-8'),
                sample_ct=num_samples,
                variant_ct=num_variants,
                hardcall_phase_present=False,
            ) as writer:
                for variant_index in range(num_variants):
                    variant_genotypes = genotypes[variant_index].astype(np.int8, copy=False)
                    writer.append_biallelic(np.ascontiguousarray(variant_genotypes))
