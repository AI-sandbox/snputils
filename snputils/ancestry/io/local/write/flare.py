import gzip
import logging
import warnings
from pathlib import Path
from typing import List, Optional, TextIO, Union

import numpy as np

from .base import LAIBaseWriter
from snputils.ancestry.genobj.local import LocalAncestryObject
from snputils.snp.genobj.snpobj import SNPObject

log = logging.getLogger(__name__)


def _open_text(path: Path, mode: str = "wt") -> TextIO:
    if path.name.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


class FLAREWriter(LAIBaseWriter):
    """
    Writer for FLARE-style local ancestry VCF files.

    FLARE output includes the phased input genotypes and variant metadata, so a
    genotype source is required. Pass either a :class:`~snputils.snp.genobj.SNPObject`
    or a genotype file readable by :func:`snputils.read_snp`. Local ancestry hard
    calls are written to the FLARE `AN1` and `AN2` FORMAT subfields.
    """

    def __init__(
        self,
        laiobj: LocalAncestryObject,
        file: Union[str, Path],
        *,
        snpobj: Optional[SNPObject] = None,
        genotype_file: Optional[Union[str, Path]] = None,
    ) -> None:
        if snpobj is not None and genotype_file is not None:
            raise ValueError("Pass only one of `snpobj` or `genotype_file`.")
        self.__laiobj = laiobj
        self.__file = Path(file)
        self.__snpobj = snpobj
        self.__genotype_file = None if genotype_file is None else Path(genotype_file)

    @property
    def laiobj(self) -> LocalAncestryObject:
        return self.__laiobj

    @property
    def file(self) -> Path:
        return self.__file

    @file.setter
    def file(self, x: Union[str, Path]):
        self.__file = Path(x)

    @property
    def snpobj(self) -> Optional[SNPObject]:
        return self.__snpobj

    @property
    def genotype_file(self) -> Optional[Path]:
        return self.__genotype_file

    def _resolve_samples(self) -> List[str]:
        if self.laiobj.samples is not None:
            return [str(sample) for sample in self.laiobj.samples]
        if self.laiobj.haplotypes is not None:
            return [str(hap).rsplit(".", 1)[0] for hap in self.laiobj.haplotypes[0::2]]
        return [f"sample_{i}" for i in range(self.laiobj.n_samples)]

    def _ensure_path(self) -> None:
        valid_suffixes = (
            (".anc", ".vcf", ".gz"),
            (".anc", ".vcf"),
            (".vcf", ".gz"),
            (".vcf",),
        )
        suffixes = tuple(s.lower() for s in self.file.suffixes)
        if not any(suffixes[-len(valid):] == valid for valid in valid_suffixes if len(suffixes) >= len(valid)):
            self.file = self.file.with_name(self.file.name + ".anc.vcf.gz")

    def _resolve_snpobj(self) -> SNPObject:
        if self.snpobj is not None:
            return self.snpobj
        if self.genotype_file is None:
            raise ValueError(
                "FLAREWriter requires genotype data. Pass `snpobj=` or `genotype_file=` "
                "so GT, ID, REF, ALT, QUAL, FILTER, and INFO can be written from real input data."
            )

        from snputils.snp.io.read import read_snp

        return read_snp(self.genotype_file, sum_strands=False)

    def _required_variant_array(self, snpobj: SNPObject, attr: str, n_variants: int) -> np.ndarray:
        values = getattr(snpobj, attr)
        if values is None:
            raise ValueError(f"SNPObject.{attr} is required to write FLARE output.")
        values = np.asarray(values)
        if values.size != n_variants:
            raise ValueError(
                f"SNPObject.{attr} length ({values.size}) must match "
                f"SNPObject.genotypes variant count ({n_variants})."
            )
        return values

    def _optional_variant_array(self, snpobj: SNPObject, attr: str, default: str, n_variants: int) -> np.ndarray:
        values = getattr(snpobj, attr)
        if values is None:
            return np.full(n_variants, default, dtype=object)
        values = np.asarray(values)
        if values.size == 0:
            return np.full(n_variants, default, dtype=object)
        if values.size != n_variants:
            raise ValueError(
                f"SNPObject.{attr} length ({values.size}) must match "
                f"SNPObject.genotypes variant count ({n_variants})."
            )
        return values

    def _match_lai_variant_indices(self, snpobj: SNPObject, samples: List[str]) -> np.ndarray:
        if snpobj.samples is not None:
            snp_samples = [str(sample) for sample in np.asarray(snpobj.samples).tolist()]
            if snp_samples != samples:
                raise ValueError("SNPObject samples must match LocalAncestryObject samples and order.")

        gt = np.asarray(snpobj.genotypes)
        if gt.ndim != 3 or gt.shape[2] != 2:
            raise ValueError("SNPObject.genotypes must have shape (n_variants, n_samples, 2).")
        if gt.shape[1] != len(samples):
            raise ValueError("SNPObject sample count must match LocalAncestryObject.n_samples.")

        n_variants = int(gt.shape[0])
        snp_chrom = self._required_variant_array(snpobj, "variants_chrom", n_variants).astype(str)
        snp_pos = self._required_variant_array(snpobj, "variants_pos", n_variants).astype(np.int64, copy=False)

        if self.laiobj.chromosomes is None or self.laiobj.physical_pos is None:
            if n_variants != self.laiobj.n_windows:
                raise ValueError(
                    "LocalAncestryObject chromosomes and physical_pos are required when "
                    "the genotype source contains extra variants."
                )
            return np.arange(n_variants, dtype=np.int64)

        lai_chrom = np.asarray(self.laiobj.chromosomes).astype(str)
        lai_pos = np.asarray(self.laiobj.physical_pos)[:, 0].astype(np.int64, copy=False)
        if lai_chrom.size != self.laiobj.n_windows or lai_pos.size != self.laiobj.n_windows:
            raise ValueError("LocalAncestryObject chromosome/position metadata must match n_windows.")

        if n_variants == self.laiobj.n_windows and np.array_equal(lai_chrom, snp_chrom) and np.array_equal(lai_pos, snp_pos):
            return np.arange(n_variants, dtype=np.int64)

        variant_lookup: dict[tuple[str, int], int] = {}
        duplicates: set[tuple[str, int]] = set()
        for idx, key in enumerate(zip(snp_chrom.tolist(), snp_pos.astype(int).tolist())):
            if key in variant_lookup:
                duplicates.add(key)
            else:
                variant_lookup[key] = idx
        if duplicates:
            example = sorted(duplicates)[0]
            raise ValueError(
                "Cannot align FLARE output to genotype source because the genotype "
                f"source has duplicate CHROM/POS records, e.g. {example[0]}:{example[1]}."
            )

        matched = np.empty(self.laiobj.n_windows, dtype=np.int64)
        missing: list[tuple[str, int]] = []
        for out_idx, key in enumerate(zip(lai_chrom.tolist(), lai_pos.astype(int).tolist())):
            source_idx = variant_lookup.get(key)
            if source_idx is None:
                missing.append(key)
            else:
                matched[out_idx] = source_idx
        if missing:
            example = missing[0]
            raise ValueError(
                "Genotype source does not contain all LocalAncestryObject CHROM/POS records; "
                f"first missing record is {example[0]}:{example[1]}."
            )
        return matched

    def _format_gt(self, gt_pair: np.ndarray) -> str:
        left = "." if int(gt_pair[0]) < 0 else str(int(gt_pair[0]))
        right = "." if int(gt_pair[1]) < 0 else str(int(gt_pair[1]))
        return f"{left}|{right}"

    def _format_filter(self, value: object) -> str:
        if isinstance(value, (bool, np.bool_)):
            return "PASS" if bool(value) else "."
        text = str(value)
        return "." if text == "" or text.lower() == "nan" else text

    def _format_scalar(self, value: object) -> str:
        if value is None:
            return "."
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            return "."
        text = str(value)
        return "." if text == "" or text.lower() == "nan" else text

    def write(self) -> None:
        if self.laiobj.lai.ndim != 2 or self.laiobj.lai.shape[1] % 2 != 0:
            raise ValueError("LocalAncestryObject.lai must have shape (n_windows, 2 * n_samples).")

        self._ensure_path()
        if self.file.exists():
            warnings.warn(f"File '{self.file}' already exists and will be overwritten.")

        snpobj = self._resolve_snpobj()
        n_windows = self.laiobj.n_windows
        samples = self._resolve_samples()
        if len(samples) != self.laiobj.n_samples:
            raise ValueError("Number of sample identifiers does not match LAI haplotype columns.")

        variant_indices = self._match_lai_variant_indices(snpobj, samples)
        n_variants = int(np.asarray(snpobj.genotypes).shape[0])

        chromosomes = self._required_variant_array(snpobj, "variants_chrom", n_variants)[variant_indices]
        positions = self._required_variant_array(snpobj, "variants_pos", n_variants)[variant_indices].astype(np.int64, copy=False)
        variant_ids = self._required_variant_array(snpobj, "variants_id", n_variants)[variant_indices]
        refs = self._required_variant_array(snpobj, "variants_ref", n_variants)[variant_indices]
        alts = self._required_variant_array(snpobj, "variants_alt", n_variants)[variant_indices]
        quals = self._optional_variant_array(snpobj, "variants_qual", ".", n_variants)[variant_indices]
        filters = self._optional_variant_array(snpobj, "variants_filter_pass", ".", n_variants)[variant_indices]
        infos = self._optional_variant_array(snpobj, "variants_info", ".", n_variants)[variant_indices]
        genotypes = np.asarray(snpobj.genotypes)

        ancestry_map = self.laiobj.ancestry_map
        if ancestry_map is None:
            codes = sorted(int(code) for code in np.unique(self.laiobj.lai))
            ancestry_map = {str(code): f"ANC{code}" for code in codes}

        ancestry_items = sorted(((int(code), str(label)) for code, label in ancestry_map.items()), key=lambda x: x[0])

        log.info("Writing FLARE local ancestry VCF to '%s'...", self.file)
        with _open_text(self.file, "wt") as handle:
            handle.write("##fileformat=VCFv4.1\n")
            handle.write('##source="snputils"\n')
            handle.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
            handle.write('##FORMAT=<ID=AN1,Number=1,Type=Integer,Description="Ancestry of first haplotype">\n')
            handle.write('##FORMAT=<ID=AN2,Number=1,Type=Integer,Description="Ancestry of second haplotype">\n')
            ancestry_line = ",".join(f"{label}={code}" for code, label in ancestry_items)
            handle.write(f"##ANCESTRY=<{ancestry_line}>\n")
            header = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"] + samples
            handle.write("\t".join(header) + "\n")

            lai = np.asarray(self.laiobj.lai)
            for row_idx in range(n_windows):
                row = [
                    str(chromosomes[row_idx]),
                    str(int(positions[row_idx])),
                    str(variant_ids[row_idx]),
                    str(refs[row_idx]),
                    str(alts[row_idx]),
                    self._format_scalar(quals[row_idx]),
                    self._format_filter(filters[row_idx]),
                    self._format_scalar(infos[row_idx]),
                    "GT:AN1:AN2",
                ]
                for sample_idx in range(len(samples)):
                    an1 = int(lai[row_idx, 2 * sample_idx])
                    an2 = int(lai[row_idx, 2 * sample_idx + 1])
                    row.append(f"{self._format_gt(genotypes[variant_indices[row_idx], sample_idx])}:{an1}:{an2}")
                handle.write("\t".join(row) + "\n")

        log.info("Finished writing FLARE local ancestry VCF to '%s'.", self.file)


LAIBaseWriter.register(FLAREWriter)
