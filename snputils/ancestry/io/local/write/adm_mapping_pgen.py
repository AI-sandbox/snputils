import gc
import logging
import warnings
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from snputils.ancestry.genobj.local import LocalAncestryObject
from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.write.pgen import PGENWriter

log = logging.getLogger(__name__)


class AdmixtureMappingPGENWriter:
    """
    A writer class for converting and writing local ancestry data into ancestry-specific
    PGEN filesets (.pgen / .psam / .pvar) for admixture mapping with PLINK 2.

    For each ancestry defined in the ancestry map, this writer produces a PLINK 2 PGEN
    fileset whose "genotypes" encode local-ancestry dosage: allele 1 means the window was
    called as that ancestry on that haplotype, allele 0 means it was not.

    The output filesets are named ``<stem>_<ancestry>.pgen`` (plus the matching ``.psam``
    and ``.pvar`` companions).  Pass the prefix (without extension) to
    ``plink2 --pfile`` to run a ``--glm`` linear/logistic regression.
    """

    def __init__(
        self,
        laiobj: LocalAncestryObject,
        file: Union[str, Path],
        ancestry_map: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            laiobj (LocalAncestryObject):
                A LocalAncestryObject instance.
            file (str or pathlib.Path):
                Base path for the output filesets.  The ancestry label and the
                ``.pgen`` / ``.psam`` / ``.pvar`` suffixes are appended automatically,
                so pass a prefix such as ``/out/admixture_mapping`` (with or without
                a ``.pgen`` or ``.vcf`` extension — any extension is stripped).
            ancestry_map (dict of str to str, optional):
                A dictionary mapping ancestry codes (as strings) to ancestry labels.
                Defaults to ``laiobj.ancestry_map`` when not provided.
        """
        self.__laiobj = laiobj
        self.__file = Path(file)
        self.__ancestry_map = ancestry_map

    @property
    def laiobj(self) -> LocalAncestryObject:
        """Retrieve `laiobj`."""
        return self.__laiobj

    @property
    def file(self) -> Path:
        """Retrieve the base output path (suffix stripped)."""
        return self.__file

    @property
    def ancestry_map(self) -> Dict[str, str]:
        """
        Retrieve the ancestry map (code → label).

        Raises:
            ValueError: If no ancestry map is available from either the constructor
                argument or ``laiobj.ancestry_map``.
        """
        if self.__ancestry_map is not None:
            return self.__ancestry_map
        if self.laiobj.ancestry_map is not None:
            return self.laiobj.ancestry_map
        raise ValueError(
            "Ancestry mapping is required but missing. Provide `ancestry_map` "
            "during initialization or ensure `laiobj.ancestry_map` is set."
        )

    def write(self, vzs: bool = False) -> None:
        """
        Write a PGEN fileset for each ancestry type defined in the ancestry map.

        Existing files with the same names are overwritten.

        Genotype encoding:

        - Allele ``1`` on a haplotype: the window is called as the target ancestry.
        - Allele ``0`` on a haplotype: the window is *not* called as the target ancestry.

        The ``.pvar`` INFO column carries ``END=<end_pos>`` when physical positions are
        available in ``laiobj``, mirroring the behaviour of
        :class:`AdmixtureMappingVCFWriter`.

        Args:
            vzs (bool, optional):
                If ``True``, compress the ``.pvar`` file with zstd and save it as
                ``.pvar.zst``.  Defaults to ``False``.
        """
        # Strip any file extension so PGENWriter can append .pgen/.psam/.pvar cleanly
        base = self.__file
        for ext in (".pgen", ".psam", ".pvar", ".pvar.zst", ".vcf", ".bcf"):
            if base.suffix == ext or str(base).endswith(".pvar.zst"):
                base = base.with_suffix("")
                break

        if self.laiobj.physical_pos is not None:
            pos_array = np.array([v1 for v1, _ in self.laiobj.physical_pos], dtype=np.int64)
            info_array = np.array(
                [f"END={v2}" for _, v2 in self.laiobj.physical_pos], dtype=object
            )
        else:
            pos_array = None
            info_array = None

        for key, anc_string in self.ancestry_map.items():
            ancestry = int(key)
            out_prefix = base.with_name(f"{base.stem}_{anc_string}")

            for old_ext in (".pgen", ".psam", ".pvar", ".pvar.zst"):
                candidate = out_prefix.with_suffix(old_ext)
                if candidate.exists():
                    warnings.warn(
                        f"File '{candidate}' already exists and will be overwritten."
                    )

            # Binary ancestry dosage: 1 where this haplotype carries the target ancestry
            match = (self.laiobj.lai == ancestry).view(np.int8)
            n_windows = self.laiobj.lai.shape[0]
            n_haplotypes = self.laiobj.lai.shape[1]
            genotypes = match.reshape(n_windows, n_haplotypes // 2, 2).astype(
                np.int8, copy=False
            )

            # Match FID and IID in .psam so PLINK --pheno / --covar files with FID IID
            # (same convention as benchmark GWAS PGEN) align with loaded sample IDs.
            sid = np.asarray(self.laiobj.samples, dtype=str)
            variants_chrom = self.laiobj.chromosomes
            variants_id = np.array(
                [str(i + 1) for i in range(n_windows)], dtype=object
            )
            variants_ref = np.full(n_windows, "A", dtype="U5")
            variants_alt = np.full(n_windows, "T", dtype="U1")

            snpobj = SNPObject(
                genotypes=genotypes,
                samples=sid,
                sample_fid=sid,
                variants_chrom=variants_chrom,
                variants_id=variants_id,
                variants_ref=variants_ref,
                variants_alt=variants_alt,
                variants_pos=pos_array,
                variants_info=info_array,
            )

            del match, genotypes
            gc.collect()

            log.info(
                f"Writing PGEN fileset for ancestry '{anc_string}' to '{out_prefix}.*'..."
            )
            PGENWriter(snpobj, str(out_prefix)).write(vzs=vzs)
            log.info(
                f"Finished writing PGEN fileset for ancestry '{anc_string}'."
            )
