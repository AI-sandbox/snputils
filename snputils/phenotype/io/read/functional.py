from pathlib import Path
from typing import Optional, Union

from snputils.phenotype.genobj import PhenotypeObject


def read_pheno(
    file: Union[str, Path],
    col: Optional[str] = None,
    *,
    quantitative: Optional[bool] = None,
) -> PhenotypeObject:
    """Read a phenotype file into a :class:`~snputils.PhenotypeObject`.

    Args:
        file: Path to a headered phenotype table (``.txt``, ``.phe``, ``.pheno``, …).
        col: Phenotype column to load (header name, with or without ``#``). If the
            file has a single phenotype column, this may be omitted.
        quantitative: If set, force quantitative (linear) or binary (logistic) mode.
            When ``None``, inferred from the column values.
    """
    from .phenotypeReader import PhenotypeReader

    return PhenotypeReader(file).read(phenotype_col=col, quantitative=quantitative)
