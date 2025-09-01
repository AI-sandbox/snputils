from pathlib import Path
from typing import Union, Optional

from snputils.ancestry.genobj.wide import GlobalAncestryObject


def read_admixture(
    Q_file: Union[str, Path],
    P_file: Optional[Union[str, Path]] = None,
    sample_file: Optional[Union[str, Path]] = None,
    snp_file: Optional[Union[str, Path]] = None,
    ancestry_file: Optional[Union[str, Path]] = None,
) -> 'GlobalAncestryObject':
    """
    Read ADMIXTURE files into a `snputils.ancestry.genobj.GlobalAncestryObject`.

    Args:
        Q_file (str or pathlib.Path):
            Path to the file containing the Q matrix (per-sample ancestry proportions).
            It should end with .Q or .txt.
            The file should use space (' ') as the delimiter.
        P_file (str or pathlib.Path, optional):
            Path to the file containing the P/F matrix (per-ancestry SNP frequencies).
            It should end with .P or .txt.
            The file should use space (' ') as the delimiter. If None, P is not loaded.
        sample_file (str or pathlib.Path, optional):
            Path to the single-column file containing sample identifiers. 
            It should end with .fam or .txt.
            If None, sample identifiers are not loaded.
        snp_file (str or pathlib.Path, optional):
            Path to the single-column file containing SNP identifiers. 
            It should end with .bim or .txt.
            If None, SNP identifiers are not loaded.
        ancestry_file (str or pathlib.Path, optional):
            Path to the single-column file containing ancestry labels for each sample.
            It should end with .map or .txt.
            If None, ancestries are not loaded.

    Returns:
            **GlobalAncestryObject:** 
                A GlobalAncestryObject instance.
    """
    from snputils.ancestry.io.wide.read.admixture import AdmixtureReader

    return AdmixtureReader(
        Q_file=Q_file,
        P_file=P_file,
        sample_file=sample_file,
        snp_file=snp_file,
        ancestry_file=ancestry_file
    ).read()


read_adm = read_admixture
