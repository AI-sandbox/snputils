import logging
from pathlib import Path
import pandas as pd
import os
from typing import List, Optional, Union

log = logging.getLogger(__name__)

from .base import PhenotypeBaseReader
from snputils.phenotype.genobj import MultiPhenotypeObject


SUPPORTED_EXTENSIONS = ('.xlsx', '.csv', '.tsv', '.txt', '.phe', '.pheno', '.map', '.smap', '.phen')


class MultiPhenReader(PhenotypeBaseReader):
    """
    Reader for multi-phenotype data from file (.xlsx, .csv, .tsv, .txt, .phe, .pheno, .map, .smap, .phen),
    constructing a `MultiPhenotypeObject`.
    """
    def __init__(self, file: Union[str, Path]) -> None:
        """
        Args:
            file (str or pathlib.Path):
                Path to the file containing phenotype data. Accepted formats: .xlsx, .csv, .tsv, .txt, .phe, .pheno, .map, .smap, .phen.
        """
        self.__file = file

    @property
    def file(self) -> Path:
        """
        Retrieve `file`.

        Returns:
            pathlib.Path:
                Path to the file containing phenotype data. Accepted formats: .xlsx, .csv, .tsv, .txt, .phe, .pheno, .map, .smap, .phen.
        """
        return self.__file

    def read(
            self,
            samples_idx: int = 0,
            phen_names: Optional[List] = None,
            sep: str = ',',
            header: int = 0,
            drop: bool = False
        ) -> 'MultiPhenotypeObject':
        """
        Read data from `file` and construct a `MultiPhenotypeObject`.

        Args:
            samples_idx (int, default=0): Index of the column containing sample identifiers.
                Default is 0, assuming the first column contains sample identifiers.
            phen_names (list of str, optional): List of phenotype column names. If provided,
                these columns will be renamed to the specified names.
            sep (str, default=','): The delimiter for separating values in `.csv`, `.tsv`,
                `.txt`, `.phe`, `.pheno`, or `.map` files. Default is ','; use `sep=r'\\s+'` for whitespace-delimited.
            header (int, default=0): Row index to use as the column names. By default,
                uses the first row (`header=0`). Set to `None` if column names are provided
                explicitly.
            drop (bool, default=False): If True, removes columns not listed in `phen_names`
                (except the samples column).

        Returns:
            MultiPhenotypeObject:
                A multi-phenotype object instance.
        """
        file_extension = os.path.splitext(self.file)[1]

        log.info(f"Reading '{file_extension}' file from '{self.file}'...")

        if file_extension == '.xlsx':
            phen_df = pd.read_excel(self.file, header=0, index_col=None)
        elif file_extension == '.csv':
            phen_df = pd.read_csv(self.file, sep=sep, header=header)
        elif file_extension in ['.map', '.smap']:
            phen_df = pd.read_csv(self.file, sep=sep, header=header)
        elif file_extension == '.tsv':
            phen_df = pd.read_csv(self.file, sep='\t')
        elif file_extension in ['.txt', '.phe', '.pheno']:
            phen_df = pd.read_csv(self.file, sep=r'\s+', header=header)
        elif file_extension == '.phen':
            with open(self.file, 'r') as f:
                contents = f.readlines()
            phen_dict = {line.split()[0]: line.split()[1].strip() for line in contents[1:]}
            phen_df = pd.DataFrame({'samples': list(phen_dict.keys()), 'phenotype': list(phen_dict.values())})
        else:
            raise ValueError(
                f"Unsupported file extension {file_extension}. Supported extensions: {SUPPORTED_EXTENSIONS}."
            )

        phen_df.rename(columns={phen_df.columns[samples_idx]: 'samples'}, inplace=True)

        if samples_idx != 0:
            cols = ['samples'] + [col for col in phen_df.columns if col != 'samples']
            phen_df = phen_df[cols]

        if phen_names is not None:
            if drop:
                non_phen_columns = list(set(phen_df.columns) - set(['samples']+phen_names))
                phen_df = phen_df.drop(non_phen_columns, axis=1)

            phenotype_col_count = phen_df.shape[1] - 1
            if phenotype_col_count == len(phen_names):
                phen_df.columns.values[1:] = phen_names
            else:
                raise ValueError(f"Mismatch between number of phenotype columns ({phenotype_col_count}) "
                                 f"and length of `phen_names` ({len(phen_names)}).")

        return MultiPhenotypeObject(phen_df=phen_df)

PhenotypeBaseReader.register(MultiPhenReader)
