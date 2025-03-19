import logging
from typing import Optional, Union, List
import tempfile
from pathlib import Path

from snputils.snp.genobj.snpobj import SNPObject
from snputils.snp.io.read.pgen import PGENReader
from snputils._utils.data_home import get_data_home
from snputils._utils.plink import execute_plink_cmd
from snputils._utils.download import download_url

log = logging.getLogger(__name__)

base_urls = {
    "1kgp": "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/release/20181203_biallelic_SNV"
}

chr_list = list(range(1, 23)) + ["X", "Y"]
chr_urls = {
    "1kgp": {
        str(i): f"ALL.chr{i}.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz" for i in chr_list
    }
}


def available_datasets_list() -> List[str]:
    """
    Get the list of available datasets.
    """
    return list(base_urls.keys())


def load_dataset(
        name: str,
        chromosomes: Union[List[str], List[int], str, int],
        variants_ids: Optional[List[str]] = None,
        sample_ids: Optional[List[str]] = None,
        verbose: bool = True,
        **read_kwargs
) -> SNPObject:
    """
    Load a genome dataset.

    Args:
        name (str): Name of the dataset to load. Call `available_datasets_list()` to get the list of available datasets.
        chromosomes (List[str] | List[int] | str | int): Chromosomes to load.
        variants_ids (List[str]): List of variant IDs to load.
        sample_ids (List[str]): List of sample IDs to load.
        verbose (bool): Whether to show progress.
        **read_kwargs: Keyword arguments to pass to `PGENReader.read()`.

    Returns:
        SNPObject: SNPObject containing the loaded dataset.
    """
    if isinstance(chromosomes, (str, int)):
        chromosomes = [chromosomes]
    chromosomes = [str(chr).lower().replace("chr", "") for chr in chromosomes]

    if variants_ids is not None:
        variants_ids_txt = tempfile.NamedTemporaryFile(mode='w')
        variants_ids_txt.write("\n".join(variants_ids))
        variants_ids_txt.flush()

    if sample_ids is not None:
        sample_ids_txt = tempfile.NamedTemporaryFile(mode='w')
        sample_ids_txt.write("\n".join(sample_ids))
        sample_ids_txt.flush()

    merge_list_txt = tempfile.NamedTemporaryFile(mode='w')

    data_home = get_data_home()

    if name == "1kgp":
        data_path = data_home / name
        data_path.mkdir(parents=True, exist_ok=True)
        for chr in chromosomes:
            chr_path = data_path / chr_urls[name][chr]
            if not Path(chr_path).exists():
                log.info(f"Downloading chromosome {chr}...")
                download_url(f"{base_urls[name]}/{chr_urls[name][chr]}", chr_path, show_progress=verbose)
            else:
                log.info(f"Chromosome {chr} already exists. Skipping download.")

            # Filter and convert to PGEN
            log.info(f"Processing chromosome {chr}...")
            out_file = chr_urls[name][chr].replace('.vcf.gz', '')
            execute_plink_cmd(
                ["--vcf", f"{chr_urls[name][chr]}"]
                + (["--keep", sample_ids_txt.name] if sample_ids is not None else [])
                + (["--extract", variants_ids_txt.name] if variants_ids is not None else [])
                + [
                    "--set-missing-var-ids", "@:#",
                    "--make-pgen",
                    "--out", out_file,
                ], cwd=data_path)
            merge_list_txt.write(f"{out_file}\n")

        if len(chromosomes) > 1:
            # Merge the PGEN files into single PGEN fileset
            log.info("Merging PGEN files...")
            merge_list_txt.flush()
            print(f"Merge list file contents: {open(merge_list_txt.name, 'r').read()}")
            execute_plink_cmd(["--pmerge-list", merge_list_txt.name, "--make-pgen", "--out", "1kgp"],
                              cwd=data_path)
        else:
            # Rename the single PGEN file
            for ext in ["pgen", "psam", "pvar"]:
                Path(data_path / f"{out_file}.{ext}").rename(data_path / f"1kgp.{ext}")

        # Read PGEN fileset with PGENReader into SNPObject
        log.info("Reading PGEN fileset...")
        snpobj = PGENReader(data_path / "1kgp").read(**read_kwargs)
    else:
        raise NotImplementedError(f"Dataset {name} not implemented.")

    if variants_ids is not None:
        variants_ids_txt.close()
    if sample_ids is not None:
        sample_ids_txt.close()
    merge_list_txt.close()

    return snpobj
