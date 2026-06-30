import inspect
import os
import pathlib
import subprocess
import urllib.request
import zipfile
import platform

import numpy as np
import pytest
from bgen import BgenReader as RawBgenReader

import snputils
from snputils.snp.io.read import BEDReader, BGENReader, PGENReader, VCFReader


def _bgen_is_readable(path: pathlib.Path) -> bool:
    if not path.exists():
        return False
    try:
        with RawBgenReader(str(path), delay_parsing=True) as bfile:
            next(iter(bfile)).probabilities
        return True
    except Exception:
        return False


def _remove_bgen_outputs(prefix: pathlib.Path) -> None:
    for suffix in (".bgen", ".bgen.bgi", ".sample", ".log"):
        path = prefix.with_suffix(suffix)
        if path.exists():
            path.unlink()


def _generate_bgen_from_vcf(data_path: pathlib.Path, subset_vcf_file: pathlib.Path) -> None:
    bgen_path = data_path / "bgen"
    os.makedirs(bgen_path, exist_ok=True)
    prefix = bgen_path / "subset"
    bgen_file = prefix.with_suffix(".bgen")
    if _bgen_is_readable(bgen_file):
        return

    # PLINK2's bgen-1.2 export is the target fixture format. Some alpha builds emit
    # bgen-1.2 files that neither PLINK2 nor bgen can decompress, so validate the
    # result and fall back to bgen-1.3 to keep the I/O tests exercising real BGEN.
    export_attempts = [
        ("bgen-1.2", ["bgen-1.2", "ref-first", "bits=8"]),
        ("bgen-1.3", ["bgen-1.3", "ref-first", "bits=8"]),
    ]
    for label, export_args in export_attempts:
        _remove_bgen_outputs(prefix)
        print(f"Generating BGEN format with PLINK2 --export {label}...")
        subprocess.run(
            [
                "./plink2",
                "--vcf",
                subset_vcf_file,
                "--export",
                *export_args,
                "--out",
                "bgen/subset",
            ],
            cwd=str(data_path),
            check=True,
        )
        if _bgen_is_readable(bgen_file):
            return

    raise RuntimeError("PLINK2 failed to produce a readable BGEN fixture.")


def _generate_bcf_from_vcf(data_path: pathlib.Path, subset_vcf_file: pathlib.Path) -> None:
    bcf_path = data_path / "bcf"
    os.makedirs(bcf_path, exist_ok=True)
    prefix = bcf_path / "subset"
    bcf_file = prefix.with_suffix(".bcf")
    if bcf_file.exists():
        return

    print("Generating BCF format with PLINK2 --export bcf...")
    subprocess.run(
        [
            "./plink2",
            "--vcf",
            subset_vcf_file,
            "--export",
            "bcf",
            "--out",
            "bcf/subset",
        ],
        cwd=str(data_path),
        check=True,
    )


@pytest.fixture(scope="module")
def data_path():
    module_path = pathlib.Path(inspect.getfile(snputils)).parent.parent
    data_path = module_path / "data"
    os.makedirs(data_path, exist_ok=True)

    system = platform.system()
    is_arm = platform.machine().lower().startswith(('arm', 'aarch'))

    plink_urls = {
        ("Darwin", True): "plink2_mac_arm64_20260504.zip",  # macOS ARM
        ("Linux", False): "plink2_linux_x86_64_20260504.zip",  # Linux x86_64
    }

    try:
        plink_filename = plink_urls[(system, is_arm)]
    except KeyError:
        raise RuntimeError(f"Unsupported platform: {system} {platform.machine()}")

    files_urls = {
        plink_filename: f"https://s3.amazonaws.com/plink2-assets/alpha7/{plink_filename}",
        "vcf/ALL.chr22.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz":
        "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/release/20181203_biallelic_SNV/ALL.chr22.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz",
    }

    # Check and download each file if it does not exist
    for file_name, url in files_urls.items():
        file_path = data_path / file_name
        dir_path = file_path.parent if file_path.is_file() else file_path
        if not dir_path.exists():
            os.makedirs(file_path.parent, exist_ok=True)
        if not file_path.exists():
            print(f"Downloading {file_name} to {data_path}. This may take a while...")
            urllib.request.urlretrieve(url, file_path)
            if file_path.suffix == ".zip":
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(data_path)
                subprocess.run(
                    ["chmod", "+x", data_path / "plink2"], cwd=str(data_path)
                )

        if file_path.suffixes[-2:] == ['.vcf', '.gz']:
            # Subset sample files
            subset_file = data_path / "subset.txt"
            if not subset_file.exists():
                print("Generating subset file...")
                four_sample_ids = ["HG00096", "HG00097", "HG00099", "HG00100"]
                with open(subset_file, "w") as file:
                    file.write("\n".join(four_sample_ids))

            plink_vcf_out = "vcf/subset"
            subset_vcf_file = data_path / (plink_vcf_out + ".vcf")
            if not subset_vcf_file.exists():
                print("Generating subset VCF...")
                subprocess.run(
                    [
                        "./plink2",
                        "--vcf",
                        data_path / file_name,
                        "--keep",
                        "subset.txt",
                        "--recode",
                        "vcf",
                        "--set-missing-var-ids",
                        "@:#",
                        "--out",
                        plink_vcf_out,
                    ],
                    cwd=str(data_path),
                )

    # Generate bed and pgen formats
    for fmt in ["bed", "pgen", "pgen_zst"]:
        fmt_path = data_path / fmt
        os.makedirs(fmt_path, exist_ok=True)
        fmt_file = fmt_path / "subset"
        if not fmt_file.exists():
            print(f"Generating {fmt} format...")
            make_fmt = "--make-pgen vzs" if fmt == "pgen_zst" else f"--make-{fmt}"
            subprocess.run(
                [
                    "./plink2",
                    "--vcf",
                    subset_vcf_file,
                    *make_fmt.split(),
                    "--out",
                    fmt_file,
                ],
                cwd=str(data_path),
            )

    _generate_bgen_from_vcf(data_path, subset_vcf_file)
    _generate_bcf_from_vcf(data_path, subset_vcf_file)

    return str(data_path)


@pytest.fixture(scope="module")
def snpobj_vcf(data_path):
    return VCFReader(data_path + "/vcf/subset.vcf").read(sum_strands=False)


@pytest.fixture(scope="module")
def snpobj_bed(data_path):
    return BEDReader(data_path + "/bed/subset").read(sum_strands=True)


@pytest.fixture(scope="module")
def snpobj_bgen(data_path):
    return BGENReader(data_path + "/bgen/subset.bgen").read(variant_idxs=np.arange(100))


@pytest.fixture(scope="module")
def snpobj_bcf(data_path):
    from snputils.snp.io.read.bcf import BCFReader

    return BCFReader(data_path + "/bcf/subset.bcf").read(sum_strands=False, variant_idxs=np.arange(100))


@pytest.fixture(scope="module")
def snpobj_pgen(data_path):
    return PGENReader(data_path + "/pgen/subset").read(sum_strands=False)
