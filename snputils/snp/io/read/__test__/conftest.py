import hashlib
import inspect
import os
import pathlib
import platform
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from http.client import HTTPException

import numpy as np
import pytest

import snputils
from snputils.snp.io.read import BEDReader, BGENReader, PGENReader, VCFReader


_CHR22_FILENAME = "ALL.chr22.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz"
_CHR22_SHA256 = "b428192af4f02507585c3775e59251974c71a968daa895a9a47acb337140614c"
_CHR22_URLS = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/"
    f"release/20181203_biallelic_SNV/{_CHR22_FILENAME}",
    f"https://ftp.mi.fu-berlin.de/pub/rmaerker/just_bench/v0.0.1/{_CHR22_FILENAME}",
)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _get_chr22_vcf(directory: pathlib.Path) -> pathlib.Path:
    path = directory / _CHR22_FILENAME
    if path.is_file() and _sha256(path) == _CHR22_SHA256:
        return path

    failures = []
    with tempfile.TemporaryDirectory(dir=directory) as tmp:
        download = pathlib.Path(tmp) / _CHR22_FILENAME
        for url in _CHR22_URLS:
            print(f"Downloading chromosome 22 from {url}")
            try:
                with urllib.request.urlopen(url, timeout=60) as source, download.open("wb") as target:
                    shutil.copyfileobj(source, target)
            except (OSError, HTTPException) as exc:
                failures.append(f"{url}: {exc}")
                continue
            if _sha256(download) == _CHR22_SHA256:
                download.replace(path)
                return path
            failures.append(f"{url}: SHA-256 mismatch")
    raise RuntimeError("Could not obtain the chromosome 22 test VCF:\n" + "\n".join(failures))


def _bgen_is_readable(path: pathlib.Path) -> bool:
    if not path.exists():
        return False
    try:
        BGENReader(path).read(fields=["GP"], variant_idxs=[0])
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
    # bgen-1.2 files that neither PLINK2 nor snputils can decompress, so validate the
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

    archive = data_path / plink_filename
    if not archive.exists():
        url = f"https://s3.amazonaws.com/plink2-assets/alpha7/{plink_filename}"
        urllib.request.urlretrieve(url, archive)
    if not (data_path / "plink2").is_file():
        with zipfile.ZipFile(archive) as zip_ref:
            zip_ref.extractall(data_path)
        (data_path / "plink2").chmod(0o755)

    vcf_dir = data_path / "vcf"
    vcf_dir.mkdir(exist_ok=True)
    subset_vcf_file = vcf_dir / "subset.vcf"
    if not subset_vcf_file.exists():
        source_vcf = _get_chr22_vcf(vcf_dir)
        subset_file = data_path / "subset.txt"
        if not subset_file.exists():
            subset_file.write_text("HG00096\nHG00097\nHG00099\nHG00100\n")
        print("Generating subset VCF...")
        with tempfile.TemporaryDirectory(dir=vcf_dir) as tmp:
            prefix = pathlib.Path(tmp) / "subset"
            subprocess.run(
                [
                    "./plink2", "--vcf", source_vcf, "--keep", subset_file,
                    "--recode", "vcf", "--set-missing-var-ids", "@:#", "--out", prefix,
                ],
                cwd=data_path,
                check=True,
            )
            prefix.with_suffix(".vcf").replace(subset_vcf_file)

    # Generate bed and pgen formats
    for fmt in ["bed", "pgen", "pgen_zst"]:
        fmt_path = data_path / fmt
        os.makedirs(fmt_path, exist_ok=True)
        fmt_file = fmt_path / "subset"
        required_suffixes = {
            "bed": (".bed", ".bim", ".fam"),
            "pgen": (".pgen", ".pvar", ".psam"),
            "pgen_zst": (".pgen", ".pvar.zst", ".psam"),
        }[fmt]
        if not all(fmt_file.with_suffix(suffix).exists() for suffix in required_suffixes):
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
                check=True,
            )

    _generate_bgen_from_vcf(data_path, subset_vcf_file)
    _generate_bcf_from_vcf(data_path, subset_vcf_file)

    return str(data_path)


@pytest.fixture(scope="module")
def snpobj_vcf(data_path):
    return VCFReader(data_path + "/vcf/subset.vcf").read(genotype_mode="phased")


@pytest.fixture(scope="module")
def snpobj_bed(data_path):
    return BEDReader(data_path + "/bed/subset").read(genotype_mode="dosage")


@pytest.fixture(scope="module")
def snpobj_bgen(data_path):
    return BGENReader(data_path + "/bgen/subset.bgen").read(variant_idxs=np.arange(100))


@pytest.fixture(scope="module")
def snpobj_bcf(data_path):
    from snputils.snp.io.read.bcf import BCFReader

    return BCFReader(data_path + "/bcf/subset.bcf").read(genotype_mode="phased", variant_idxs=np.arange(100))


@pytest.fixture(scope="module")
def snpobj_pgen(data_path):
    return PGENReader(data_path + "/pgen/subset").read(genotype_mode="phased")
