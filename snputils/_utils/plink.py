import platform
import zipfile
import subprocess
from pathlib import Path

from snputils._utils.data_home import get_data_home
from snputils._utils.download import download_url


def download_plink(suppress_output: bool = True) -> Path:
    data_path = get_data_home()
    plink_path = data_path / "plink2"

    if not plink_path.exists():
        system = platform.system()
        is_arm = platform.machine().lower().startswith(('arm', 'aarch'))

        plink_urls = {
            ("Darwin", True): "plink2_mac_arm64_20241114.zip",  # macOS ARM
            ("Linux", False): "plink2_linux_x86_64_20241114.zip",  # Linux x86_64
        }

        try:
            zip_filename = plink_urls[(system, is_arm)]
        except KeyError:
            raise RuntimeError(f"Unsupported platform: {system} {platform.machine()}")

        plink_url = f"https://s3.amazonaws.com/plink2-assets/alpha6/{zip_filename}"

        zip_path = data_path / zip_filename

        download_url(plink_url, zip_path, show_progress=not suppress_output)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_path)
        subprocess.run(["chmod", "+x", plink_path], cwd=str(data_path),
                       stdout=subprocess.DEVNULL if suppress_output else None)

    return plink_path


def execute_plink_cmd(args, cwd=None, suppress_output=True) -> subprocess.CompletedProcess:
    cwd = cwd or get_data_home()
    plink_path = download_plink(suppress_output=suppress_output)  # Download plink if it does not exist
    args = [arg for arg in args if arg is not None]  # Remove None arguments
    return subprocess.run([str(plink_path), *args], cwd=str(cwd),
                          stdout=subprocess.DEVNULL if suppress_output else None)
