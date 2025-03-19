from typing import Union
import os
from pathlib import Path
import shutil


def get_data_home(data_home: Union[Path, str] = None) -> Path:
    """
    Get the data home directory, creating it if it doesn't exist.

    Args:
        data_home (Path | str): Path to the data home directory. If not provided, it will be set to the SNPUTILS_DATA
            environment variable if it exists, otherwise to ~/.snputils/data.

    Returns:
        data_home (Path): Path to the data home directory.
    """
    if data_home is None:
        data_home = os.environ.get("SNPUTILS_DATA", Path.home() / ".snputils" / "data")
    data_home = Path(data_home)
    data_home.mkdir(parents=True, exist_ok=True)
    return data_home


def clear_data_home(data_home: Union[Path, str] = None) -> None:
    """"
    Remove the data home directory and all its contents.

    Args:
        data_home (Path | str): Path to the data home directory. If not provided, it will be set to the SNPUTILS_DATA
            environment variable if it exists, otherwise to ~/.snputils/data

    Returns:
        None
    """
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)
