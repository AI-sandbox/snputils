import urllib.request
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """
    tqdm progress bar for download progress.
    """
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        Update the progress bar.

        Args:
            b (int): Number of blocks transferred.
            bsize (int): Size of each block.
            tsize (int): Total size of the file.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path, show_progress=False):
    """
    Download a file from a URL.

    Args:
        url (str): URL of the file to download.
        output_path (str): Path to save the downloaded file.
        show_progress (bool): Whether to show download progress with a tqdm progress bar.
    """
    if show_progress:
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=output_path,
                                       reporthook=t.update_to)
    else:
        urllib.request.urlretrieve(url, filename=output_path)
