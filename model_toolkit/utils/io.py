from typing import Any
from pathlib import Path
import pickle
import os, urllib


def dump_pickle(obj: Any, path: Path):
    with open(path, 'w+b') as f:
        pickle.dump(obj, f)


def load_pickle(path: Path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def download_file(url, output_dir='', output_filename=None):
    """Downloads a file to the filesystem.
    
    Args:
        url: a `str`, a URL pointing to the file to download
        output_dir: a `str` or `Path`, the path to the download directory
        output_filename: a `str`, what to name the downloaded file.
            If this value is 'None', the filename is extracted from the URL.
            Default is 'None'.
    """
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = output_filename or url.split('/')[-1]
    output_filepath = output_dir / output_filename
    
    bytestream = urllib.request.urlopen(url)
    with open(output_filepath, 'w+b') as output_file:
        output_file.write(bytestream.read())
