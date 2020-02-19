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


def remove_url_args(url):
    return url.split('?')[0]

def download_file(url, output_dir='', output_filename=None, overwrite=False):
    """Downloads a file to the filesystem.
    
    Args:
        url: a `str`, a URL pointing to the file to download
        output_dir: a `str` or `Path`, the path to the download directory
        output_filename: a `str`, what to name the downloaded file.
            If this value is 'None', the filename is extracted from the URL.
            Default is 'None'.
        overwrite: a `bool`, indicates whether to overwrite the file if 
            it already exists, default is 'False'
    """
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = output_filename or remove_url_args(url).split('/')[-1]
    output_filepath = output_dir / output_filename
    
    if output_filepath.exists() and not overwrite:
        print(f"File {str(output_filepath)} already exists, nothing to do.")
    else:
        bytestream = urllib.request.urlopen(url)
        with open(output_filepath, 'w+b') as output_file:
            output_file.write(bytestream.read())
            print(f"File saved to {str(output_filepath)}.")
    return str(output_filepath.resolve())
