from typing import Any
from pathlib import Path
import pickle

def dump_pickle(obj: Any, path: Path):
    with open(path, 'w+b') as f:
        pickle.dump(obj, f)


def load_pickle(path: Path):
    with open(path, 'rb') as f:
        return pickle.load(f)