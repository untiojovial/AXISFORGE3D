"""Fetch the pinned OpenCLIP checkpoint from its publisher and verify the bytes."""
import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen

from .vision_openclip import CHECKPOINT_SHA256

NAME='vit_b_32-quickgelu-laion400m_e31-d867053b.pt'
URL='https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/'+NAME


def fetch(directory):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    destination=directory/NAME
    def verified(path):
        with path.open('rb') as stream:
            return path.stat().st_size==605159845 and hashlib.file_digest(stream,'sha256').hexdigest()==CHECKPOINT_SHA256
    if destination.exists():
        if verified(destination):return destination
        raise ValueError('local checkpoint differs from pinned bytes')
    with tempfile.NamedTemporaryFile(dir=directory,delete=False) as partial:
        path=Path(partial.name)
        try:
            with urlopen(URL,timeout=90) as response:
                shutil.copyfileobj(response,partial)
            partial.flush()
            if not verified(path):raise ValueError('downloaded checkpoint failed verification')
            os.replace(path,destination)
        finally:
            path.unlink(missing_ok=True)
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    print(fetch(parser.parse_args().directory))
