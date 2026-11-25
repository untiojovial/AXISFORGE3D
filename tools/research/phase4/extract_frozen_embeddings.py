"""Alias for the checkpointed extraction-and-export development runner."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name('run_experiments.py')), run_name='__main__')
