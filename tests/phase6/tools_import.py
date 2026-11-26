"""Imports the Phase VI sweep conditions from the tool script without running it."""
import importlib.util
from pathlib import Path


def sweep_conditions():
    path = Path(__file__).resolve().parents[2]/'tools'/'research'/'phase6'/'sweep.py'
    spec = importlib.util.spec_from_file_location('phase6_sweep', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CONDITIONS
