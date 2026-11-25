"""Compare the published table and printed figure labels with the supplied thesis."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def compare(pdf, manifest):
    data = json.loads(manifest.read_text())
    if hashlib.sha256(pdf.read_bytes()).hexdigest() != data["source_sha256"]:
        raise ValueError("unexpected thesis edition")
    pages = subprocess.check_output(
        ["pdftotext", "-layout", str(pdf), "-"], text=True
    ).split("\f")
    table_page = next(
        page for page in pages if "Tabla 6.1: Maximum Grasp Quality" in page
    )
    values = {
        name: float(value)
        for name, value in re.findall(
            r"^\s*(Boat|Bottle|Duck|Charger|Mouse|Ice Cream|Trophy|Jar)\s+(0\.\d+)\s*$",
            table_page,
            re.MULTILINE,
        )
    }
    if values != data["reported_maximum_quality"]:
        raise ValueError("table 6.1 disagrees with transcription")
    for number, expected in [
        (3, data["first_sequence"][2:]),
        (4, data["second_sequence_positions_3_through_8"]),
    ]:
        page = next(p for p in pages if f"Figure 6.{number}: Optimization results" in p)
        labels = re.findall(
            r"[3-8]º\.\s+(Ice Cream|Jar|Mouse|Charger|Duck|Trophy)", page
        )
        if labels != expected:
            raise ValueError(f"figure 6.{number} labels differ")
    return {
        "printed_table": "6.1",
        "object_rows": len(values),
        "checked_figure_positions": 12,
        "maximum_object": max(values, key=values.get),
        "reported_maximum": max(values.values()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/legacy_tfm/published.json")
    )
    args = parser.parse_args()
    print(json.dumps(compare(args.pdf, args.manifest), indent=2))
