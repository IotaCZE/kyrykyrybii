import pydicom
from pydicom import dcmread
from matplotlib import pyplot as plt
import numpy as np

import sys

import os

def list_beams(ds: pydicom.Dataset) -> str:
    """Summarizes the RTPLAN beam information in the dataset."""
    lines = [f"{'Beam name':^13s} {'Number':^8s} {'Gantry':^8s} {'SSD (cm)':^11s}"]
    for beam in ds.BeamSequence:
        cp0 = beam.ControlPointSequence[0]
        ssd = float(cp0.SourceToSurfaceDistance / 10)
        lines.append(
            f"{beam.BeamName:^13s} {beam.BeamNumber:8d} {cp0.GantryAngle:8.1f} {ssd:8.1f}"
        )
    return "\n".join(lines)

def main():
    path = sys.argv[-1]
    ds = pydicom.dcmread(path)
    print(list_beams(ds))

if __name__ == "__main__":
    main()