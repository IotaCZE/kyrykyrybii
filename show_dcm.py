from pydicom import dcmread
from matplotlib import pyplot as plt
import numpy as np

import sys

import os

def main():
    if len(sys.argv) != 2:
        print(f"Error: Not enough arguments! Number of args:{len(sys.argv)}")
        exit(1)
    file_path = sys.argv[-1]
    ds = dcmread(file_path)
    arr = ds.pixel_array

    plt.figure()
    plt.imshow(arr)
    plt.title(os.path.basename(file_path))
    plt.show(block=True)

if __name__ == "__main__":
    main()