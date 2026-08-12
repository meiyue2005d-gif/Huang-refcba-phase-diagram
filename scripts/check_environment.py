#!/usr/bin/env python3

from __future__ import annotations

import platform
import sys

import matplotlib
import networkx
import numpy
import openmm
import pandas
import scipy
import yaml
from openmm import Platform


def main() -> None:
    print("=" * 72)
    print("Huang coarse-grained colloidal MD environment")
    print("=" * 72)

    print(f"Operating system : {platform.platform()}")
    print(f"Python           : {sys.version.split()[0]}")
    print(f"OpenMM           : {openmm.__version__}")
    print(f"NumPy            : {numpy.__version__}")
    print(f"SciPy            : {scipy.__version__}")
    print(f"Pandas           : {pandas.__version__}")
    print(f"Matplotlib       : {matplotlib.__version__}")
    print(f"NetworkX         : {networkx.__version__}")
    print(f"PyYAML           : {yaml.__version__}")

    print("\nAvailable OpenMM platforms:")

    platform_names: list[str] = []

    for index in range(Platform.getNumPlatforms()):
        current = Platform.getPlatform(index)
        name = current.getName()
        platform_names.append(name)

        print(
            f"  [{index}] {name:<12s} "
            f"speed={current.getSpeed():.1f}"
        )

    print("\nEnvironment assessment:")

    if "CUDA" in platform_names:
        print("  PASS: CUDA platform is available.")
        print("  The production simulations can run on the GPU.")
    elif "OpenCL" in platform_names:
        print("  WARNING: CUDA was not found, but OpenCL is available.")
    else:
        print("  WARNING: only CPU/Reference platforms are available.")

    print("\nEnvironment check completed.")


if __name__ == "__main__":
    main()
