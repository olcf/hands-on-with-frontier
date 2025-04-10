from setuptools import find_packages, setup
import os

name = "quantum_linear_solvers"
version = "0.1.0"    # pinned to Qiskit 1.x
description = (
    "Quantum linear solvers package"
)

with open("README.md") as f:
    long_description = f.read()

setup(
    name=name,
    version=version,
    description=description,
    long_description=long_description,
    use_scm_version=False,
    include_package_data=False,
)
