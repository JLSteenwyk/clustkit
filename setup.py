from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def _read_version():
    version_ns = {}
    with open(path.join(here, "clustkit", "__init__.py"), encoding="utf-8") as f:
        exec(f.read(), version_ns)
    return version_ns["__version__"]


CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
]

REQUIRES = [
    "numpy>=1.24.0",
    "numba>=0.58.0",
    "biopython>=1.82",
    "typer>=0.9.0",
    "rich>=13.0.0",
    "scipy>=1.11.0",
    "leidenalg",
    "python-igraph",
    "pypubfigs>=1.1.0",
    "matplotlib>=3.7.0",
]

EXTRAS = {
    "gpu": ["cupy-cuda12x"],
    "dev": [
        "pytest>=7.0",
        "pytest-cov>=4.0",
        "scikit-learn>=1.3.0",
    ],
}

setup(
    name="clustkit",
    description="Accurate protein sequence clustering via LSH, Smith-Waterman alignment, and Leiden community detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jacob L. Steenwyk",
    author_email="jlsteenwyk@gmail.com",
    url="https://github.com/JLSteenwyk/ClustKIT",
    packages=find_packages(),
    classifiers=CLASSIFIERS,
    entry_points={"console_scripts": ["clustkit = clustkit.cli:app"]},
    version=_read_version(),
    include_package_data=True,
    install_requires=REQUIRES,
    extras_require=EXTRAS,
    python_requires=">=3.10",
)

## push new version to pypi
# rm -rf dist
# python3 setup.py sdist bdist_wheel --universal
# twine upload dist/* -r pypi
