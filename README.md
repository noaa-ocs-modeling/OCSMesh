![lint workflow](https://github.com/noaa-ocs-modeling/OCSMesh/actions/workflows/pylint.yml/badge.svg?branch=main)
![fnc workflow](https://github.com/noaa-ocs-modeling/OCSMesh/actions/workflows/functional_test.yml/badge.svg?branch=main)
![fnc2 workflow](https://github.com/noaa-ocs-modeling/OCSMesh/actions/workflows/functional_test_2.yml/badge.svg?branch=main)

## OCSMesh
OCSMesh is a Python package for processing DEM data into georeferenced
unstructured meshes using different meshing engine libraries.

### Installation
Two ways of installing OCSMesh are described below:

#### Using `conda`
The recommended way to setup the environment for installing OCSMesh is to
use [`conda`](https://docs.conda.io/en/latest/miniconda.html#linux-installers)
with the `environment.yml` file provided in the repo to install
required libraries.

You need to make sure you've installed the mesh engine you'd like to
use (i.e. Jigsaw, Triangle, or GMsh).

First you need to download the `environment.yml` file.
```
wget https://raw.githubusercontent.com/noaa-ocs-modeling/OCSMesh/main/environment.yml

conda env create -f environment.yml -n your-env-name
conda activate your-env-name

pip install ocsmesh
```

#### From GitHub repo
You need to make sure you've installed the mesh engine you'd like to
use (i.e. Jigsaw, Triangle, or GMsh).

OCSMesh can be installed from the GitHub repository as follows:
```
git clone https://github.com/noaa-ocs-modeling/ocsmesh
cd ocsmesh
pip install .[all]
```

#### Requirements
* 3.10 <= Python
* CMake 
* C/C++ compilers

## How to Cite
```
Title : OCSMesh: a data-driven automated unstructured mesh generation software for coastal ocean modeling
Personal Author(s) : Mani, Soroosh;Calzada, Jaime R.;Moghimi, Saeed;Zhang, Y. Joseph;Myers, Edward;Pe’eri, Shachak;
Corporate Authors(s) : Coast Survey Development Laboratory (U.S.)
Published Date : 2021
Series : NOAA Technical Memorandum NOS CS ; 47
DOI : https://doi.org/10.25923/csba-m072
```
```
Title : OCSMesh and an end-to-end workflow for fully automatic mesh generation with application to compound flood studies
Personal Author(s) : Cassalho, Felicio;Mani, Soroosh;Ye, Fei;Zhang, Y. Joseph;Moghimi, Saeed;
Published Date : Pre-print(2025)
DOI : http://dx.doi.org/10.2139/ssrn.5226658
```

