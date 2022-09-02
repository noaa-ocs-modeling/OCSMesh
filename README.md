![lint workflow](https://github.com/noaa-ocs-modeling/OCSMesh/actions/workflows/pylint.yml/badge.svg?branch=main)
![fnc workflow](https://github.com/noaa-ocs-modeling/OCSMesh/actions/workflows/functional_test.yml/badge.svg?branch=main)
![fnc2 workflow](https://github.com/noaa-ocs-modeling/OCSMesh/actions/workflows/functional_test_2.yml/badge.svg?branch=main)

## OCSMesh
OCSMesh is a Python package for processing DEM data into georeferenced
unstructured meshes using the
[jigsaw-python](https://github.com/dengwirda/jigsaw-python) library.

### Installation
Two ways of installing OCSMesh are described below:

#### Using `conda`
The recommended way to setup the environment for installing OCSMesh is to
use [`conda`](https://docs.conda.io/en/latest/miniconda.html#linux-installers)
with the `environment.yml` file provided in the repo to install
required libraries.

The Jigsaw library and its Python wrapper  must be instlled
before OCSMesh can be used. Jigsaw is available on `conda-forge`
channel.

First you need to download the `environment.yml` file.

```
wget https://raw.githubusercontent.com/noaa-ocs-modeling/OCSMesh/main/environment.yml

conda env create -f environment.yml -n your-env-name
conda activate your-env-name

conda install -y -c conda-forge jigsawpy
pip install ocsmesh
```

#### From GitHub repo
OCSMesh can be installed from the GitHub repository as well.
After downloading the repo, you need to first install Jigsaw using
the script provided in OCSMesh repo by calling:
`./setup.py install_jigsaw` in the OCSMesh root directory.
Then OCSMesh can be installed. 

```
git clone https://github.com/noaa-ocs-modeling/ocsmesh
cd ocsmesh
python ./setup.py install_jigsaw # To install latest Jigsaw from GitHub
python ./setup.py install # Installs the OCSMesh library to the current Python environment
# OR
python ./setup.py develop # Run this if you are a developer.
```

#### Requirements
* 3.7 <= Python < 3.10
* CMake 
* C/C++ compilers

## How to Cite
Title : OCSMesh: a data-driven automated unstructured mesh generation software for coastal ocean modeling

Personal Author(s) : Mani, Soroosh;Calzada, Jaime R.;Moghimi, Saeed;Zhang, Y. Joseph;Myers, Edward;Pe’eri, Shachak;

Corporate Authors(s) : Coast Survey Development Laboratory (U.S.)

Published Date : 2021

Series : NOAA Technical Memorandum NOS CS ; 47

DOI : https://doi.org/10.25923/csba-m072

