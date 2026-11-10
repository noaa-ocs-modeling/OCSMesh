[![Functional tests](https://github.com/noaa-ocs-modeling/OCSMesh/actions/workflows/functional_test.yml/badge.svg?branch=main)](https://github.com/noaa-ocs-modeling/OCSMesh/actions/workflows/functional_test.yml)
[![Pylint](https://github.com/noaa-ocs-modeling/OCSMesh/actions/workflows/pylint.yml/badge.svg?branch=main)](https://github.com/noaa-ocs-modeling/OCSMesh/actions/workflows/pylint.yml)
[![codecov](https://codecov.io/gh/noaa-ocs-modeling/OCSMesh/branch/main/graph/badge.svg)](https://codecov.io/gh/noaa-ocs-modeling/OCSMesh)
[![Documentation](https://github.com/noaa-ocs-modeling/OCSMesh/actions/workflows/documentation.yml/badge.svg)](https://noaa-ocs-modeling.github.io/OCSMesh/)
[![PyPI](https://img.shields.io/pypi/v/ocsmesh)](https://pypi.org/project/ocsmesh/)
[![Python](https://img.shields.io/pypi/pyversions/ocsmesh)](https://pypi.org/project/ocsmesh/)
[![License: CC0-1.0](https://img.shields.io/badge/license-CC0--1.0-blue)](LICENSE)

## OCSMesh
OCSMesh is a Python package for processing DEM data into georeferenced
unstructured meshes using different meshing engine libraries
(Gmsh, Triangle). It builds the domain geometry and the mesh
size function from any number of DEM tiles, applies refinements
(contours, channels, depth bands, flow limiters, patches, line
features) and constraints, and runs the mesh engine. Size functions can
be computed serially, on all cores of a node or with MPI across nodes.

Documentation: https://noaa-ocs-modeling.github.io/OCSMesh/

### Installation
Two ways of installing OCSMesh are described below:

#### Using `conda`
The recommended way to setup the environment for installing OCSMesh is to
use [`conda`](https://docs.conda.io/en/latest/miniconda.html#linux-installers)
with the `environment.yml` file provided in the repo to install
required libraries.

You need to make sure you've installed the mesh engine you'd like to
use (e.g., Triangle or Gmsh).

First you need to download the `environment.yml` file.
```
wget https://raw.githubusercontent.com/noaa-ocs-modeling/OCSMesh/main/environment.yml

conda env create -f environment.yml -n your-env-name
conda activate your-env-name

pip install ocsmesh
```

#### From GitHub repo
You need to make sure you've installed the mesh engine you'd like to
use (e.g., Triangle or Gmsh).

OCSMesh can be installed from the GitHub repository as follows:
```
git clone https://github.com/noaa-ocs-modeling/ocsmesh
cd ocsmesh
pip install .[all]
```

#### MPI (optional)
MPI execution needs `mpi4py` built against the MPI library of your
machine. `environment.yml` installs it from conda-forge; on an HPC
system, prefer the system MPI (e.g. `module load` the MPI, then
`pip install --no-binary mpi4py mpi4py`), or `pip install ocsmesh[mpi]`.

#### Requirements
* 3.10 <= Python < 3.14
* CMake
* C/C++ compilers

### Quick start
```python
import ocsmesh

dems = ["tile_1.tif", "tile_2.tif"]       # later inputs take priority
domain = ocsmesh.Geom(dems, zmax=10)      # everything below 10 m

hfun = ocsmesh.Hfun(dems, hmin=50, hmax=3000, method="exact",
                    nprocs=8, execution_mode="parallel")
hfun.add_contour(level=0, expansion_rate=0.01, target_size=100)
hfun.add_constant_value(value=500, lower_bound=-20, upper_bound=0)

mesh = ocsmesh.MeshDriver(domain, hfun, engine_name="gmsh").run()
mesh.write("mesh.2dm", format="2dm", overwrite=True)
```

### Execution modes
| `execution_mode` | runs on | use it for |
|---|---|---|
| `"serial"` (default) | this process | small domains, debugging |
| `"parallel"` | a pool of `nprocs` processes on one node | a workstation or one HPC node |
| `"mpi"` | MPI ranks under `mpiexec`/`srun`, one tile per rank | many tiles, several nodes |

All modes give the same size function. With MPI every rank runs the
script and calls `hfun.meshdata()`; the result is returned on rank 0.
Temporary files go to `$TMPDIR`, which must be on a shared filesystem
for multi-node runs. See
[Parallel and MPI execution](https://noaa-ocs-modeling.github.io/OCSMesh/parallel.html)
for details.

### Contributing
Bug reports and pull requests are welcome; see
[CONTRIBUTING.md](CONTRIBUTING.md). Changes are listed in
[CHANGELOG.md](CHANGELOG.md).

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

## Disclaimer
This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an "as is" basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.
