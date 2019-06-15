# geomesh

geomesh is a Python toolkit for generating and developing unstructured meshes.

## Pre-requisites

This package requires Python 3.7. Please create a Python 3.7 virtual environment and activate it using virtualenv before installing this package.

python-3.7
git
Cmake
Boost
GDAL

### Note about GDAL
GDAL is available as a system package in every major OS or it can be compiled directly from source.


## Cloning the package

geomesh uses several C/C++ dependencies. Make sure you have cloned the submodules as well as the submodule depencies. This can be done with a single line:

```bash
git clone --recurse-submodules git://github.com/jreniel/geomesh.git
```
Otherwise, if you have already cloned the repo siply do:

```bash
git submodule update --init --recursive
```

## Installation

This package uses CMake to build C/C++ dependencies. You must first build, then install the package. To do so first run:

```bash
./setup.py build
```
Then you can run

```bash
./setup.py install
```


## Usage
Coming soon