# Geomesh
## A Python package for processing DEM data into georeferenced unstructured meshes using the [jigsaw-python](https://github.com/dengwirda/jigsaw-python) library.

### Installation

The Jigsaw library must be initialized first by running `./setup.py bootstrap_jigsaw`, then, the package can be installed normally by doing `./setup.py install`:

```bash
./setup.py bootstrap_jigsaw # installs the Jigsaw library to the current Python environment
./setup.py install # Installs the geomesh libraru to the current Python environment
./setup.py develop # run this if you are a developer.
```
This package requires Python>=3.8