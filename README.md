# geomesh
#### A Python package for generating georeferenced unstructured meshes using the [jigsaw](https://github.com/dengwirda/jigsaw) library.

Requires:
 - Python >= 3.7
 - CMake (??)
 - gcc >= 5.0

Optional
 - git-lfs (to fetch the example data)

```bash
./setup.py install_jigsaw
./setup.py install
./setup.py develop # optional
``` 
To run the examples, you must have git-lfs installed and initialize git-lfs on the repository:

```bash
git-lfs install
git submodule update --init examples/data
./examples/example_1.py && ./examples/example_2.py
```
