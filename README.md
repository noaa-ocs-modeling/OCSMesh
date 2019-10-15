# geomesh
```bash
./setup.py install_jigsaw
./setup.py install
./setup.py develop # optional
``` 
Example 2 should work fine out of the box. Example 1 uses a very large DEM and will likely run out of memory. Work is being done to implement numpy.memmap arrays in order to mitigate memory usage issues.