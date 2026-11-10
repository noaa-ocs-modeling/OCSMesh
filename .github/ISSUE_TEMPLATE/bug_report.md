---
name: Bug report
about: Something does not work as documented
labels: bug
---

**What happened**
A clear description, and what you expected instead.

**How to reproduce**
A minimal script (synthetic rasters or a small public DEM if possible):

```python
```

**Environment**
- OCSMesh version (`python -c "import ocsmesh, importlib.metadata as m; print(m.version('ocsmesh'))"`):
- Python and OS:
- `method` (`exact`/`fast`) and `execution_mode` (`serial`/`parallel`/`mpi`), `nprocs`:
- For MPI: launcher (`mpiexec`/`srun`), number of ranks and nodes, MPI library:
- Mesh engine (`gmsh`/`triangle`):

**Logs**
The full traceback or the relevant part of the log.
