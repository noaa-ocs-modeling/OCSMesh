# Changelog

All notable changes to OCSMesh are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions
follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Distributed size function computation: `Hfun(..., execution_mode=...)`
  (or the `execution_mode` property) with `'serial'`, `'parallel'`
  (multiprocessing Pool, `nprocs` workers) and `'mpi'` (one rank per
  tile under `mpiexec`/`srun`, rank 0 coordinates). MPI support is an
  optional extra: `pip install ocsmesh[mpi]` (#248-#254).
- `execution_mode` as an `HfunCollector` constructor argument. Under an
  MPI launcher, `'serial'` or `'parallel'` gives every rank its own
  independent collector; `'mpi'` or `None` share one collector.
- `utils.clip_elevation(mesh, min_elevation, max_elevation)` clamps the
  node elevations of a `Mesh` or `MeshData` in place.
- `Mesh.interpolate` warns when some nodes get no value from any raster.
- `add_subtidal_flow_limiter(..., flat_areas='hmin' | 'keep')` on
  `HfunRaster` and `HfunCollector`: size on flat cells (zero topography
  gradient), where the limiter formula is undefined. The default,
  `'hmin'`, gives them the limiter's `hmin` as before; `'keep'` leaves
  their size unchanged.
- Gmsh engine option `bnd_factor` (default 1.0): with
  `bnd_representation='adapt'`, the boundary is resampled with edges
  `bnd_factor` times the size function (e.g. 1.5 gives 4.5 km edges where
  the size function is 3 km), and Gmsh keeps those edges as they are.
  Also `utils.resample_geom_by_hfun(..., factor=1.0)`.

### Fixed
- `utils.finalize_mesh` (called by `MeshDriver.run`, the remesh and
  combine CLIs) now orients every element counterclockwise, so written
  meshes no longer contain clockwise triangles. `utils.order_mesh` is
  vectorized and takes `in_place=True`; `ops.combine_mesh` called it
  and discarded the result.
- Parallel and MPI workers refined the unclipped DEM when rasters were
  clipped by `base_shape`, so depth-based refinements landed on the
  wrong pixels; results now match serial exactly.
- A raster tile with no valid data inside the domain, or a refinement
  that found nothing (no contour, no channel), turned the whole size
  function into `hmin`. Such tiles are now skipped with a warning, and
  empty refinements leave the size function unchanged.
- `exact` size functions are composited by each tile's valid-data
  footprint instead of its bounding box: nodata margins of a
  higher-priority tile no longer override valid lower-priority data
  (strips of wrong sizes along tile edges).
- Constant value, flow limiter and topography constraints used raster
  nodata as an elevation. `Raster.clip` filled the outside of the clip shape with 0
  (sea level) when the DEM declares no nodata value.
- `HfunCollector.meshdata()` returned a stale cached result after new
  refinements were added (and an MPI run could hang on a second call).
- A failed worker in a parallel (Pool) stage was logged and skipped,
  silently dropping a refinement or a tile; it now raises.
- `Mesh.interpolate(method='spline')` used raster nodata as values;
  with `chunk_size` set, interpolation raised `IndexError`.
- Gmsh failed with "Unsupported geometry type" when the domain became a
  `GeometryCollection` after `make_valid`; only its polygons are meshed.
- Lambdas are accepted again in `add_topo_func_constraint`; in parallel
  and MPI modes such constraints are applied on the coordinator.
- No more "Exception ignored in HfunCollector.__del__" at interpreter
  exit.
- The automatic `meshdata()` stride assumed a DEM in degrees; projected
  DEMs (e.g. UTM lidar) now get the intended subsampling.
- Contour and channel definitions kept the rasters of the first
  `meshdata()` call attached.
- `import ocsmesh` pinned BLAS threads to 1 and set the `spawn` start
  method in every SLURM job, serial ones included; it now does so only
  when a launcher reports more than one task.
- `utils.estimate_bounds_utm` warns instead of silently returning
  `None` for geographic bounds outside [-180, 180] x [-90, 90].

### Performance
- Refinement stages with nothing to apply are skipped (they rebuilt and
  rewrote every tile in parallel and MPI modes).
- Contour and channel detection run per tile in parallel and MPI modes.
- Workers no longer write a blank full-resolution raster before reading
  each size function.
- NJ benchmark (18 tiles, NOAA Gaea C6): depth-band size function
  4.0x faster than serial in parallel and MPI modes; channel refinement
  1.9x (bounded by the slowest tile's channel detection).

### Changed
- Documentation extra: `sphinx-mdinclude` with `mistune>=3.3.3` replace
  `m2r2` and `mistune==0.8.4`, which had known vulnerabilities; the
  release workflow uses `pypa/gh-action-pypi-publish@v1.14.3` (was
  v1.5.0). Fixes the Dependabot alerts on both.
- CI uploads test coverage to Codecov (`codecov.yml`; informational
  checks that do not block merges).
- Internal: one task path for all distributed refinements and one
  writer for the per-tile size function files, shared by the serial,
  parallel and MPI modes.
