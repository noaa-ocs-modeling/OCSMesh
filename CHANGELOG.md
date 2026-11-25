# Changelog

All notable changes to OCSMesh are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions
follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `metric_crs='auto' | 'utm' | <CRS>` on `Hfun`/`HfunCollector`,
  `HfunRaster`, `HfunMesh`, `MeshDriver` and `Geom.geoseries`: where
  sizes and distances are measured (`ocsmesh.geodesy`). With `'auto'`
  (the default) the domain-wide steps (composite size function,
  meshing) use UTM when its worst size error over the domain is at most
  0.5%, and otherwise the conformal projection fitted to the domain with
  the smallest error (transverse Mercator, Lambert conformal conic or
  stereographic): e.g. Alaska 5.0% -> 0.9%, US Atlantic coast 7.0% ->
  2.6%. Refinements keep the UTM zone of each window. An explicit CRS
  (e.g. `'EPSG:3338'`) is used for every step, in the workers too.
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
- DEM resampling for raster size functions, off by default:
  `Hfun(..., resampling_factor=None, resampling_method='average')` computes
  the sizes on pixels of `resampling_factor * hmin` metres (DEMs are never
  resampled to finer pixels). `add_contour`, `add_channel`,
  `add_subtidal_flow_limiter` and `add_constant_value` take their own
  `resampling_factor` (`None` for the collector's, `'native'` for the DEM's
  own pixels) and `resampling_method`, so e.g. channels can be detected on
  coarser pixels while flow limiter slopes come from the full DEM.
  `HfunRaster.refine_on(topo, apply)` computes refinements on another DEM
  grid and reduces them onto the size function's. `Raster.pixel_size_m()`.
  `resampling_factor='optimum'` uses the factors found on the New Jersey
  test domain (grid at `0.5`, channels detected at `0.125`, contours at
  `0.5` with `'min'`) for every refinement that does not set its own,
  with a warning that other domains may need other factors. It does not
  cover flow limiters: adding one to a resampled grid warns that flow
  limiters do not resample well yet.
- Split channel and contour detection for parallel and MPI runs, off by
  default: `Hfun(..., split_size=None, split_halo=None)` detects on
  windows of the DEMs larger than `split_size` pixels (of the DEM's own
  resolution), as separate tasks, so that a few large DEMs no longer keep
  a few workers busy. Only the detection is split; each DEM's size
  function is computed whole. Windows are cut from the DEM's detection
  grid with a halo (the channel width and a few pixels, or `split_halo`
  metres) and keep the shapes in their core; channel pieces cut by window
  edges are joined before small channels are dropped, and contour pieces
  are merged back into lines. `split_size='optimum'` uses the size found
  on the New Jersey test domain (4100), with a warning.
  `Raster.get_channels(..., drop_small=True)` and `utils.channel_min_area`.

### Fixed
- Inputs in different longitude conventions (-180..180 and 0-360), or
  shapes split at 180°, now meet each other: before each geographic
  comparison one side is shifted by a multiple of 360 to the copy
  nearest the other. Region constraints in 0-360 apply east of 180;
  patches and features in either convention (or across 180) refine a
  raster in the other; `base_shape` clips rasters in the other
  convention instead of skipping them; `Mesh.interpolate` finds raster
  values for nodes in the other convention; `Raster.get_bbox` and
  `Mesh.get_bbox` keep boxes across 180 on the right side instead of
  spanning the globe.
- Rasters and shapes east of 180° (0-360 longitudes) got no UTM CRS,
  so feature distances, flow limiter slopes, channel widths and areas,
  and constraint rates were computed in degrees with sizes in metres
  (e.g. every channel dropped, or the whole sea taken as channel), and
  the `fast` method failed. `utils.estimate_bounds_utm` now accepts
  0-360 longitudes and picks the zone on the shortest arc (a mesh
  across 180° in -180..180 longitudes no longer gets the zone at 0°),
  and raises instead of returning `None` for bounds that cannot be
  geographic or span more than 180°, so nothing falls back to degrees.
  The `exact` composite keeps the nodes of 0-360 rasters, the `fast`
  composite keeps their part east of 180, and detected channels stay
  in their raster's longitude convention.
- `Raster.resample` raised for any `resampling_method` passed to it; it
  now takes a `Resampling` member or name, leaves nodata pixels out when
  combining pixels and keeps the raster's extent.
- `utils.get_polygon_channels` (channel detection) raised `AttributeError:
  'Polygon' object has no attribute 'geoms'` when the detected area was a
  single polygon, e.g. one creek left in a small tile.
- Compositing the per-tile size functions (`exact` method) read the valid
  data mask of every higher-priority tile for every tile, so its cost grew
  with the square of the tile count (about 430 s on the coordinator for
  199 tiles). Tiles whose bounds are away from a tile's nodes are now
  skipped without opening their file; the result is unchanged.
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
- `HfunRaster.add_feature` and `add_patch` leave out the shapes that
  cannot reach a raster window (farther than where their expansion
  reaches `hmax`) before repartitioning and reprojecting them. Every
  tile used to process the channels and contours of every other tile;
  the sizes are unchanged.
- Refinement stages with nothing to apply are skipped (they rebuilt and
  rewrote every tile in parallel and MPI modes).
- Contour and channel detection run per tile in parallel and MPI modes.
- Workers no longer write a blank full-resolution raster before reading
  each size function.
- NJ benchmark (18 tiles, NOAA Gaea C6): depth-band size function
  4.0x faster than serial in parallel and MPI modes; channel refinement
  1.9x (bounded by the slowest tile's channel detection).

### Changed
- The size function of a wide domain (where UTM's worst size error
  is over 0.5%, e.g. the Gulf of Mexico or the US Atlantic coast) is
  projected to a conformal projection fitted to the domain instead of
  one UTM zone, and `MeshDriver` meshes such domains in it, so their
  meshes change (sizes are closer to the requested ones).
  `metric_crs='utm'` gives the previous projection. Small domains are
  unchanged.
- `Mesh.write(format='grd')` writes the open boundaries from the
  largest to the smallest (number of nodes), as SCHISM expects; before,
  they were written in the order of `mesh.boundaries`, which after
  `set_open(..., merge=True)` follows the geometry, not the size.
  Boundaries of the same size keep their order, and `mesh.boundaries`
  itself is not changed. `sort_open_boundaries=False` gives the old
  order. Check the open boundary order of an existing `bctides.in`
  against the new .gr3.
- Documentation extra: `sphinx-mdinclude` with `mistune>=3.3.3` replace
  `m2r2` and `mistune==0.8.4`, which had known vulnerabilities; the
  release workflow uses `pypa/gh-action-pypi-publish@v1.14.3` (was
  v1.5.0). Fixes the Dependabot alerts on both.
- README badges: the Pylint badge shows the score (written by the
  Pylint workflow on main to the `badges` branch), and the Python badge
  reads `requires-python` from `pyproject.toml` (it showed "missing").
  `pyproject.toml` lists the Python version classifiers.
- CI uploads test coverage to Codecov (`codecov.yml`; informational
  checks that do not block merges).
- Internal: one task path for all distributed refinements and one
  writer for the per-tile size function files, shared by the serial,
  parallel and MPI modes.
