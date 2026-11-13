DEM resampling
==============

A size function from fine DEMs (e.g. 3 m CUDEM tiles) is computed on
every DEM pixel, although the mesh is never finer than ``hmin``. Most of
the time then goes into pixels far smaller than any element.
``resampling_factor`` computes the sizes on coarser pixels instead. It
is off by default, so existing scripts give the same size functions.

The factor is a fraction of ``hmin``: with ``hmin=50`` m, ``1.0`` gives
50 m pixels, ``0.5`` 25 m and ``0.25`` 12.5 m. DEMs are never resampled
to pixels finer than their own, so a factor finer than a tile leaves
that tile as it is.

Size function grid
------------------

The collector's factor sets the grid the sizes are computed on. Each
input DEM is resampled once, after it is clipped to ``base_shape``:

.. code-block:: python

    import ocsmesh

    hfun = ocsmesh.Hfun(
        tile_paths, hmin=50, hmax=3000, method='exact',
        base_shape=domain, base_shape_crs='EPSG:4326',
        resampling_factor=0.5,            # 25 m grid
        resampling_method='average')

``resampling_method`` is a ``rasterio.enums.Resampling`` name:
``'average'`` (default), ``'min'`` (deepest value: keeps narrow channels
and creeks connected), ``'max'``, ``'bilinear'``, ``'nearest'``, ...
Nodata pixels are left out when pixels are combined.

Resolution per refinement
-------------------------

Contours, channels, flow limiters and constant values (depth bands)
take their own ``resampling_factor`` and ``resampling_method``, for the
DEM they are computed from:

``None`` (default)
    the collector's value, i.e. the size function grid
``'native'``
    the DEM's own pixels
a number
    ``factor * hmin`` pixels

.. code-block:: python

    # contours traced on the full DEM
    hfun.add_contour(level=0, expansion_rate=0.001, target_size=200,
                     resampling_factor='native')
    # channels detected on hmin/8 pixels
    hfun.add_channel(level=0, width=2000, target_size=500,
                     resampling_factor=0.125)
    # depth bands from a 50 m DEM
    hfun.add_constant_value(value=500, lower_bound=-20, upper_bound=-5,
                            resampling_factor=1.0)

Detected contours and channels are lines and polygons, so they are
applied to the size function grid whatever DEM they come from. Flow
limiters and constant values computed on finer pixels than the grid
are reduced onto it keeping the smallest size in each cell. Constraints,
patches and line features are always computed on the grid.

Resampling applies to the ``exact`` method, to DEMs given to the
collector as rasters or file paths, and works the same in every
execution mode (see :doc:`parallel`).

Choosing a factor
-----------------

Tests on a New Jersey domain (3 m CUDEM, ``hmin=50`` m, serial runs
compared with the same run without resampling) give a starting point;
the right values depend on the DEMs and the refinements:

* constant values (depth bands), region constraints, patches, line
  features, the Courant constraint and ``add_topo_func_constraint``
  barely change up to ``1.0`` (10 to 100 times faster, mesh nodes
  within 1%);
* constraints with a ``rate``: ``0.5`` with ``resampling_method='min'``
  (130 times faster, nodes within 2%), or ``0.25`` with ``'average'``;
* constraints without a ``rate`` follow every pixel at the edge of
  their depth range, so the size function changes at any factor, but
  the mesh does not (nodes within 2% at ``1.0``);
* contours: ``0.5`` with ``'min'`` kept the size function and the line
  within 63 m (95th percentile), 20 times faster; ``'native'`` contours
  are exact but cost most of the original run time;
* channels: ``resampling_factor=0.125`` on ``add_channel`` found the
  channels of the full DEM with an overlap (intersection over union) of
  0.97 (``0.25``: 0.94). Detection is most of the cost, so a coarse
  grid comes free: with the collector at ``1.0`` the run was 16 times
  faster in serial and 18 times faster with MPI, and the mesh had 6%
  fewer nodes. ``'min'`` detects channels that are not there;
* flow limiters: not yet. Slopes from a coarse DEM are too gentle (40%
  fewer nodes), and slopes from the native DEM reduced to a coarse grid
  keep the smallest size in each cell, which refines far too much.
  Leave the collector without ``resampling_factor`` when using
  ``add_subtidal_flow_limiter``.
