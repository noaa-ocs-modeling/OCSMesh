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
    # slopes from the DEM's own pixels
    hfun.add_subtidal_flow_limiter(hmin=100, hmax=3000,
                                   lower_bound=-10, upper_bound=0,
                                   resampling_factor='native')

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

Tests on a New Jersey domain (3 m CUDEM, ``hmin=50`` m) give a starting
point; the right values depend on the DEMs and the refinements:

* constant values, region constraints, patches, line features and the
  Courant constraint barely change up to ``1.0``;
* depth bands and constraints with a ``rate`` stay within a few percent
  of the mesh up to ``0.5`` and ``0.25``;
* contours keep their size function at ``0.5``, but the line itself
  moves where marshes and creeks are thinner than the pixels: trace it
  on the native DEM, or with ``resampling_method='min'``;
* channels: ``0.125`` with ``'average'`` matched the channels found on
  the full DEM with an overlap (intersection over union) of 0.97 and
  ran 14 times faster; ``'min'`` detects channels that are not there;
* flow limiters depend on slopes, which change with the pixel size:
  use ``'native'``.
