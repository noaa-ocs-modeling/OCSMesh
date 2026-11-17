Splitting large rasters
=======================

With ``execution_mode='parallel'`` or ``'mpi'`` the size function work
is shared out one raster per task. A raster much larger (or busier)
than the others then keeps one worker busy while the rest wait.
``split_size`` cuts such rasters into windows, which are processed as
separate tasks. It is off by default.

.. code-block:: python

    import ocsmesh

    hfun = ocsmesh.Hfun(
        tile_paths, hmin=50, hmax=3000, method='exact',
        execution_mode='mpi',
        split_size=4000)          # windows of at most ~4000 x 4000 pixels

``split_size`` counts pixels of each DEM at its own resolution (where
channel and contour detection is the costly part), whatever
``resampling_factor`` the sizes are computed with. Rasters no larger
than ``split_size`` are left whole. Splitting applies to the
``'exact'`` method.

Halo and core
-------------

Each window is computed on its *core* plus a *halo* around it, and only
the core is kept:

* channels and contours detected on a window are clipped to its core;
* the window's size function only provides sizes inside its core when
  the inputs are composited.

The cores of a raster tile it exactly, so every pixel is kept once.

The halo gives the core the neighbourhood its refinements look at.
Without it the window edges themselves look like channels: a water body
cut by an edge leaves narrow strips that the channel detection keeps.
Its width, ``split_halo`` in metres, is derived from the refinements
when not given:

* channels: the channel ``width`` (plus the simplification
  ``tolerance``), since a channel is only recognised when its whole
  width is visible;
* constraints with a ``rate``: the distance at which the rate reaches
  ``hmax`` from ``hmin``, limited to the window size (a warning says
  when it is limited; the spread then stops at the window edges, as it
  does at the edges of separate tiles);
* a few pixels, for contours and slopes.

Channel detection drops channels smaller than ``width`` squared minus
a quarter circle (0.86 km² for a 2 km width). That is decided on the
whole channel, not on the window: a creek network cut by a window edge
can leave pieces below that size whose channel is not. Windows keep
their small pieces, and once all windows are detected the pieces that
touch across cores are joined and the cutoff is applied to the joined
area.

Detected channels and contours, patches and line features are applied
to every window from the full set of shapes, so their expansion crosses
the window edges as it does without splitting.

Choosing ``split_size``
-----------------------

The halo is computed for every window and then thrown away, so a
window should be several times wider than its halo: with a 2 km
channel width and 3 m pixels, 4000 pixel (12 km) cores are computed on
16 km windows, 1.8 times the pixels of the core, and 8000 pixel cores
on 1.4 times. Splitting helps when there are fewer rasters than
workers, or when one raster takes much longer than the others.

Results match the unsplit raster up to one pixel along the edges of
detected channels: their polygons lie on pixel edges, and whether a
pixel next to one counts as touched can change with rounding.
