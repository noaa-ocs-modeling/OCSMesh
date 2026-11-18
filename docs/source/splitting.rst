Splitting channel and contour detection
=======================================

With ``execution_mode='parallel'`` or ``'mpi'`` channels and contours
are detected one raster per task. Detection is the costly part of these
refinements on a large DEM, so a raster much larger (or busier) than the
others keeps one worker busy while the rest wait. ``split_size`` detects
on windows of such rasters instead, as separate tasks. It is off by
default.

.. code-block:: python

    import ocsmesh

    hfun = ocsmesh.Hfun(
        tile_paths, hmin=50, hmax=3000, method='exact',
        execution_mode='mpi',
        split_size=4000)          # windows of at most ~4000 x 4000 pixels
    hfun.add_channel(level=0, width=2000, target_size=500,
                     resampling_factor=0.125)

Only the detection is split. The size function of every raster is
computed whole, from the shapes detected on all its windows, exactly as
without splitting, so depth bands, flow limiters, constraints and their
rates, patches and line features are not affected.

``split_size`` counts pixels of each DEM at its own resolution, whatever
resolution the shapes are detected on. Rasters no larger than
``split_size`` are detected whole. Splitting applies to the ``'exact'``
method.

Windows, halo and core
----------------------

The windows of a raster are cut from its *detection grid*: the DEM at
the resolution the definition is detected on (its own
``resampling_factor``), laid out for the whole raster. Every window
pixel is therefore a pixel of the whole raster's grid; resampling each
window on its own would shift its pixels.

Each window is a *core* plus a *halo* around it, and only the shapes in
the core are kept. The cores of a raster tile its grid, so every pixel is
detected once. The halo gives the core the neighbourhood detection looks
at: a channel is only recognised when its whole width is visible, and
without a halo the window edges themselves look like channels (a water
body cut by an edge leaves narrow strips that the detection keeps). Its
width, ``split_halo`` in metres, is the channel width plus its
simplification ``tolerance`` and four pixels when not given; contours
only need the four pixels.

The shapes of all windows are then put together on the coordinator:

* channel detection drops channels smaller than ``width`` squared minus
  a quarter circle (0.86 km² for a 2 km width). That is decided on the
  whole channel, not on the window: a creek network cut by a window edge
  can leave pieces below that size whose channel is not. Windows keep
  their small pieces, and the pieces that continue each other across the
  edge between two cores are joined before the cutoff is applied;
* contour pieces are merged back into whole lines, so that they are
  sampled along their length (``add_feature``) as unsplit lines are.

Choosing ``split_size``
-----------------------

A raster ``W`` pixels wide is cut into ``ceil(W / split_size)`` equal
parts along each axis, and the same along y, so no window is larger
than ``split_size`` and rasters no larger than it are detected whole.
With tiles of about 8,100 pixels (0.25 degree at 3 m), any value from
4,050 to 8,099 gives 2 x 2 windows, 2,700 to 4,049 gives 3 x 3.

``split_size='optimum'`` uses the size found on the New Jersey test
domain, 4100 (2 x 2 windows of its 3 m tiles; channels detected 2.2
times faster than unsplit with MPI, with the same result), and warns
that other domains or DEMs may need another value.

Splitting helps when there are fewer rasters than workers, or when one
raster takes much longer than the others, and only when detection is a
large part of the run. The halo is detected for every window and then
thrown away, so a window should be several times wider than its halo:
with a 2 km channel width and 3 m pixels, 4000 pixel (12 km) cores are
detected on 16 km windows, 1.8 times the pixels of the core. Smaller
windows also have more edges, where the pieces of channels have to be
joined.

Results match detection on whole rasters up to one pixel along the edges
of detected channels: their polygons lie on pixel edges, and whether a
pixel next to one counts as touched can change with rounding.
