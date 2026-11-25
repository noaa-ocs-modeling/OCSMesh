Metric CRS
==========

OCSMesh measures sizes, distances, slopes and areas in metres. For
geographic inputs (longitudes and latitudes) it projects to a metric
CRS first. ``metric_crs`` chooses that CRS. It is an argument of
``Hfun`` (and ``HfunCollector``, ``HfunRaster``, ``HfunMesh``) and of
``MeshDriver``, which takes the size function's value when it is not
given.

.. code-block:: python

    import ocsmesh

    hfun = ocsmesh.Hfun(tile_paths, hmin=500, hmax=5000,
                        metric_crs='auto')          # the default
    driver = ocsmesh.MeshDriver(geom, hfun)          # uses hfun's

``'auto'`` (default)
    The composite size function and the meshing use UTM, with the zone
    at the centre of the domain, when UTM's worst size error over the
    domain is at most 0.5%; small and medium domains are projected as
    before. Wider domains get the conformal projection fitted to the
    domain with the smallest worst error: transverse Mercator centred
    on the domain, Lambert conformal conic with standard parallels at
    1/6 and 5/6 of the domain's latitudes, or oblique stereographic
    centred on the domain. Conformal projections keep angles, so mesh
    elements are not stretched; their scale factor is set so that the
    size error is the same at both extremes. The choice is logged.
    Refinements (distances to features, slopes, channel widths and
    areas, constraint rates) keep the UTM zone of each window, which
    is the most accurate locally.
``'utm'``
    UTM everywhere, as before ``metric_crs`` existed.
A CRS (``'EPSG:3338'``, a PROJ string, a ``pyproj.CRS``)
    That CRS for every step, refinements included, in the parallel and
    MPI workers too. It must be projected, in metres.

Worst size error in any direction over each domain's bounding box:

=========================  ==========  ======================  ======
Domain                     UTM         ``'auto'`` picks        Error
=========================  ==========  ======================  ======
New Jersey                 0.05%       UTM (unchanged)         0.05%
Hawaii                     0.31%       UTM (unchanged)         0.31%
Gulf of Mexico             1.66%       LCC                     0.32%
Alaska (172E-130W)         4.97%       LCC                     0.92%
US Atlantic (98W-60W)      6.97%       stereographic           2.59%
South Pacific (160E-150W)  12.2%       LCC                     1.21%
=========================  ==========  ======================  ======

Domains across 180°
-------------------

Longitudes may be in -180..180 or 0-360. The centre of the domain is
taken on the shortest arc that holds all its longitudes, so a domain
crossing 180° (Alaska, the Aleutians, the Pacific) gets the right UTM
zone or fitted projection in either convention. Bounds that cannot be
longitudes and latitudes, or that span more than 180° of longitude,
raise an error instead of computing in degrees.

.. automodule:: ocsmesh.geodesy
    :members: pick_metric_crs, check_metric_crs, estimate_bounds_utm,
              scale_error, lon_arc
