"""Shared fixtures for benchmark tests."""

import gc

import numpy as np
import pytest

from ocsmesh import Raster
from ocsmesh.utils import raster_from_numpy


@pytest.fixture(scope="module")
def benchmark_raster_list(tmp_path_factory):
    """Create 4 tiled rasters with 20% overlap for benchmarking.

    Layout is a 2x2 grid covering [0,1] x [0,1].  Each tile spans
    60% of each axis (50% base + 10% overlap per edge), giving 20%
    overlap between adjacent tiles.
    """

    tdir = tmp_path_factory.mktemp("benchmark_rasters")

    # 2x2 tiles with 20% overlap between neighbors
    tile_ranges = [
        (0.0, 0.6, 0.0, 0.6),  # top-left
        (0.4, 1.0, 0.0, 0.6),  # top-right
        (0.0, 0.6, 0.4, 1.0),  # bottom-left
        (0.4, 1.0, 0.4, 1.0),  # bottom-right
    ]

    raster_list = []
    for i, (x0, x1, y0, y1) in enumerate(tile_ranges):
        gx, gy = np.mgrid[x0:x1:60j, y0:y1:60j]
        dem_data = (gx * 20) - 10
        p = tdir / f'dem_{i}.tif'
        raster_from_numpy(p, dem_data, (gx, gy), 4326)
        raster_list.append(Raster(p))

    yield raster_list

    # Cleanup
    del raster_list
    gc.collect()
