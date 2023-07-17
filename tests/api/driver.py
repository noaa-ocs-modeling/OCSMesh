import unittest
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import warnings

from jigsawpy import jigsaw_msh_t
import geopandas as gpd
import numpy as np
import rasterio as rio
from shapely import geometry

import ocsmesh

from tests.api.common import (
    topo_2rast_1mesh,
)


class Driver(unittest.TestCase):

    def setUp(self):
        self.tdir = Path(tempfile.mkdtemp())
        self.rast1 = self.tdir / 'rast_1.tif'
        self.rast2 = self.tdir / 'rast_2.tif'
        self.mesh1 = self.tdir / 'mesh_1.gr3'
        topo_2rast_1mesh(self.rast1, self.rast2, self.mesh1)


    def tearDown(self):
        shutil.rmtree(self.tdir)


    def test_simple_mesh_gen(self):

        mesh1 = ocsmesh.Mesh.open(self.mesh1, crs=4326)
        hfun_mesh_1 = ocsmesh.Hfun(deepcopy(mesh1))
        hfun_mesh_1.size_from_mesh()

        geom = ocsmesh.Geom(
            [self.rast1, self.rast2, mesh1],
            zmin=-100,
            zmax=10
        )

        hfun = ocsmesh.Hfun(
            [self.rast1, self.rast2, hfun_mesh_1],
            hmin=1000,
            hmax=5000,
            method='exact')
        hfun.add_contour(target_size=1200, expansion_rate=0.5, level=0)
        hfun.add_constant_value(value=1200, lower_bound=0)

        driver = ocsmesh.driver.JigsawDriver(geom, hfun)
        mesh = driver.run()

        self.assertTrue(isinstance(mesh, ocsmesh.mesh.base.BaseMesh))
        self.assertTrue(len(mesh.msh_t.vert2['coord']) > 0)

if __name__ == '__main__':
    unittest.main()
