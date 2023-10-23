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


class GeomType(unittest.TestCase):
    def setUp(self):
        self.tdir = Path(tempfile.mkdtemp())

        self.rast = self.tdir / 'rast_1.tif'
        self.mesh = self.tdir / 'mesh_1.gr3'

        rast_xy = np.mgrid[-1:0.1:0.1, -0.7:0.1:0.1]
        rast_xy = np.mgrid[0:1.1:0.1, -0.7:0.1:0.1]
        rast_z = np.ones_like(rast_xy[0]) * 10

        ocsmesh.utils.raster_from_numpy(
            self.rast, rast_z, rast_xy, 4326
        )

        msh_t = ocsmesh.utils.create_rectangle_mesh(
            nx=17, ny=7, holes=[40, 41], x_extent=(-1, 1), y_extent=(0, 1))

        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', category=UserWarning, message='Input mesh has no CRS information'
            )
            mesh = ocsmesh.Mesh(msh_t)
            mesh.write(str(self.mesh), format='grd', overwrite=False)


    def tearDown(self):
        shutil.rmtree(self.tdir)


    def test_create_raster_geom(self):
        geom = ocsmesh.Geom(
            ocsmesh.Raster(self.rast),
            zmin=-100,
            zmax=10
        )
        self.assertTrue(isinstance(geom, ocsmesh.geom.raster.RasterGeom))


    def test_create_mesh_geom(self):
        geom = ocsmesh.Geom(
            ocsmesh.Mesh.open(self.mesh, crs=4326),
        )
        self.assertTrue(isinstance(geom, ocsmesh.geom.mesh.MeshGeom))


    def test_create_shape_geom(self):
        geom = ocsmesh.Geom(
            geometry.box(0, 0, 1, 1),
            crs=4326
        )
        self.assertTrue(isinstance(geom, ocsmesh.geom.shapely.PolygonGeom))


    def test_create_shape_geom_no_crs(self):
        self.assertRaises(
            TypeError,
            ocsmesh.Geom,
            geometry.box(0, 0, 1, 1),
        )


    def test_create_collector_geom(self):
        geom = ocsmesh.Geom(
            [self.rast],
            zmin=-100,
            zmax=10
        )
        self.assertTrue(isinstance(geom, ocsmesh.geom.collector.GeomCollector))


class GeomRaster(unittest.TestCase):
    def setUp(self):
        self.tdir = Path(tempfile.mkdtemp())
        self.rast1 = self.tdir / 'rast_1.tif'
        self.rast2 = self.tdir / 'rast_2.tif'
        self.mesh1 = self.tdir / 'mesh_1.gr3'
        topo_2rast_1mesh(self.rast1, self.rast2, self.mesh1)
        

    def tearDown(self):
        shutil.rmtree(self.tdir)


    def test_creation(self):

        rast1 = ocsmesh.Raster(self.rast1)
        geom_rast = ocsmesh.Geom(
            rast1,
            zmin=-100,
            zmax=10
        )

        geom_poly = geom_rast.get_multipolygon()
        self.assertTrue(
            isinstance(geom_poly, (geometry.Polygon, geometry.MultiPolygon))
        )

        geom_msht = geom_rast.msh_t()
        self.assertTrue(isinstance(geom_msht, jigsaw_msh_t))



class GeomCollector(unittest.TestCase):
    # NOTE: Testing a mixed collector geom indirectly tests almost
    # all the other types as it is currently calling all the underlying
    # geoms to calculate msh_t or polygons

    def setUp(self):
        self.tdir = Path(tempfile.mkdtemp())
        self.rast1 = self.tdir / 'rast_1.tif'
        self.rast2 = self.tdir / 'rast_2.tif'
        self.mesh1 = self.tdir / 'mesh_1.gr3'
        self.bx = geometry.box(-1, -2, 1, -1)
        topo_2rast_1mesh(self.rast1, self.rast2, self.mesh1)


    def tearDown(self):
        shutil.rmtree(self.tdir)


    def test_multi_path_input(self):
        geom_coll = ocsmesh.Geom(
            [self.rast1, self.rast2, self.mesh1],
            zmin=-100,
            zmax=10
        )

        geom_poly = geom_coll.get_multipolygon()
        self.assertTrue(
            isinstance(geom_poly, (geometry.Polygon, geometry.MultiPolygon))
        )

        geom_msht = geom_coll.msh_t()
        self.assertTrue(isinstance(geom_msht, jigsaw_msh_t))


    def test_multi_str_input(self):
        geom_coll = ocsmesh.Geom(
            [str(i) for i in [self.rast1, self.rast2, self.mesh1]],
            zmin=-100,
            zmax=10
        )

        geom_poly = geom_coll.get_multipolygon()
        self.assertTrue(
            isinstance(geom_poly, (geometry.Polygon, geometry.MultiPolygon))
        )

        geom_msht = geom_coll.msh_t()
        self.assertTrue(isinstance(geom_msht, jigsaw_msh_t))


    def test_multi_raster_input(self):

        rast1 = ocsmesh.Raster(self.rast1)
        rast2 = ocsmesh.Raster(self.rast2)
        geom_coll = ocsmesh.Geom(
            [rast1, rast2],
            zmin=-100,
            zmax=10
        )

        geom_poly = geom_coll.get_multipolygon()
        self.assertTrue(
            isinstance(geom_poly, (geometry.Polygon, geometry.MultiPolygon))
        )

        geom_msht = geom_coll.msh_t()
        self.assertTrue(isinstance(geom_msht, jigsaw_msh_t))


    def test_multi_mix_input(self):
        rast1 = ocsmesh.Raster(self.rast1)
        mesh1 = ocsmesh.Mesh.open(self.mesh1, crs=4326)
        geom_coll = ocsmesh.Geom(
            # TODO: Note supported yet
            [rast1, self.rast2, mesh1],
#            [rast1, self.rast2, mesh1, self.bx],
            zmin=-100,
            zmax=10
        )

        geom_poly = geom_coll.get_multipolygon()
        self.assertTrue(
            isinstance(geom_poly, (geometry.Polygon, geometry.MultiPolygon))
        )

        geom_msht = geom_coll.msh_t()
        self.assertTrue(isinstance(geom_msht, jigsaw_msh_t))


    def test_mesh_input(self):
        mesh1 = ocsmesh.Mesh.open(self.mesh1, crs=4326)
        geom_coll = ocsmesh.Geom(
            [mesh1],
            zmin=-100,
            zmax=10
        )

        geom_poly = geom_coll.get_multipolygon()
        self.assertTrue(
            isinstance(geom_poly, (geometry.Polygon, geometry.MultiPolygon))
        )

        geom_msht = geom_coll.msh_t()
        self.assertTrue(isinstance(geom_msht, jigsaw_msh_t))


    # TODO: Note supported yet
#    def test_shape_input(self):
#        geom_coll = ocsmesh.Geom(
#            [self.bx],
#            zmin=-100,
#            zmax=10
#        )
#
#
#        geom_poly = geom_coll.get_multipolygon()
#        self.assertTrue(
#            isinstance(geom_poly, (geometry.Polygon, geometry.MultiPolygon))
#        )
#
#        geom_msht = geom_coll.msh_t()
#        self.assertTrue(isinstance(geom_msht, jigsaw_msh_t))


    def test_add_patch(self):
        rast1 = ocsmesh.Raster(self.rast1)
        mesh1 = ocsmesh.Mesh.open(self.mesh1, crs=4326)
        geom_coll = ocsmesh.Geom(
            # TODO: Note supported yet
            [rast1, self.rast2, mesh1],
#            [rast1, self.rast2, mesh1, self.bx],
            zmin=-100,
            zmax=10
        )

        geom_coll.add_patch(
            shape=geometry.box(-0.25, -0.5, 0.5, -0.25),
            level=50,
        )

        geom_poly = geom_coll.get_multipolygon()
        self.assertTrue(
            isinstance(geom_poly, (geometry.Polygon, geometry.MultiPolygon))
        )

        geom_msht = geom_coll.msh_t()
        self.assertTrue(isinstance(geom_msht, jigsaw_msh_t))


    def test_add_patch_shape_should_overlap_all_rasters(self):
        rast1 = ocsmesh.Raster(self.rast1)
        mesh1 = ocsmesh.Mesh.open(self.mesh1, crs=4326)
        geom_coll = ocsmesh.Geom(
            [rast1, self.rast2, mesh1],
            zmin=-100,
            zmax=10
        )

        geom_coll.add_patch(
            shape=geometry.box(-0.75, -0.5, 0, -0.2),
            level=50,
        )

        self.assertRaises(
            ValueError,
            geom_coll.get_multipolygon
        )




if __name__ == '__main__':
    unittest.main()
