import shutil
import tempfile
import unittest
import platform
from pathlib import Path

import numpy as np

import ocsmesh
from ocsmesh.utils import raster_from_numpy


IS_WINDOWS = platform.system() == 'Windows'



class Raster(unittest.TestCase):
    def setUp(self):
        self.tdir = Path(tempfile.mkdtemp())
        self.rast1 = self.tdir / 'rast_1.tif'
        self.rast2 = self.tdir / 'rast_2.tif'
        self.rast3 = self.tdir / 'rast_3.tif'

        rast_xy = np.mgrid[-1:1:0.01, -0.7:0.7:0.01]
        rast_z_1 = np.ones_like(rast_xy[0]) * 10
        rast_z_2 = np.ma.MaskedArray(
            np.ones_like(rast_xy[0]) * 10,
            mask=np.random.random(size=rast_xy[0].shape) < 0.2,
            fill_value=np.nan
        )
        rast_z_3 = np.ma.MaskedArray(
            np.ones_like(rast_xy[0]) * 10,
            mask=np.random.random(size=rast_xy[0].shape) < 0.2,
            fill_value=-1e+15
        )

        raster_from_numpy(self.rast1, rast_z_1, rast_xy, 4326)
        raster_from_numpy(self.rast2, rast_z_2, rast_xy, 4326)
        raster_from_numpy(self.rast3, rast_z_3, rast_xy, 4326)
        

    def tearDown(self):
        shutil.rmtree(self.tdir)


    @unittest.skipIf(IS_WINDOWS, 'Not supported due to LowLevelFunction int')
    def test_avg_filter_nomask(self):
        try:
            rast = ocsmesh.Raster(self.rast1)
            rast.average_filter(size=17)
            self.assertTrue(np.all(rast.get_values() == 10))
        finally:
            del rast


    @unittest.skipIf(IS_WINDOWS, 'Not supported due to LowLevelFunction int')
    def test_avg_filter_masked_nanfill(self):
        try:
            rast = ocsmesh.Raster(self.rast2)
            rast.average_filter(size=17)
            self.assertTrue(
                np.all(rast.values[~np.isnan(rast.values)] == 10))
        finally:
            del rast


    @unittest.skipIf(IS_WINDOWS, 'Not supported due to LowLevelFunction int')
    def test_avg_filter_masked_nonnanfill(self):
        try:
            rast = ocsmesh.Raster(self.rast3)
            rast.average_filter(size=17)
            self.assertTrue(
                np.all(rast.values[rast.values != rast.nodata] == 10))
        finally:
            del rast
