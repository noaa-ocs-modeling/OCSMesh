'''Test pickling and unpickling of HfunRaster objects using unittest framework.'''
import unittest
import platform
import tempfile
import shutil
import pickle
from pathlib import Path
import numpy as np
import numpy.testing as npt
import ocsmesh
from ocsmesh.utils import raster_from_numpy


IS_WINDOWS = platform.system() == 'Windows'


def get_value(rast_obj):
    """Return values from a raster object."""
    return rast_obj.get_values()


class TestRasterPickling(unittest.TestCase):
    """Test pickling and unpickling of Raster objects."""
    def setUp(self):
        """Set up a temporary directory and a representative raster file."""
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


    @unittest.skipIf(IS_WINDOWS, 'Pickle tests not guaranteed stable on Windows due to I/O issues')
    def test_hfunRaster_type(self):
        """Test pickling and unpickling of HfunRaster objects."""
        hfun_original = ocsmesh.Hfun(
                ocsmesh.Raster(self.rast1),
                hmin=500,
                hmax=10000
            )
        hfun_pickled=pickle.loads(pickle.dumps(hfun_original))
        self.assertIsInstance(hfun_pickled, ocsmesh.hfun.raster.HfunRaster)
        self.assertIsInstance(hfun_pickled.raster, ocsmesh.Raster)


    @unittest.skipIf(IS_WINDOWS, 'Pickle tests not guaranteed stable on Windows due to I/O issues')
    def test_tmpfile_path(self):
        """Test pickling and unpickling of HfunRaster objects."""
        hfun_original = ocsmesh.Hfun(
                ocsmesh.Raster(self.rast1),
                hmin=500,
                hmax=10000
            )
        hfun_pickled=pickle.loads(pickle.dumps(hfun_original))
        self.assertNotEqual(hfun_original.tmpfile,hfun_pickled.tmpfile)
        self.assertEqual(hfun_original.raster.tmpfile,hfun_pickled.raster.tmpfile)


    @unittest.skipIf(IS_WINDOWS, 'Pickle tests not guaranteed stable on Windows due to I/O issues')
    def test_values_nofilldata(self):
        """Test pickling and unpickling of HfunRaster objects."""
        hfun_original = ocsmesh.Hfun(
                ocsmesh.Raster(self.rast2),
                hmin=500,
                hmax=10000
            )
        hfun_original.raster.fill_nodata()
        values1=hfun_original.raster.get_values()
        hfun_pickled=pickle.loads(pickle.dumps(hfun_original))
        values2=hfun_pickled.raster.get_values()
        npt.assert_array_equal(values1,values2)


if __name__ == '__main__':
    unittest.main()
