'''Test pickling and unpickling of HfunRaster objects using unittest framework.'''
import unittest
import tempfile
import shutil
import pickle
from pathlib import Path
import numpy as np
import numpy.testing as npt
import ocsmesh
from ocsmesh.utils import raster_from_numpy

class TestRasterPickling(unittest.TestCase):
    """Test pickling and unpickling of Raster objects."""
    def setUp(self):
        """Set up a temporary directory and a representative raster file."""
        self.tdir = Path(tempfile.mkdtemp())
        self.rast = self.tdir / 'rast_1.tif'

        rast_xy = np.mgrid[-1:1:0.1, -0.7:0.7:0.1]
        rast_z = np.random.rand(*rast_xy[0].shape) * 10

        raster_from_numpy(self.rast, rast_z, rast_xy, 4326)

        self.hfun_original = ocsmesh.Hfun(
                ocsmesh.Raster(self.rast),
                hmin=500,
                hmax=10000
            )

        self.hfun_pickled=pickle.loads(pickle.dumps(self.hfun_original))


    def tearDown(self):
        shutil.rmtree(self.tdir)


    def test_path(self):
        """Test pickling and unpickling of HfunRaster objects."""
        self.assertIsInstance(self.hfun_pickled, ocsmesh.hfun.raster.HfunRaster)
        self.assertIsInstance(self.hfun_pickled.raster, ocsmesh.Raster)


    def test_tmpfile_path(self):
        """Test pickling and unpickling of HfunRaster objects."""
        self.assertEqual(self.hfun_original.tmpfile,self.hfun_pickled.tmpfile)
        self.assertEqual(self.hfun_original.source.name,self.hfun_pickled.source.name)


    def test_data(self):
        """Test pickling and unpickling of HfunRaster objects."""
        self.assertEqual(self.hfun_original._hmin, self.hfun_pickled._hmin)
        self.assertEqual(self.hfun_original._hmax, self.hfun_pickled._hmax)
        original_values = self.hfun_original.raster.get_values()
        pickled_values = self.hfun_pickled.raster.get_values()
        npt.assert_array_equal(original_values, pickled_values)


    def test_coordinates(self):
        """Test pickling and unpickling of HfunRaster objects."""
        npt.assert_array_equal(self.hfun_original.raster.x, self.hfun_pickled.raster.x)
        npt.assert_array_equal(self.hfun_original.raster.y, self.hfun_pickled.raster.y)
        self.assertEqual(self.hfun_original.raster.crs, self.hfun_pickled.raster.crs)


if __name__ == '__main__':
    unittest.main()
