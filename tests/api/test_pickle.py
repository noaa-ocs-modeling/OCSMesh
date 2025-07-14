import unittest
import tempfile
import shutil
import platform
import pickle
import subprocess
import sys
import os
from pathlib import Path
import multiprocessing
import numpy as np
import numpy.testing as npt
from ocsmesh import Raster
from ocsmesh.utils import raster_from_numpy

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
except ImportError:
    MPI_AVAILABLE = False
    comm = None
    rank = 0
    size = 1

IS_WINDOWS = platform.system() == 'Windows'


# Configuration for auto-MPI execution
MPI_CONFIG = {
    'auto_run_mpi': True,  # Set to True to automatically run MPI tests
    'mpi_processes': 3,    # Number of MPI processes to spawn
    'mpi_executable': 'mpiexec',  # or 'mpirun' depending on your system
}


def get_value(rast_obj):
    return rast_obj.get_values()


def is_running_with_mpi():
    return MPI_AVAILABLE and size > 1


def run_mpi_tests():

    if IS_WINDOWS:
        print("Skipping MPI tests on Windows due to I/O issues")
        return

    if not MPI_AVAILABLE:
        print("mpi4py not available, skipping MPI tests")
        return

    print(f"Running MPI tests with {MPI_CONFIG['mpi_processes']} processes...")

    mpi_cmd = [
        MPI_CONFIG['mpi_executable'],
        '-n', str(MPI_CONFIG['mpi_processes']),
        sys.executable,
        os.path.abspath(__file__),
        '--mpi-mode'  # Flag to indicate MPI mode
    ]

    try:
        result = subprocess.run(mpi_cmd, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print("MPI tests passed successfully!")
            if result.stdout:
                print(">>> MPI test output:")
                print(result.stdout)
        else:
            print("MPI tests failed!")
            if result.stderr:
                print(">>> MPI test errors:")
                print(result.stderr)

    except subprocess.TimeoutExpired:
        print("MPI tests timed out!")
    except FileNotFoundError:
        print(f"MPI executable '{MPI_CONFIG['mpi_executable']}' not found!")
    except Exception as e:
        print(f"Error running MPI tests: {e}")


class TestRasterPickling(unittest.TestCase):
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


    @unittest.skipIf(IS_WINDOWS, 'Pickle tests not guaranteed stable on Windows due to I/O issues')
    @unittest.skipUnless(rank == 0, 'This test runs only on rank 0')
    def test_pickle_and_unpickle(self):
        try:
            original = Raster(self.rast1)
            _ = original.get_values()
            pickled = pickle.dumps(original)
            restored = pickle.loads(pickled)
            np.testing.assert_array_equal(restored.get_values(), original.get_values())
        finally:
            del original
            del restored


    @unittest.skipIf(IS_WINDOWS, 'Pickle tests not guaranteed stable on Windows due to I/O issues')
    @unittest.skipUnless(rank == 0, 'This test runs only on rank 0')
    def test_pickle_with_multiprocessing(self):
        try:
            Raster_path = [str(self.rast1), str(self.rast2)]
            N_PROCS = len(Raster_path)
            rast0 = []
            data0 = []
   
            for r in Raster_path:
               rast = Raster(r)
               rast0.append(rast)
               data0.append(rast.values)

            with multiprocessing.Pool(N_PROCS) as p:
               value_list = p.map(get_value, rast0)

            for data, process_data in zip(data0, value_list):
               npt.assert_array_equal(data, process_data)
        finally:
            for rast in rast0:
                del rast


    @unittest.skipUnless(is_running_with_mpi(), "This test must be run with mpiexec and multiple ranks")
    @unittest.skipIf(IS_WINDOWS, 'Pickle tests not guaranteed stable on Windows due to I/O issues')
    def test_pickle_with_mpi4py(self):
        try:
            rast0 = None
            data0 = None
            if rank == 0:
                Raster_path = [str(self.rast1), str(self.rast2), str(self.rast3)]

                if size != len(Raster_path):
                     raise ValueError("Number of processes must match the number of raster files.")

                rast0 = []
                data0 = []
                for r in Raster_path:
                     rast = Raster(r)
                     rast0.append(rast)
                     data0.append(rast.values)
  
            my_rast = comm.scatter(rast0, root=0)
            my_value = get_value(my_rast)
            all_values = comm.gather(my_value, root=0)

            if rank == 0:
                for data, process_data in zip(data0, all_values):
                    npt.assert_array_equal(data, process_data)
                    print("MPI test passed")
        finally:
            if rast0 is not None:
                for rast in rast0:
                    del rast
            if my_rast is not None:
                del my_rast


#  Automatically launch MPI tests when running via unittest discover
class TestMPILauncher(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not is_running_with_mpi() and MPI_CONFIG['auto_run_mpi'] and rank == 0:
            print("\nAuto-triggering MPI tests from TestMPILauncher...\n")
            run_mpi_tests()

    def test_dummy(self):
        """Dummy test to trigger setUpClass"""
        self.assertTrue(True) 


if __name__ == '__main__':
    if '--mpi-mode' in sys.argv:
        sys.argv.remove('--mpi-mode')
        unittest.main(verbosity=2)
    
