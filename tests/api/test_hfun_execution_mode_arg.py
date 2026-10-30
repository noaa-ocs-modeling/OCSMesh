"""HfunCollector(execution_mode=...) at construction.

The mode used to be settable only after construction. Under an MPI
launcher the ranks then always shared one collector (rank 0 holding the
rasters), so a script running an independent serial collector on every
rank got no rasters on ranks 1 and up.

Run serially and under ``mpiexec -n 2`` or more.
"""

import unittest

import numpy as np

from ocsmesh import Hfun
from tests.api import multitile


def _stats(md):
    values = np.asarray(md.values).ravel()
    return len(values), float(values.mean())


class TestExecutionModeArgument(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.stack = multitile.build_stack(scale=4.0, include_empty=False)
        cls.paths = cls.stack.paths[:2]

    @classmethod
    def tearDownClass(cls):
        multitile.cleanup(cls.stack)

    def _hfun(self, **kwargs):
        hfun = Hfun(list(self.paths), hmin=multitile.HMIN,
                    hmax=multitile.HMAX, **kwargs)
        multitile.add_depth_bands(hfun)
        return hfun

    def test_same_as_property(self):
        reference = self._hfun().meshdata()   # shared, serial by default
        for mode in multitile.available_modes():
            nprocs = 2 if mode == 'parallel' else 1
            hfun = self._hfun(execution_mode=mode, nprocs=nprocs)
            with self.subTest(mode=mode):
                self.assertEqual(hfun.execution_mode, mode)
                md = hfun.meshdata()
                if multitile.is_manager():
                    self.assertEqual(_stats(md), _stats(reference))

    def test_serial_collector_on_every_rank(self):
        """Every rank gets the full result from its own collector."""
        reference = self._hfun(execution_mode='serial')
        md = reference.meshdata()
        self.assertIsNotNone(md)
        # pylint: disable=protected-access
        self.assertEqual(len(reference._hfun_list), len(self.paths))
        comm = multitile.mpi_comm()
        if comm is None:
            return
        stats = comm.allgather(_stats(md))
        work_dirs = comm.allgather(reference._work_dir)
        self.assertEqual(len(set(stats)), 1, stats)
        self.assertEqual(len(set(work_dirs)), comm.Get_size())

    def test_per_rank_collector_cannot_switch_to_mpi(self):
        if 'mpi' not in multitile.available_modes():
            self.skipTest('needs mpiexec -n 2 or more')
        hfun = self._hfun(execution_mode='serial')
        with self.assertRaises(ValueError):
            hfun.execution_mode = 'mpi'


if __name__ == '__main__':
    unittest.main()
