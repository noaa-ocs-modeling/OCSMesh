"""Contour and channel detection distributed per raster.

Detection ran on the coordinator, one raster after the other, even in
parallel and MPI modes; on the NJ channels run it took 99% of the time.
It is now one task per definition and raster, and the shapes are
collected in the serial order, so the refinements are identical.

Run serially (serial, parallel) and under ``mpiexec -n 2`` or more (mpi).
"""

import unittest
from unittest import mock

from ocsmesh.hfun.collector import HfunCollector
from tests.api import multitile


def _two_of_each(hfun):
    multitile.add_depth_bands(hfun)
    hfun.add_contour(level=0.0, expansion_rate=0.01, target_size=500.0)
    hfun.add_contour(level=-20.0, expansion_rate=0.02, target_size=800.0)
    hfun.add_channel(level=0.0, width=2000.0, target_size=500.0,
                     expansion_rate=0.01, tolerance=50)
    hfun.add_channel(level=-10.0, width=1500.0, target_size=700.0,
                     expansion_rate=None, tolerance=None)


class TestDistributedDetection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.stack = multitile.build_stack(scale=4.0, include_empty=False)
        cls.paths = cls.stack.paths[:3]

    @classmethod
    def tearDownClass(cls):
        multitile.cleanup(cls.stack)

    def test_same_refinement_as_serial(self):
        reference = multitile.refined_rasters(
            self.paths, 'serial', configure=_two_of_each)
        for mode in multitile.available_modes()[1:]:
            other = multitile.refined_rasters(
                self.paths, mode, configure=_two_of_each)
            if multitile.is_manager():
                with self.subTest(mode=mode):
                    self.assertEqual(
                        multitile.compare_rasters(reference, other), '')

    def test_one_task_per_definition_and_raster(self):
        for mode in multitile.available_modes()[1:]:
            hfun = multitile.make_hfun(self.paths, mode,
                                       configure=_two_of_each)
            with mock.patch.object(
                    HfunCollector, '_dispatch_refinement_tasks',
                    autospec=True,
                    side_effect=HfunCollector._dispatch_refinement_tasks
                    ) as spy:
                hfun._apply_features()  # pylint: disable=protected-access
            if not multitile.is_manager():
                continue
            sizes = {c.args[1]: len(c.args[2]) for c in spy.call_args_list
                     if c.args[1].endswith('_detect')}
            with self.subTest(mode=mode):
                n = len(self.paths)
                self.assertEqual(sizes, {'contour_detect': 2 * n,
                                         'channel_detect': 2 * n})


if __name__ == '__main__':
    unittest.main()
