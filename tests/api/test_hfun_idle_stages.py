"""Refinement stages with nothing to apply are skipped.

The contour, channel, patch and line-feature stages built one task per
raster even when no refinement of that kind was added, so in parallel
and MPI modes every tile was rebuilt and rewritten unchanged: about
13 s per idle stage on the NJ depth-bands run.

Run serially (serial, parallel) and under ``mpiexec -n 2`` or more (mpi).
"""

import unittest
from unittest import mock

from ocsmesh.hfun.collector import HfunCollector
from tests.api import multitile


def _patch_only(hfun):
    multitile.add_depth_bands(hfun)
    hfun.add_patch(shape=multitile.default_aoi().buffer(-0.05),
                   target_size=500.0, expansion_rate=None)


class TestIdleStagesSkipped(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.stack = multitile.build_stack(scale=4.0, include_empty=False)
        cls.paths = cls.stack.paths[:2]

    @classmethod
    def tearDownClass(cls):
        multitile.cleanup(cls.stack)

    def _dispatched(self, mode, configure):
        hfun = multitile.make_hfun(self.paths, mode, configure=configure)
        with mock.patch.object(
                HfunCollector, '_dispatch_refinement_tasks', autospec=True,
                side_effect=HfunCollector._dispatch_refinement_tasks) as spy:
            hfun._apply_features()  # pylint: disable=protected-access
        return [c.args[1] for c in spy.call_args_list]

    def test_only_stages_with_work_dispatch(self):
        for mode in multitile.available_modes():
            kinds = self._dispatched(mode, _patch_only)
            if mode == 'serial':
                expected = []
            else:
                expected = ['const_val', 'patch']
            with self.subTest(mode=mode):
                self.assertEqual(kinds, expected)

    def test_same_result_as_serial(self):
        reference = multitile.refined_rasters(
            self.paths, 'serial', configure=_patch_only)
        for mode in multitile.available_modes()[1:]:
            other = multitile.refined_rasters(
                self.paths, mode, configure=_patch_only)
            if multitile.is_manager():
                with self.subTest(mode=mode):
                    self.assertEqual(
                        multitile.compare_rasters(reference, other), '')


if __name__ == '__main__':
    unittest.main()
