"""Contour and channel definitions keep no sources after meshdata().

The collectors copied each definition before attaching the rasters, but
the copy was shallow and shared the source list, so the user's
definition kept the rasters of the first run: old size functions (and
their files) stayed alive, and only rank 0 (which runs detection) had
sources, so the ranks could take different paths on a later run.
"""

import unittest

from tests.api import multitile


def _contour_and_channel(hfun):
    hfun.add_contour(level=0.0, expansion_rate=0.01, target_size=500.0)
    hfun.add_channel(level=0.0, width=2000.0, target_size=500.0,
                     expansion_rate=0.01, tolerance=50)


class TestDefinitionsStaySourceless(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.stack = multitile.build_stack(scale=4.0, include_empty=False)
        cls.paths = cls.stack.paths[:2]

    @classmethod
    def tearDownClass(cls):
        multitile.cleanup(cls.stack)

    def test_after_meshdata(self):
        for mode in multitile.available_modes():
            hfun = multitile.make_hfun(self.paths, mode,
                                       configure=_contour_and_channel)
            hfun.meshdata()
            # pylint: disable=protected-access
            defns = [d for d, _ in hfun._contour_info_coll]
            defns += [d for d, _ in hfun._ch_info_coll]
            with self.subTest(mode=mode):
                self.assertEqual([d.has_source for d in defns],
                                 [False] * len(defns))


if __name__ == '__main__':
    unittest.main()
