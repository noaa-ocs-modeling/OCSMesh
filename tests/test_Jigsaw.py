import unittest
import os
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from geomesh import PlanarStraightLineGraph
from geomesh.drivers import Jigsaw


class JigsawTestCase(unittest.TestCase):

    def setUp(self):
        self.ds = gdal.Open(os.getenv('PR_1s'))
        h0 = 500.
        res = h0/np.sqrt(2.)
        ds = gdal.Warp('', self.ds, format='VRT', xRes=res, yRes=res)
        PSLG = PlanarStraightLineGraph.from_Dataset(ds, -1500, 10.)
        jigsaw = Jigsaw(PSLG)
        jigsaw.opts.hfun_hmax = h0
        jigsaw.opts.hfun_scal = 'absolute'
        jigsaw.opts.mesh_top1 = True
        jigsaw.opts.optm_qlim = .95
        jigsaw.opts.verbosity = 1
        self.mesh = jigsaw.run()
        self.mesh.interpolate(self.ds)

    def _test_interpolate(self):
        self.mesh.make_plot()
        plt.show()

    def _test_fix_invalid(self):
        self.mesh.fix_invalid()
        ax = self.mesh.make_plot()
        ax.tricontour(self.mesh.mpl_tri, self.mesh.values, levels=[0])
        ax.triplot(self.mesh.mpl_tri, linewidth=0.1, color='k')
        plt.show()

    def _test_interpolate_single_tile(self):
        self.mesh._values = None
        ds = gdal.Open(os.getenv('ncei19_n18x00_w066x75_2019v1'))
        self.mesh.interpolate(ds)
        ax = self.mesh.make_plot()
        ax.triplot(self.mesh.mpl_tri, linewidth=0.05, color='k')
        plt.show()

    def _test_ocean_boundaries(self):
        self.mesh.ocean_boundaries

    def _test_land_boundaries(self):
        self.mesh.fix_invalid()
        self.mesh.land_boundary

    def _test_SpatialReference(self):
        self.mesh.SpatialReference = 4326
        self.mesh.make_plot()
        plt.show()

    def _test_compute_planar_straight_line_graph(self):
        self.mesh.planar_straight_line_graph.make_plot()

    def _test_ocean_boundary(self):
        self.mesh.fix_invalid()
        self.mesh.ocean_boundary
        # plt.scatter(self.mesh.x[idxs], self.mesh.y[idxs])
        # plt.show()

    def test_write_fort14(self):
        mesh = self.mesh
        mesh.fix_invalid()
        # mesh.SpatialReference = 4326
        fort14 = "PRUSVI Triangle test\n"
        fort14 += "{}  {}\n".format(mesh.num_elements, mesh.num_nodes)
        for i in range(mesh.num_nodes):
            fort14 += "{:d} ".format(mesh.node_id[i])
            fort14 += "{:<.16E} ".format(mesh.x[i])
            fort14 += " {:<.16E} ".format(mesh.y[i])
            fort14 += "{:<.16E}\n".format(-mesh.values[i])
        for i in range(mesh.num_elements):
            fort14 += "{:d} ".format(mesh.element_id[i])
            fort14 += "{:d} ".format(3)
            fort14 += "{:d} ".format(mesh.elements[i, 0]+1)
            fort14 += "{:d} ".format(mesh.elements[i, 1]+1)
            fort14 += "{:d}\n".format(mesh.elements[i, 2]+1)
        fort14 += "{:d}".format(1)
        fort14 += " ! Number of open boundaries\n"
        fort14 += "{:d}".format(len(mesh.ocean_boundary))
        fort14 += " ! Total number of open boundary nodes\n"
        fort14 += "{:d}".format(len(mesh.ocean_boundary))
        fort14 += " ! Number of nodes for open boundary\n"
        for i in range(len(mesh.ocean_boundary)):
            fort14 += "{:d}\n".format(i+1)
        fort14 += "{:d} ! Number of land boundaries\n".format(2)
        fort14 += "{:d} ! Total number of land boundary nodes\n".format(
            len(mesh.land_boundary) + len(mesh.inner_boundary))
        fort14 += "{:d} 20".format(len(mesh.land_boundary))
        fort14 += " ! Number of nodes for land boundary\n"
        for i in range(len(mesh.land_boundary)):
            fort14 += "{:d}\n".format(i+1)
        fort14 += "{:d} 21".format(len(mesh.inner_boundary))
        fort14 += " ! Number of nodes for inner boundary\n"
        for i in range(len(mesh.inner_boundary)):
            fort14 += "{:d}\n".format(i+1)
        fort14 += "{}\n".format(mesh.SpatialReference.ExportToWkt())
        with open(os.getenv('OUTPUT_FORT14'), 'w') as f:
            f.write(fort14)


if __name__ == '__main__':
    unittest.main()
