#!/usr/bin/env python
import os
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
from geomesh import PlanarStraightLineGraph, Jigsaw


def main():
    ds = gdal.Open(os.getenv('PR_1s'))
    h0 = 500.
    res = h0/np.sqrt(2.)
    ds = gdal.Warp('', ds, format='VRT', xRes=res, yRes=res)
    PSLG = PlanarStraightLineGraph.from_Dataset(ds, -1500, 10.)
    jigsaw = Jigsaw(PSLG)
    jigsaw.opts.hfun_hmax = h0
    jigsaw.opts.hfun_scal = 'absolute'
    jigsaw.opts.mesh_top1 = True
    jigsaw.opts.optm_qlim = .95
    jigsaw.opts.verbosity = 1
    mesh = jigsaw.run()
    mesh.interpolate(ds)
    # ax = mesh.make_plot()
    # ax.triplot(mesh.mpl_tri, linewidth=0.07, color='k')
    # plt.show()
    mesh.write_gr3('./PR_1s.gr3')
    # os.remove('./PR_1s.gr3')


if __name__ == "__main__":
    main()
