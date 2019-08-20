#!/usr/bin/env python
import os
import subprocess
# import numpy as np
# from osgeo import gdal
# import matplotlib.pyplot as plt
from geomesh import PlanarStraightLineGraph  # , Jigsaw

# initialize demo data
data = os.path.dirname(os.path.abspath(__file__)) + '/data'
subprocess.check_call(["git", "submodule", "update", "--init", data])


def main():

    pslg = PlanarStraightLineGraph()
    pslg.add_Dataset(os.path.abspath(data + '/PR_1s.tif'))
    pslg.h0 = 500.
    pslg.zmin = -3000.
    pslg.zmax = 20.
    pslg.make_plot(show=True)

    # jigsaw = Jigsaw(pslg)
    # jigsaw.opts.hfun_hmax = h0
    # jigsaw.opts.hfun_scal = 'absolute'
    # jigsaw.opts.mesh_top1 = True
    # jigsaw.opts.optm_qlim = .95
    # jigsaw.opts.verbosity = 1
    # mesh = jigsaw.run()
    # mesh.interpolate(ds)
    # ax = mesh.make_plot()
    # ax.triplot(mesh.mpl_tri, linewidth=0.07, color='k')
    # plt.show()
    # mesh.write_gr3('./PR_1s.gr3')


if __name__ == "__main__":
    main()
