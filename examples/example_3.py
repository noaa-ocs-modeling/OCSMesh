#!/usr/bin/env python
import os
# import numpy as np
# from osgeo import gdal
import matplotlib.pyplot as plt
from geomesh import DatasetCollection, \
                    PlanarStraightLineGraph, \
                    SizeFunction, \
                    Jigsaw


def main():

    # ------- init test DEM files
    file = os.getenv('COASTAL_ACT_POST_SANDY_DEM_DIR')
    file1 = os.path.abspath(file + '/zip19/ncei19_n41x00_w074x00_2015v1.tif')
    file2 = os.path.abspath(file + '/zip19/ncei19_n41x00_w073x75_2015v1.tif')

    # ------- init test DatasetCollection object
    dsc = DatasetCollection()
    dsc.add_dataset(file1)
    dsc.add_dataset(file2)

    # ------- generate PSLG
    pslg = PlanarStraightLineGraph(-1500, 15.)
    # pslg.make_plot(show=True)

    # ------- generate size function
    hfun = SizeFunction()

    # ------- init jigsaw and set options
    jigsaw = Jigsaw(
        pslg,
        hfun
    )
    jigsaw.verbosity = 1
    jigsaw._opts.hfun_hmax = 50.
    jigsaw._opts.hfun_scal = 'absolute'
    # jigsaw._opts.mesh_top1 = True
    jigsaw._opts.optm_qlim = .95

    # ------- run jigsaw, get mesh
    mesh = jigsaw.run()

    # ------- interpolate bathymtery to output mesh
    mesh.interpolate(fix_invalid=True)
    fig = plt.figure()
    axes = fig.add_subplot(111)
    mesh.make_plot(axes=axes)
    axes.triplot(mesh.mpl_tri, linewidth=0.07, color='k')
    plt.show()


if __name__ == "__main__":
    main()
