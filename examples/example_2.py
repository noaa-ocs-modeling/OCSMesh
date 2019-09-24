#!/usr/bin/env python
import os
import matplotlib.pyplot as plt
from geomesh import DatasetCollection, \
                    PlanarStraightLineGraph, \
                    SizeFunction, \
                    Jigsaw
try:
    import colored_traceback
    colored_traceback.add_hook(always=True)
except ModuleNotFoundError:
    pass


def main():

    # ------- init test DEM files
    data = os.path.dirname(os.path.abspath(__file__)) + '/data'
    file1 = os.path.abspath(data + '/ncei19_n41x00_w074x00_2015v1.tif')
    # file2 = os.path.abspath(data + '/ncei19_n41x00_w073x75_2015v1.tif')

    # ------- init test DatasetCollection object
    dsc = DatasetCollection()
    dsc.add_dataset(file1)
    # dsc.add_dataset(file2)

    # ------- generate PSLG
    pslg = PlanarStraightLineGraph(dsc, zmin=-1500, zmax=15.)
    # pslg.make_plot(show=True)

    # ------- generate size function
    hfun = SizeFunction(pslg)
    hfun.set_shoreline(25.)
    hfun.set_ocean_boundaries(1000.)
    hfun.set_land_boundaries(100.)
    hfun.set_inner_boundaries(50.)
    # hfun.limgrad(0.2)
    # hfun.make_plot(show=True)

    # ------- init jigsaw and set options
    jigsaw = Jigsaw(pslg, hfun)
    jigsaw.verbosity = 1
    # jigsaw._opts.mesh_top1 = True
    # jigsaw._opts.optm_qlim = .95

    # ------- run jigsaw, get mesh
    mesh = jigsaw.run()

    # ------- interpolate bathymtery to output mesh
    mesh.interpolate(dsc, fix_invalid=True)
    fig = plt.figure()
    axes = fig.add_subplot(111)
    mesh.make_plot(axes=axes)
    axes.triplot(mesh.mpl_tri, linewidth=0.07, color='k')
    plt.show()


if __name__ == "__main__":
    main()

