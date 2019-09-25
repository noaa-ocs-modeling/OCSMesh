#!/usr/bin/env python
import os
import matplotlib.pyplot as plt
from geomesh import DatasetCollection, \
                    PlanarStraightLineGraph, \
                    SizeFunction, \
                    Jigsaw
import numpy as np
try:
    import colored_traceback
    colored_traceback.add_hook(always=True)
except ModuleNotFoundError:
    pass


def main():

    zmin = -36.
    z0 = 0.
    zmax = 15.
    zmin_size = 86.
    z0_size = 50.
    zmax_size = 65.
    num_levels = 5  # int

    # ------- init test DEM files
    data = os.path.dirname(os.path.abspath(__file__)) + '/data'
    file1 = os.path.abspath(data + '/ncei19_n41x00_w074x00_2015v1.tif')
    file2 = os.path.abspath(data + '/ncei19_n41x00_w073x75_2015v1.tif')

    # ------- init test DatasetCollection object
    dsc = DatasetCollection()
    dsc.add_dataset(file1)
    dsc.add_dataset(file2)

    # ------- generate PSLG
    pslg = PlanarStraightLineGraph(dsc, zmin, zmax)
    # pslg.make_plot(show=True)

    # ------- generate size function
    hfun = SizeFunction(pslg)
    hfun.add_contour(zmax, zmax_size)
    levels = np.linspace(z0, zmin, num_levels)
    values = np.linspace(z0_size, zmin_size, num_levels)
    for i in range(num_levels):
        hfun.add_contour(levels[i], values[i])
    hfun.make_plot(show=True)

    # ------- init jigsaw and set options
    jigsaw = Jigsaw(pslg, hfun)
    jigsaw.verbosity = 1

    # ------- run jigsaw, get mesh
    mesh = jigsaw.run()

    # ------- interpolate bathymtery to output mesh
    mesh.interpolate(dsc, fix_invalid=True)
    # fig = plt.figure()
    # axes = fig.add_subplot(111)
    # mesh.make_plot(axes=axes)
    # axes.triplot(mesh.mpl_tri, linewidth=0.07, color='k')
    # plt.show()
    print("NP={}".format(mesh.values.size))
    print("elements={}".format(mesh.elements.shape[0]))
    # -------- write to disk
    mesh.dump(
        os.path.dirname(os.path.abspath(__file__)) + '/example_2.gr3',
        overwrite=True)


if __name__ == "__main__":
    main()
