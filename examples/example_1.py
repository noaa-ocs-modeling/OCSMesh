#!/usr/bin/env python
import os
import subprocess
import numpy as np
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

    zmin = -1500.
    z0 = 0.
    zmax = 15.
    zmin_size = 50.
    zmax_size = 1000.
    num_levels = 6  # int >= 3

    # ------- init test DEM files
    data = os.path.dirname(os.path.abspath(__file__)) + '/data'
    subprocess.check_call(["git", "submodule", "update", "--init", data])
    file = os.path.abspath(data + '/PR_1s.tif')

    # ------- init test DatasetCollection object
    dsc = DatasetCollection()
    dsc.add_dataset(file)

    # ------- generate PSLG
    pslg = PlanarStraightLineGraph(dsc, zmin, zmax)
    # pslg.make_plot(show=True)

    # ------- generate size function
    hfun = SizeFunction(pslg)
    hfun.add_contour(zmax, zmin_size)
    levels = np.linspace(z0, zmin, num_levels)
    values = np.linspace(zmin_size, zmax_size, num_levels)
    for i in range(num_levels):
        hfun.add_contour(levels[i], values[i])
    # hfun.make_plot(show=True)

    # ------- init jigsaw and set options
    jigsaw = Jigsaw(pslg, hfun)
    jigsaw.verbosity = 1

    # ------- run jigsaw, get mesh
    mesh = jigsaw.run()

    # ------- interpolate bathymetry to output mesh
    mesh.interpolate(dsc, fix_invalid=True)
    # fig = plt.figure()
    # axes = fig.add_subplot(111)
    # mesh.make_plot(axes=axes)
    # axes.triplot(mesh.mpl_tri, linewidth=0.07, color='k')
    # plt.show()

    # ------- write gr3 file
    print("NP={}".format(mesh.values.size))
    print("elements={}".format(mesh.elements.shape[0]))
    mesh.dump(
        os.path.dirname(os.path.abspath(__file__)) + '/PR_1s.gr3',
        overwrite=True)


if __name__ == "__main__":
    main()
