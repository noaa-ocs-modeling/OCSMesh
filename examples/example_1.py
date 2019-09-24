#!/usr/bin/env python
import os
import subprocess
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
    subprocess.check_call(["git", "submodule", "update", "--init", data])
    file = os.path.abspath(data + '/PR_1s.tif')

    # ------- init test DatasetCollection object
    dsc = DatasetCollection()
    dsc.add_dataset(file)

    # ------- generate PSLG
    pslg = PlanarStraightLineGraph(dsc, zmin=-1500, zmax=15.)
    # pslg.make_plot(show=True)

    # ------- generate size function
    hfun = SizeFunction(pslg)
    hfun.set_shoreline(25.)
    hfun.set_ocean_boundaries(1000.)
    hfun.set_land_boundaries(100.)
    hfun.set_inner_boundaries(50.)
    # ------- init jigsaw and set options
    jigsaw = Jigsaw(pslg, hfun)
    jigsaw.verbosity = 1

    # ------- run jigsaw, get mesh
    mesh = jigsaw.run()

    # ------- interpolate bathymetry to output mesh
    mesh.interpolate(dsc, fix_invalid=True)
    fig = plt.figure()
    axes = fig.add_subplot(111)
    mesh.make_plot(axes=axes)
    axes.triplot(mesh.mpl_tri, linewidth=0.07, color='k')
    plt.show()


if __name__ == "__main__":
    main()
