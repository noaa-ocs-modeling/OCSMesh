#!/usr/bin/env python
import os
import subprocess
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from geomesh import RasterCollection, \
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

    # ------- init = RasterCollection object
    dsc = RasterCollection()
    dsc.add_dataset(file)

    # ------- generate PSLG
    pslg = PlanarStraightLineGraph(dsc, -1500., 15.)
    # pslg.make_plot(show=True)

    # ------- generate size function
    hfun = SizeFunction(pslg)
    hfun.add_contour(0., 50., 0.001, hmax=1500.)
    hfun.add_subtidal_flow_limiter(hmin=50., hmax=1500.)
    # print(hfun.values)
    # hfun.make_plot(show=True)

    # ------- init jigsaw and set options
    jigsaw = Jigsaw(
        pslg,
        hfun
    )
    jigsaw.verbosity = 1
    # jigsaw._opts.optm_qlim = .95

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
