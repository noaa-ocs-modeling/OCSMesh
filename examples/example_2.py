#!/usr/bin/env python
# import os
# import subprocess
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from geomesh import GdalDataset, \
                    DatasetCollection, \
                    PlanarStraightLineGraph, \
                    SizeFunction, \
                    Jigsaw
import colored_traceback
colored_traceback.add_hook(always=True)


def main():

    # ------- global options
    hmin = 1.
    h0 = 50.
    hmax = 500.
    zmin = -3000.
    zmax = 15.

    # ------- init DatasetCollection
    path = Path(str(Path.home()) + '/postSandyDEM')
    dsc = DatasetCollection()
    for file in path.glob('**/*.tif'):
        ds = GdalDataset(file, feature_size=h0)
        dsc.add_dataset(ds)

    # ------- generate PSLG
    pslg = PlanarStraightLineGraph(dsc, zmin, zmax)
    # pslg.SpatialReference = 4326
    pslg.make_plot(show=True)

    # ------- generate size function
    # hfun = SizeFunction(pslg)
    # print(hfun.values)
    # print(hfun.ocean_boundaries)
    # BREAKME
    # hfun.make_plot(show=True)

    # ------- input mesh
    #

    # ------- init jigsaw and set options
    jigsaw = Jigsaw(
        pslg,
        # hfun,
    )
    jigsaw.verbosity = 1
    # jigsaw._opts.hfun_hmin = hmin
    jigsaw._opts.hfun_hmax = hmax
    jigsaw._opts.hfun_scal = 'absolute'
    # jigsaw._opts.mesh_top1 = True
    jigsaw._opts.optm_qlim = .95

    # ------- run jigsaw, get mesh
    mesh = jigsaw.run()

    # ------- interpolate bathymtery to output mesh
    mesh.interpolate(dsc, fix_invalid=True)
    fig = plt.figure()
    axes = fig.add_subplot(111)
    mesh.make_plot(axes=axes)
    axes.triplot(mesh.mpl_tri, linewidth=0.05, color='k')
    plt.show()


if __name__ == "__main__":
    main()
