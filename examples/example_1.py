#!/usr/bin/env python
import os
import subprocess
import matplotlib.pyplot as plt
from geomesh import DatasetCollection, \
                    PlanarStraightLineGraph, \
                    Jigsaw  # SizeFunction, \
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
    # hfun = SizeFunction()

    # ------- init jigsaw and set options
    jigsaw = Jigsaw(
        pslg,
        # hfun
    )
    jigsaw.verbosity = 1
    jigsaw._opts.hfun_hmax = 150.
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
    axes.triplot(mesh.mpl_tri, linewidth=0.07, color='k')
    plt.show()


if __name__ == "__main__":
    main()
