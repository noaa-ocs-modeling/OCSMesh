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
    file1 = os.path.abspath(data + '/ncei19_n41x00_w074x00_2015v1.tif')
    file2 = os.path.abspath(data + '/ncei19_n41x00_w073x75_2015v1.tif')

    # ------- init test DatasetCollection object
    dsc = DatasetCollection()
    dsc.add_dataset(file1)
    dsc.add_dataset(file2)

    # ------- generate PSLG
    pslg = PlanarStraightLineGraph(zmin=-1500, zmax=15.)
    # pslg.make_plot(show=True)

    # ------- generate size function
    # hfun = SizeFunction()

    # ------- init jigsaw and set options
    jigsaw = Jigsaw(
        pslg,
        # hfun
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