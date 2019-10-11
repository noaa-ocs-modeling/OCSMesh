#!/usr/bin/env python
import os
import matplotlib.pyplot as plt
from geomesh import RasterCollection, \
                    PlanarStraightLineGraph, \
                    SizeFunction, \
                    Jigsaw
# import numpy as np
try:
    import colored_traceback
    colored_traceback.add_hook(always=True)
except ModuleNotFoundError:
    pass


def main():

    # ------- init test DEM files
    data = os.path.dirname(os.path.abspath(__file__)) + '/data'
    file1 = os.path.abspath(data + '/ncei19_n41x00_w074x00_2015v1.tif')
    file2 = os.path.abspath(data + '/ncei19_n41x00_w073x75_2015v1.tif')

    # ------- init test RasterCollection object
    dsc = RasterCollection()
    dsc.add_dataset(file1)
    dsc.add_dataset(file2)

    # ------- generate PSLG
    pslg = PlanarStraightLineGraph(dsc, -1500., 15.)
    # pslg.make_plot(show=True)

    # ------- generate size function
    hfun = SizeFunction(pslg)
    hfun.add_contour(0., 50., 0.001, hmax=1500.)
    hfun.add_subtidal_flow_limiter(50., 1500.)
    # hfun.make_plot(show=True)

    # ------- init jigsaw and set options
    jigsaw = Jigsaw(pslg, hfun)
    jigsaw.verbosity = 1

    # ------- run jigsaw, get mesh
    mesh = jigsaw.run()

    # ------- interpolate bathymtery to output mesh
    mesh.interpolate(dsc, fix_invalid=True)
    fig = plt.figure()
    axes = fig.add_subplot(111)
    mesh.make_plot(axes=axes)
    axes.triplot(mesh.mpl_tri, linewidth=0.07, color='k')
    plt.show()
    print("NP={}".format(mesh.values.size))
    print("elements={}".format(mesh.elements.shape[0]))

    # -------- write to disk
    mesh.dump(
        os.path.dirname(os.path.abspath(__file__)) + '/example_2.gr3',
        overwrite=True)


if __name__ == "__main__":
    main()
