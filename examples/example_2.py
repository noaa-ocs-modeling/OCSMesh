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

    # ------- make list of file paths to process
    rootdir = os.path.dirname(os.path.abspath(__file__))
    datadir = rootdir + '/data'
    files = [
        'ncei13_n41x00_w074x00_2015v1.tif',
        # 'ncei19_n41x00_w073x75_2015v1.tif',
    ]
    files = [os.path.abspath(datadir + '/' + file) for file in files]

    # ------- init RasterCollection object
    rc = RasterCollection(files)

    # ------- init PSLG
    pslg = PlanarStraightLineGraph(rc, -1500., 15., nproc=1)
    # pslg.plot(show=True)
    # pslg.triplot(show=True)

    # ------- generate size function
    hfun = SizeFunction(pslg, 50., 1500., nproc=1)
    # hfun.add_contour(0., 0.001)
    hfun.add_subtidal_flow_limiter()
    hfun.memmap_elements
    # hfun.tricontourf(cmap='jet')
    # hfun.triplot(show=True)

    # ------- init jigsaw
    jigsaw = Jigsaw(hfun)

    # ------- set additional jigsaw options
    jigsaw.verbosity = 1

    # ------- run jigsaw, get mesh
    mesh = jigsaw.run()

    # ------- interpolate bathymetry to output mesh
    mesh.interpolate_collection(rc, fix_invalid=True)

    # ------- reproject mesh to WGS84
    mesh.transform_to('EPSG:4326')

    # ------- visualize results
    fig = plt.figure()
    axes = fig.add_subplot(111)
    mesh.make_plot(axes=axes)
    axes.triplot(mesh.triangulation, linewidth=0.07, color='k')
    plt.show()
    print("NP={}".format(mesh.values.size))
    print("elements={}".format(mesh.elements.shape[0]))

    # -------- write to disk
    fname = os.path.abspath(rootdir + '/example_2.grd')
    mesh.save(fname, overwrite=True)


if __name__ == "__main__":
    main()
