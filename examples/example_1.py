#!/usr/bin/env python
import os
import subprocess
import time
import matplotlib.pyplot as plt
from geomesh import Raster, \
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
    rootdir = os.path.dirname(os.path.abspath(__file__))
    datadir = rootdir + '/data'
    file = os.path.abspath(datadir + '/PR_1s.tif')

    # ------- init Raster object
    rast = Raster(file)

    # ------- init PSLG object
    pslg = PlanarStraightLineGraph(
        rast,
        # zmin=-3000,
        zmax=15.)

    # ------- init size function
    hfun = SizeFunction(pslg, hmin=50., hmax=2000., verbosity=1)
    hfun.add_contour(0., 0.01)
    hfun.add_subtidal_flow_limiter()
    hfun.add_gaussian_filter(6)

    # ------- init jigsaw and set options
    jigsaw = Jigsaw(hfun)
    jigsaw.verbosity = 1

    # ------- run jigsaw, get mesh
    mesh = jigsaw.run()

    # ------- interpolate bathymetry to output mesh
    mesh.interpolate(rast, fix_invalid=True)

    # ------- reproject mesh to WGS84
    mesh.transform_to('EPSG:4326')

    # ------- visualize results
    fig = plt.figure(figsize=(15, 11))
    axes = fig.add_subplot(111)
    mesh.make_plot(axes=axes)
    axes.triplot(mesh.triangulation, linewidth=0.07, color='k')
    plt.savefig(rootdir + '/example_1.png', bbox_inches='tight')
    plt.show()

    # -------- write to disk
    # fname = os.path.abspath(rootdir + '/example_1.grd')
    # mesh.save(fname, overwrite=True)


if __name__ == "__main__":
    main()
