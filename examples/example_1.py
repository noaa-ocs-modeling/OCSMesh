#!/usr/bin/env python
import os
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

import pathlib
import urllib.request
import tempfile
import tarfile


def main():

    # Download example data
    parent = pathlib.Path(__file__).parent / 'data'
    parent.mkdir(exist_ok=True)
    raster = parent / 'PR_1s.tif'
    if not raster.is_file():
        url = "https://www.dropbox.com/s/0duc0tnp43rjgrt/PR_1s.tar.gz?dl=1"
        g = urllib.request.urlopen(url)
        tmpfile = tempfile.NamedTemporaryFile()
        with open(tmpfile.name, 'b+w') as f:
            f.write(g.read())
        with tarfile.open(tmpfile.name, "r:gz") as tar:
            tar.extractall(parent)

    # ------- init Raster object
    rast = Raster(raster)

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
    plt.savefig(parent.parent / 'example_1.png', bbox_inches='tight')
    plt.show()

    # -------- write to disk
    fname = os.path.abspath(parent.parent / 'example_1.grd')
    mesh.save(fname, overwrite=True)


if __name__ == "__main__":
    main()
