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
    start = time.time()
    # ------- init test DEM files
    rootdir = os.path.dirname(os.path.abspath(__file__))
    datadir = rootdir + '/data'
    subprocess.check_call(["git", "submodule", "update", "--init", datadir])
    file = os.path.abspath(datadir + '/PR_1s.tif')

    # ------- init Raster object
    rast = Raster(file)

    # ------- init PSLG object
    pslg = PlanarStraightLineGraph(rast, zmax=15.)

    # ------- visualize PSLG object
    # pslg.plot(show=True)

    # ------- init size function
    hfun = SizeFunction(pslg, hmin=50., verbosity=1)

    # ------- add size function constraints
    hfun.add_contour(0., 0.001)
    hfun.add_subtidal_flow_limiter()
    tri, values = hfun(0)
    # plt.title('Size Function')
    # plt.tricontourf(tri, values, levels=256, cmap='jet')
    # plt.triplot(tri, linewidth=0.2, color='k')
    # plt.gca().axis('scaled')
    # plt.show()
    # plt.close(plt.gcf())
    from geomesh import Mesh
    import numpy as np
    vertices = np.vstack([tri.x, tri.y]).T
    from scipy.interpolate import RectBivariateSpline
    f = RectBivariateSpline(
        rast.x,
        np.flip(rast.y),
        np.flipud(rast.values).T,
        )
    values = f.ev(vertices[:, 0], vertices[:, 1])
    mesh = Mesh(vertices, tri.triangles, 'EPSG:3395', values=values)
    mesh.transform_to('EPSG:4326')
    # mesh.make_plot(show=True)
    mesh.save('PuertoRicoTestCase.grd', overwrite=True)
    # bbox = raster.bbox
    # mesh.interpolate(rast, fix_invalid=True)
    # mesh.tricontourf(show=True, levels=256)
    # import numpy as np

    exit()
    # hfun.tripcolor(show=True)

    # ------- init jigsaw and set options
    jigsaw = Jigsaw(hfun)
    jigsaw.verbosity = 1

    # ------- run jigsaw, get mesh
    jig_start = time.time()
    mesh = jigsaw.run()
    print(f"Jigsaw took {time.time()-jig_start} seconds.")

    # ------- interpolate bathymetry to output mesh
    mesh.interpolate(rast, fix_invalid=True)

    # ------- reproject mesh to WGS84
    mesh.transform_to('EPSG:4326')

    # ------- visualize results
    fig = plt.figure()
    axes = fig.add_subplot(111)
    mesh.make_plot(axes=axes)
    axes.triplot(mesh.triangulation, linewidth=0.07, color='k')
    plt.savefig('pr_mesh.png')

    # -------- write to disk
    fname = os.path.abspath(rootdir + '/example_1.grd')
    mesh.save(fname, overwrite=True)
    print(f"The whole thing took {time.time()-start} seconds.")


if __name__ == "__main__":
    main()
