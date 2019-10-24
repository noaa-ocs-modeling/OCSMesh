#!/usr/bin/env python
import os
import time
import matplotlib.pyplot as plt
from geomesh import PlanarStraightLineGraph, \
                    SizeFunction, \
                    Jigsaw
try:
    import colored_traceback
    colored_traceback.add_hook(always=True)
except ModuleNotFoundError:
    pass


def main():
    start = time.time()
    # ------- make list of file paths to process
    rootdir = os.path.dirname(os.path.abspath(__file__))
    datadir = rootdir + '/data'
    files = [
        'ncei19_n41x00_w074x00_2015v1.tif',
        # 'ncei13_n41x00_w073x75_2015v1.tif',
    ]
    files = [os.path.abspath(datadir + '/' + file) for file in files]

    # ------- init PSLG
    pslg = PlanarStraightLineGraph(files, zmax=15.)
    # pslg.plot(show=True)

    # ------- generate size function
    hfun = SizeFunction(pslg, 50.)
    hfun.add_contour(0., 0.001, n_jobs=-1)
    hfun.add_subtidal_flow_limiter()
    hfun.add_gaussian_filter(4)
    hfun.tripcolor(show=True, cmap='jet')

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
    print("NP={}".format(mesh.values.size))
    print("elements={}".format(mesh.elements.shape[0]))
    print(f"Took {time.time() - start} seconds.")

    plt.show()
    # -------- write to disk
    fname = os.path.abspath(rootdir + '/example_2.grd')
    mesh.save(fname, overwrite=True)


if __name__ == "__main__":
    main()
