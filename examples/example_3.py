#!/usr/bin/env python
"""
DEM dataset used in this examples can be downloaded from:
https://www.ngdc.noaa.gov/mgg/inundation/sandy/sandy_geoc.html
This example requires about 40Gb of virtual memory to run.
"""
from pathlib import Path
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

    # ------- global options
    hmax = 100.
    zmin = -3000.
    zmax = 15.

    # ------- init DatasetCollection
    dsc = DatasetCollection()
    # i = 0
    for file in Path(str(Path.home()) + '/postSandyDEM').glob('**/*.tif'):
        # if 'zip19' in str(file):
        dsc.add_dataset(file)
        # if i == 3:
        #     break
        # i += 1

    # ------- generate PSLG
    pslg = PlanarStraightLineGraph(dsc, zmin, zmax)
    # pslg.SpatialReference = 4326
    # pslg.make_plot(show=True)

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
    import time
    start = time.time()
    mesh = jigsaw.run()
    print('took: {}'.format(time.time() - start))

    # ------- interpolate bathymtery to output mesh
    mesh.interpolate(dsc, fix_invalid=True)
    fig = plt.figure()
    axes = fig.add_subplot(111)
    mesh.make_plot(axes=axes)
    axes.triplot(mesh.mpl_tri, linewidth=0.05, color='k')
    plt.show()


if __name__ == "__main__":
    main()
