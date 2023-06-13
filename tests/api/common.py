import rasterio as rio
import numpy as np
from pyproj import CRS
from jigsawpy import jigsaw_msh_t

import ocsmesh

def topo_2rast_1mesh(r1_path, r2_path, m1_path):

        rast_xy_1 = np.mgrid[-1:0.1:0.1, -0.7:0.1:0.1]
        rast_xy_2 = np.mgrid[0:1.1:0.1, -0.7:0.1:0.1]
        nx_z_1, ny_z_1 = rast_xy_1[0].shape
        rast_z_1 = np.ones_like(rast_xy_1[0]) * 100
        rast_z_1[:, ny_z_1*1//3:ny_z_1*2//3] = 10
        rast_z_1[:, ny_z_1*2//3:] = 0
        rast_z_1[nx_z_1*7//16:nx_z_1*11//16, :] = -10
        nx_z_2, ny_z_2 = rast_xy_2[0].shape
        rast_z_2 = np.ones_like(rast_xy_2[0]) * -100
        rast_z_2[:, :ny_z_2*1//3] = 0
        rast_z_2[:, ny_z_2*1//3:ny_z_2*2//3] = -10
        rast_z_2[nx_z_2*7//16:nx_z_2*11//16, :ny_z_2*2//3] = -10

        ocsmesh.utils.raster_from_numpy(
            r1_path, rast_z_1, rast_xy_1, 4326
        )
        ocsmesh.utils.raster_from_numpy(
            r2_path, rast_z_2, rast_xy_2, 4326
        )

        msh_t = ocsmesh.utils.create_rectangle_mesh(
            nx=17, ny=7, holes=[40, 41], x_extent=(-1, 1), y_extent=(0, 1))
        msh_t.crs = CRS.from_epsg(4326)
        msh_t.value[:] = 10

        mesh = ocsmesh.Mesh(msh_t)
        # NOTE: Assuming the interpolation works fine!
        mesh.interpolate(
            [ocsmesh.Raster(i) for i in [r1_path, r2_path]],
            method='nearest'
        )
        mesh.write(str(m1_path), format='grd', overwrite=False)
