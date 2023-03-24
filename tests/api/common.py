import rasterio as rio
import numpy as np
from pyproj import CRS
from jigsawpy import jigsaw_msh_t


def create_rectangle_mesh(nx, ny, holes, x_extent=None, y_extent=None):
    """
    Note:
        x = x-index
        y = y-index

        node-index(node-id)

          25(26)             29(30)
      5     *---*---*---*---*
            | \ | \ | \ | \ |
      4     *---*---*---*---*
            | \ | \ | \ | \ |
      3     *---*---*---*---*
            | \ |   | \ | \ |
      2     *---*---*---*---*
            | \ | \ | \ | \ |
      1     *---*---*---*---*
            | \ | \ | \ | \ |
      0     *---*---*---*---*
          0(1)               4(5)

            0   1   2   3   4
    """

    if x_extent is None:
        x_range = range(nx)
    else:
        x_range = np.linspace(x_extent[0], x_extent[1], nx)

    if y_extent is None:
        y_range = range(ny)
    else:
        y_range = np.linspace(y_extent[0], y_extent[1], nx)

    X, Y = np.meshgrid(x_range, y_range)
    verts = np.array(list(zip(X.ravel(), Y.ravel())))
    cells = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            if (i + 1) + ((nx-1) * j) in holes:
                continue
            cells.append([j * nx + i, j * nx + (i + 1), (j + 1) * nx + i])
            cells.append([j * nx + (i + 1), (j + 1) * nx + (i + 1), (j + 1) * nx + i])
    # NOTE: Everywhere is above 0 (auto: land) unless modified later
    vals = np.ones((len(verts), 1)) * 10

    # TODO: Replace with "make_mesh" util function
    mesh_msht = jigsaw_msh_t()
    mesh_msht.ndims = +2
    mesh_msht.mshID = 'euclidean-mesh'
    mesh_msht.tria3 = np.array(
        [(c, 0) for c in cells], dtype=jigsaw_msh_t.TRIA3_t
    )
    mesh_msht.vert2 = np.array(
        [(v, 0) for v in verts], dtype=jigsaw_msh_t.VERT2_t
    )
    mesh_msht.value = np.array(
        vals, dtype=jigsaw_msh_t.REALS_t
    )
    return mesh_msht


# TODO: Move these helper functions to `utils`
def raster_from_numpy(
    filename,
    data,
    mgrid,
    crs=CRS.from_epsg(4326)
):
    x = mgrid[0][:, 0]
    y = mgrid[1][0, :]
    res_x = (x[-1] - x[0]) / data.shape[1]
    res_y = (y[-1] - y[0]) / data.shape[0]
    transform = rio.transform.Affine.translation(
        x[0], y[0]
    ) * rio.transform.Affine.scale(res_x, res_y)
    if not isinstance(crs, CRS):
        crs = CRS.from_user_input(crs)

    with rio.open(
        filename,
        'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)


def msht_from_numpy(
    coordinates,
    triangles,
    crs=CRS.from_epsg(4326)
):
    if not isinstance(crs, CRS):
        crs = CRS.from_user_input(crs)
    mesh = jigsaw_msh_t()
    mesh.mshID = 'euclidean-mesh'
    mesh.ndims = +2
    mesh.crs = crs
    mesh.vert2 = np.array(
        [(coord, 0) for coord in coordinates],
        dtype=jigsaw_msh_t.VERT2_t
        )
    mesh.tria3 = np.array(
        [(index, 0) for index in triangles],
        dtype=jigsaw_msh_t.TRIA3_t
        )

    return mesh


