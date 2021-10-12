#!/usr/bin/env python
"""
CLI interface for interpolating rasters into a mesh.
"""
import sys
import argparse
import pathlib
import shutil
import tempfile
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
import fiona
import requests
from tqdm import tqdm
from matplotlib.path import Path  # type: ignore[import]
from scipy.interpolate import RectBivariateSpline  # , griddata
from geoalchemy2.shape import from_shape
from shapely.geometry import box
from shapely.ops import transform
from pyproj import CRS, Transformer
from ocsmesh import Mesh, Raster, db


def _mesh_interpolate_worker(args):
    coords, raster, crs, chunk_size = args
    raster = Raster(raster)
    raster.warp(crs)
    results = []
    for window in raster.iter_windows(chunk_size=chunk_size, overlap=2):
        xi = raster.get_x(window)
        yi = raster.get_y(window)
        zi = raster.get_values(window=window)
        f = RectBivariateSpline(
            xi, np.flip(yi), np.fliplr(zi).T,
            bbox=[
                np.min(xi),
                np.max(xi),
                np.min(yi),
                np.max(yi)],
            kx=3,
            ky=3,
            s=0
        )
        idxs = np.where(
            np.logical_and(
                np.logical_and(
                    np.min(xi) < coords[:, 0],
                    np.max(xi) > coords[:, 0]),
                np.logical_and(
                    np.min(yi) < coords[:, 1],
                    np.max(yi) > coords[:, 1])))[0]

        values = f.ev(coords[idxs, 0], coords[idxs, 1])
        results.append((idxs, values))
    return results


class Interp:

    def __init__(self, args):
        self._args = args

    def main(self):

        if self._args.nprocs is None:
            self._main_serial()
        else:
            self._main_parallel()

        # write output to file
        self.mesh.write(
            self._args.output_mesh_path,
            overwrite=self._args.overwrite,
            fmt='gr3'
            )
        self.mesh.write(
            self._args.output_mesh_path + '.2dm',
            overwrite=self._args.overwrite,
            fmt='2dm'
            )

    def _main_serial(self):
        if self._args.use_anti_aliasing is True:
            self._initial_values = self.mesh.values.copy()

        for tile in (
                tqdm(self._rasters) if self._args.verbose else self._rasters):
            raster = Raster(tile)
            self.mesh.interpolate(raster)

            if self._args.use_anti_aliasing is True:
                self._resolve_aliasing(raster)

    def _main_parallel(self):
        # https://pythonspeed.com/articles/python-multiprocessing/
        # https://stackoverflow.com/questions/29864707/python-multiprocessing-for-matplotlib-griddata
        # mesh_values = self.mesh.values.copy()

        chunk_size = self._args.chunk_size
        tmpfile = tempfile.NamedTemporaryFile()
        fp = np.memmap(
            tmpfile.name,
            dtype='float32',
            mode='w+',
            shape=self.mesh.coords.shape
            )
        fp[:] = self.mesh.coords[:]
        with Pool(processes=self._args.nprocs) as pool:
            if self._args.verbose:
                res = list(tqdm(pool.imap(
                    _mesh_interpolate_worker,
                    [(fp, raster, self.mesh.crs, chunk_size)
                     for raster in self._rasters]
                    ), total=len(self._rasters)))
            else:
                res = pool.map(
                    _mesh_interpolate_worker,
                    [(fp, raster, self.mesh.crs, chunk_size)
                     for raster in self._rasters]
                    )
        del tmpfile
        output = np.full(self.mesh.values.shape, np.nan)
        for _ in res:
            for idxs, values in _:
                for i, idx in enumerate(idxs):
                    output[idx] = np.nanmean([output[idx], values[i]])
        idxs = np.where(np.isnan(output))
        output[idxs] = self.mesh.values[idxs]
        self.mesh._values = output

    @property
    @lru_cache(maxsize=None)
    def mesh(self):
        return Mesh.open(
            pathlib.Path(self._args.input_mesh_path),
            CRS.from_user_input(self._args.crs)
            )

    def _resolve_aliasing(self, raster):
        xy = self.mesh.get_xy(crs=raster.crs)
        rbbox = raster.bbox
        idxs = np.where(np.logical_and(
            np.logical_and(xy[:, 0] >= rbbox.xmin, xy[:, 0] <= rbbox.xmax),
            np.logical_and(xy[:, 1] >= rbbox.ymin, xy[:, 1] <= rbbox.ymax)
            ))

        zero_cross = np.sign(
            self.mesh._values[idxs]) - np.sign(self._initial_values[idxs])
        if np.any(zero_cross == 0.):
            _idxs = np.where(zero_cross != 0.)
            if self._args.anti_aliasing_method == "reuse":
                self.mesh._values[idxs][_idxs] = self._initial_values[idxs][_idxs]

            elif self._args.anti_aliasing_method == "fv":
                all_rings = []
                for rings in self.mesh.index_ring_collection.values():
                    all_rings.extend([rings['exterior'], *rings['interiors']])

                for i, idx in enumerate(idxs):
                    if zero_cross[i] == 0:
                        vertices = []
                        x0, y0 = xy[idx, 0], xy[idx, 1]

                        # include source point if it's a boundary node
                        for ring in all_rings:
                            if idx in ring:
                                vertices.append((x0, y0))
                                break

                        # find midpoint between neighbors
                        for neigh in self.mesh.node_neighbors[idx]:
                            vertices.append(
                                ((x0 + xy[neigh, 0])/2, (y0 + xy[neigh, 1])/2))

                        # compute centroids of neighbors
                        elements = self.mesh.triangulation.triangles[
                            np.where(
                                np.any(
                                    np.isin(
                                        self.mesh.triangulation.triangles,
                                        idx
                                        ),
                                    axis=1
                                    )
                                )
                            ]
                        cx = np.sum(self.mesh.x[elements], axis=1) / 3
                        cy = np.sum(self.mesh.y[elements], axis=1) / 3
                        for j in range(elements.shape[0]):
                            vertices.append((cx[j], cy[j]))
                        # finally make sure the vertices are ordered.
                        vertices = list(polygon_sort(vertices))
                        vertices.append(vertices[0])
                        path = Path(vertices, closed=True)
                        xin, yin = np.meshgrid(raster.x, raster.y)
                        xy_in = np.vstack([xin.flatten(), yin.flatten()]).T
                        ridxs = np.where(
                            np.logical_and(
                                np.logical_and(
                                    xy_in[:, 0] >= np.min(path.vertices[:, 0]),
                                    xy_in[:, 0] <= np.max(path.vertices[:, 0])
                                    ),
                                np.logical_and(
                                    xy_in[:, 1] >= np.min(path.vertices[:, 1]),
                                    xy_in[:, 1] <= np.max(path.vertices[:, 1]))
                                ))
                        ridxs_mask = path.contains_points(xy_in[ridxs])
                        rvalues = raster.values.flatten()
                        rvalues = rvalues[ridxs][np.where(ridxs_mask)]
                        if self._initial_values[idx] < 0:
                            self.mesh._values[i] = np.min(rvalues)
                        elif self._initial_values[idx] > 0:
                            self.mesh._values[i] = np.max(rvalues)
                        else:
                            raise Exception('unreachable')

            else:
                msg = 'duck-type error for anti aliasing method '
                msg += f'{self._args.anti_aliasing_method}'
                raise Exception(msg)

    def _expand_tile_index(self, path):
        validate_tile_index(path)
        raster_paths = []
        with fiona.open(path) as src:
            for feature in src:
                url = feature['properties']['URL']
                # Check if raster is in database
                res = self._session.query(db.TileIndexRasters).get(url)
                if res is None:
                    tmpfile = request_raster_from_url(url, self._args.verbose)
                    self._put_raster_in_cache(url, tmpfile)
                    res = self._session.query(db.TileIndexRasters).get(url)
                raster_paths.append(self._cache / 'data' / res.name)
        return raster_paths

    def _put_raster_in_cache(self, url, tmpfile):
        print('debug:_put_raster_in_cache()')
        datadir = self._cache / 'data'
        datadir.mkdir(exist_ok=True)
        target_path = datadir / url.split('/')[-1]
        if not target_path.is_file():
            shutil.copyfile(tmpfile.name, target_path)
    #         self._validate_raster_local(
    #             Raster(target_path), tmpraster.md5
    #     os.copyfile()
    #     tgtraster.save(target_path)
        raster = Raster(target_path)
        bbox = raster.bbox
        # pylint: disable=no-member
        geom = box(
                bbox.xmin,
                bbox.ymin,
                bbox.xmax,
                bbox.ymax
                )
        geom = transform_polygon(geom, raster.crs, 4326)
    #     md5 = raster.md5
        self._session.add(db.TileIndexRasters(
            geom=from_shape(
                geom,
                srid=4326
                ),
            url=url,
            name=target_path.name,
            md5=raster.md5))
        self._session.commit()
    #     return target_path, raster.m
    #     breakpoint()

    @property
    @lru_cache(maxsize=None)
    def _rasters(self):
        raster_paths = []
        for path in self._args.DEM:
            path = pathlib.Path(path)
            if not path.is_file():
                msg = f"No such file: {str(path)}"
                raise FileNotFoundError(msg)
            if check_if_uri_is_tile_index(path):
                raster_paths.extend(self._expand_tile_index(path))
            else:
                raster_paths.append(path)
        return raster_paths

    @property
    @lru_cache(maxsize=None)
    def _pyenv(self):
        return pathlib.Path("/".join(sys.executable.split('/')[:-2]))

    @property
    @lru_cache(maxsize=None)
    def _cache(self):
        cache = self._pyenv / '.cache'
        cache.mkdir(exist_ok=True)
        return cache

    @property
    @lru_cache(maxsize=None)
    def _session(self):
        return db.spatialite_session(self._cache / 'index.db', echo=False)


def polygon_sort(corners):
    # calculate centroid of the polygon
    n = len(corners)  # of corners
    cx = float(sum(x for x, y in corners)) / n
    cy = float(sum(y for x, y in corners)) / n
    # create a new list of corners which includes angles
    cornersWithAngles = []
    for x, y in corners:
        an = (np.arctan2(y - cy, x - cx) + 2.0 * np.pi) % (2.0 * np.pi)
        cornersWithAngles.append((x, y, an))
    # sort it using the angles
    cornersWithAngles.sort(key=lambda tup: tup[2])
    # return the sorted corners w/ angles removed
    return [(x, y) for x, y, _ in cornersWithAngles]


def check_if_uri_is_tile_index(uri):
    try:
        fiona.open(uri, 'r')
        return True
    except fiona.errors.DriverError:
        return False


def validate_tile_index(path):
    with fiona.open(path) as src:
        for feature in src:
            url = feature['properties'].get("URL")
            if url is None:
                raise AttributeError(
                    f"No 'URL' entry for feature with id {feature['id']} "
                    f"on file {path}")


def request_raster_from_url(url, verbose=False):
    """ returns :class:`tempfile.NamedTemporaryFile` object """
    # pylint: disable=R1732
    tmpfile = tempfile.NamedTemporaryFile()
    with open(tmpfile.name, 'wb') as f:
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(
                    chunk_size=max(int(total/1000), 1024*1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50*downloaded/total)
                if verbose:
                    sys.stdout.write(
                        '\r[{}{}]'.format('█' * done, '.' * (50-done)))
                    sys.stdout.flush()
    if verbose:
        sys.stdout.write('\n')
    return tmpfile


def transform_polygon(polygon, src_crs, dst_crs):
    src_crs = CRS.from_user_input(src_crs)
    dst_crs = CRS.from_user_input(dst_crs)
    if not dst_crs.equals(src_crs):
        transformer = Transformer.from_crs(
            src_crs, dst_crs, always_xy=True)
        polygon = transform(transformer.transform, polygon)
    return polygon


def validate_raster_local(raster, md5):
    if raster.md5 != md5:
        raise Exception(f'Checksum mismatch for path {str(raster.path)}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_mesh_path', help="Path to input mesh.")
    parser.add_argument('output_mesh_path', help="Path to output mesh.")
    parser.add_argument(
        'DEM', nargs="+",
        help="List of at least one input DEM(s). "
        "These will be interpolated in the same order as "
        "they are given. Recommended is to pass them in order of lowest "
        "priority to highest priority."
        )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Used in case the output_mesh_path exists and the user wants to "
        "allow overwrite.")
    parser.add_argument(
        "--nprocs", type=int, help="Total number of processors to use. This "
        "algorithm can make use of virtual cores, so this value is not "
        "restricted to the number of physical cores or the amount of DEM's "
        "to be interpolated.")
    parser.add_argument(
        "--chunk-size", type=int, help="Useful when passing large rasters that"
        " do not fit in memory, it will subdivide the rasters into boxes "
        " of maximum pixel size chunk-size x chunk-size. If your job runs out "
        "of memory, try using --chunk-size=3000")
    parser.add_argument(
        "--crs",
        help="Input mesh CRS. Output will have the same CRS as the input.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use-anti-aliasing", action="store_true")
    parser.add_argument(
        "--anti-aliasing-method",
        default="reuse",
        choices=['reuse', 'fv']
        )
    return parser.parse_args()


def main():
    Interp(parse_args()).main()


if __name__ == '__main__':
    main()
