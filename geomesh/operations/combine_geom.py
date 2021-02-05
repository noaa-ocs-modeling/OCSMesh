import os
import pathlib
import logging
import warnings
from typing import Union, Sequence, Literal

import fiona
from shapely import ops
from shapely.geometry import mapping, box

from geomesh import Raster, Geom
from geomesh.mesh.mesh import Mesh

_logger = logging.getLogger(__name__)


def run(dem_files: Sequence[Union[str, os.PathLike]],
        out_file: Union[str, os.PathLike],
        out_format: Literal["shapefile"] = "shapefile",
        mesh_file: Union[str, os.PathLike, None] = None,
        zmin: Union[float, None] = None,
        zmax: Union[float, None] = None,
        chunk_size: Union[int, None] = None,
        overlap: Union[int, None] = None):

    base_mult_poly = None
    poly_coll = list()
    if mesh_file and pathlib.Path(mesh_file).is_file():
        _logger.info("Creating mesh object from file...")
        base_mesh = Mesh.open(mesh_file)
        _logger.info("Done")

        _logger.info("Getting mesh hull polygons...")
        base_mult_poly = base_mesh.hull.multipolygon()
        _logger.info("Done")

    z_info = dict()
    if zmin is not None:
        z_info['zmin'] = zmin
    if zmax is not None:
        z_info['zmax'] = zmax

    _logger.info("Getting DEM info")
    for dem_path in dem_files:

        if not pathlib.Path(dem_path).is_file():
            warnings.warn(f"File {dem_path} not found!")
            _logger.debug(f"File {dem_path} not found!")
            continue

        # Calculate Polygon
        _logger.info("Loading raster from file...")
        rast = Raster(
                dem_path,
                chunk_size=chunk_size,
                overlap=overlap)
        _logger.info("Creating geom from raster...")
        geom = Geom(rast)

        _logger.info("Getting polygons from geom...")
        geom_mult_poly = geom.get_multipolygon(**z_info)
        poly_coll.append(geom_mult_poly)

        if base_mult_poly is not None:
            _logger.info("Subtract DEM bounds from base mesh polygons...")
            rast_box = box(*rast.src.bounds)
            base_mult_poly = base_mult_poly.difference(rast_box)

        # TODO: Store an on-disk raster so that combined raster
        # can be exported(?)

    _logger.info("Generating final boundary polygon...")
    if base_mult_poly is not None:
        poly_coll.append(base_mult_poly)
    fin_mult_poly = ops.unary_union(poly_coll)
    _logger.info("Done")

    if out_format == "shapefile":
        # TODO: Consider projection(?)
        schema = {'geometry': 'MultiPolygon',
                  'properties': {'id': 'int'}, }
        out_dir = pathlib.Path(out_file).parent
        out_dir.mkdir(exist_ok=True, parents=True)
        with fiona.open(
                out_file, 'w', 'ESRI Shapefile',  schema, 'EPSG:4326') as tgt:

            tgt.write({'geometry': mapping(fin_mult_poly),
                       'properties': {'id': 1}})
    else:
        raise NotImplementedError(f"Output type {out_format} is not supported")
