#! /usr/bin/env python
"""
This program parses a json configuration file and generates a model-ready mesh.
"""
import argparse
from functools import lru_cache
import pathlib
# import json
import logging
import os
import warnings
from copy import deepcopy
from collections import namedtuple
import fiona
# from sqlalchemy.event import listen
# from sqlalchemy.orm import sessionmaker
# from sqlalchemy.sql import select, func
# from sqlalchemy import create_engine
from scipy.spatial import cKDTree
import numpy as np
from pyproj import CRS  # , Transformer
from shapely import ops
from shapely.geometry import (
    LineString,
    MultiLineString,
    MultiPolygon,
    box
    )
from geomesh import (
    cmd,
    logger,
    # db,
    Geom,
    Raster,
    RasterCollection,
    JigsawDriver,
    )


class Geomesh:
    """
    Mixin class for CLI.
    """

    def __init__(self, args):
        self._args = args
        self.generate_mesh()
        self.run_postprocessing()
        self.save_outputs()

    def generate_mesh(self):
        driver = JigsawDriver(
            self.geom,
            self.hfun,
            self.hfun.hfun,
            )
        driver.optm_qlim = self._conf.driver.optm_qlim
        driver.verbosity = self._conf.verbosity
        self.mesh = driver.run()

    def run_postprocessing(self):
        self._interpolate_mesh()
        self._generate_boundaries()
        self._fix_levee_elevations()

    def save_outputs(self):
        self._save_mesh()
        self._save_boundaries()

    def _interpolate_mesh(self):
        for raster in self._interp_rasters:  # noqa: .iter(priority=self.raster_conf.get("priority", "auto")): 
            self.mesh.interpolate(raster)

    def _generate_boundaries(self):
        if self._conf_boundaries is not None:
            if self._conf_boundaries is True:
                self.mesh.generate_boundaries()
            else:
                self.mesh.generate_boundaries(**self._conf_boundaries)

    def _fix_levee_elevations(self):
        levees = self._conf_levees
        if levees is None:
            return
        for id in levees:
            features = self._get_feature_collection_by_id(
                id, self.mesh.crs)
            # find intersection between levees and element edges
            multipoint = MultiLineString(features).intersection(
                MultiLineString(
                    [LineString(
                        [(self.mesh.x[e0], self.mesh.y[e0]),
                         (self.mesh.x[e1], self.mesh.y[e1])])
                        for e0, e1 in self.mesh.triangulation.edges]))
            coords = np.squeeze(
                np.array([point.coords for point in multipoint]))
            tree = cKDTree(self.mesh.xy)
            dd, ii = tree.query(coords[:, :2], n_jobs=-1)
            values = self.mesh.values.copy()
            for i, idx in enumerate(ii):
                values[idx] = np.max([values[idx], coords[i, 2]])
            self.mesh._values = values

    def _save_mesh(self):
        outputs = self._conf_outputs.get("mesh", [])
        for output in outputs:
            path = pathlib.Path(os.path.expandvars(output['name']))
            fmt = output['format']
            crs = output.get("crs", None)
            if crs is not None:
                mesh = deepcopy(self.mesh)
                mesh.transform_to(crs)
            else:
                if fmt == '2dm':
                    mesh = deepcopy(self.mesh)
                    mesh.transform_to("EPSG:4326")
                else:
                    mesh = self.mesh
            mesh.write(
                path,
                overwrite=self._args.overwrite,
                fmt=fmt
                )

    def _save_boundaries(self):
        outputs = self._conf_outputs.get("boundaries", False)
        assert isinstance(outputs, bool)
        if outputs:
            self.mesh.write_boundaries('boundaries', overwrite=True)

    def _load_rasters(self, config):
        self._logger.debug('_get_config_rasters')
        rasters = config.get('rasters')

        if rasters is None:
            return

        msg = "config.rasters must contain a dictionary. The keys must be "
        msg += "an indentifier and each indentifier must contain a dictionary "
        msg += "with at least 'uri' entry."
        assert isinstance(rasters, dict), msg

        for id, data in rasters.copy().items():
            if id.startswith("_"):
                continue
            self._logger.debug(f'_get_config_rasters:{id}')
            msg = "raster entry with id {id} does not contain an uri entry."
            assert 'uri' in data.keys(), msg
            config['rasters'][id].update(
                {"_obj": self._get_config_raster_data(data)})
        return rasters

    def _get_config_geom(self, config):
        # self._logger.debug('_get_config_geom')
        # geom = config.get('geom')
        # if geom is None:
        #     msg = "geom entry must be defined."
        #     raise IOError(msg)
        # msg = "geom must contain at least one of 'rasters' or 'features' keys."
        # assert "rasters" in geom or "features" in geom, msg
        # geom_collection = []
        # if 'rasters' in geom:
        #     rasters = config.get('rasters')
        #     for id, opts in geom['rasters'].items():
        #         if id.startswith("_"):
        #             continue
        #         self._logger.debug(f'_get_config_geom:{id}')
        #         msg = f"config.geom.raster with id '{id}' not listed in "
        #         msg += f"config.rasters.keys(): {rasters}"
        #         assert id in rasters, msg
        #         # raise NotImplementedError("Must do checksum check.")
        #         geom_collection.append(rasters[id]["_obj"].get_geom(
        #             zmin=opts.get("zmin"),
        #             zmax=opts.get("zmax"),
        #             join_method=opts.get("join_method"),
        #             driver=opts.get("driver"),
        #             overlap=opts.get("overlap"),
        #             ))

        if "features" in geom:
            raise NotImplementedError("features")

        mpc = []
        for geom in geom_collection:
            mpc.append(geom.multipolygon)
        mp = ops.unary_union(mpc)

        geom = Geom(mp, geom.crs)
        geom.make_plot(show=True)
        raise NotImplementedError

        config["geom"].update({"_obj": geom})
        return namedtuple(
            "config_geom",
            [
                "rasters",
                # "features",
                "geom"
            ]
            )(
            rasters=rasters,
            geom=config["geom"]["_obj"]
            )

    def _get_config_hfun(self, config):
        self._logger.debug('_get_config_hfun')
        hfun = config.get('hfun')
        if hfun is None:
            msg = "hfun entry must be defined."
            raise IOError(msg)
        # msg = "geom must contain at least one of 'rasters' or 'features' keys."
        # assert "rasters" in geom or "features" in geom, msg
        hfun_collection = []
        if 'rasters' in hfun:
            hfun_rasters = hfun.get('rasters')
            msg = "hfun.rasters must be  a dictionary."
            assert isinstance(hfun_rasters, dict), msg
            for id, opts in hfun_rasters.items():
                self._logger.debug(f'_get_config_hfun:{id}')
                hfun = config['rasters'][id]["_obj"].get_hfun(
                    # geom=config['geom']["_obj"],
                    # hmin=None,
                    # hmax=None,
                    )
                self._update_hfun_raster_criteria(hfun, opts)
                hfun.contourf(show=True)
                hfun_collection.append(hfun)
                exit()

                # # process contours
                # contours = opts.get("contours", {})
                # if isinstance(contours, dict):
                #     contours = [contours]
                # assert isinstance(contours, list)
                # for contour in contours:
                #     level = float(contour['level'])
                #     exp_rate = float(contour['expansion_rate'])
                # # process features
                # features = opts.get("features", {})
                # if isinstance(features, dict):
                #     features = [features]
                # assert isinstance(features, list)
                # for feature in features:
                #     raise NotImplementedError('need to add features')

    def _get_config_outputs(self, config):
        outputs = config.get('outputs', {})
        if len(outputs) == 0:
            warnings.warn('No outputs set in configuration file.')
        msg = 'outputs key must contain a dictionary with  "mesh" and '
        msg += '"boundaries" entries.'
        assert isinstance(outputs, dict), msg
        return namedtuple(
            "config_outputs",
            [
                "mesh",
                "boundaries"
            ]
            )(
            mesh=self._get_config_outputs_mesh(outputs),
            boundaries=self._get_config_outputs_boundaries(outputs)
            )

    def _get_config_outputs_mesh(self, outputs):
        mesh = outputs.get('mesh')
        if mesh is None:
            return

        # check input type
        msg = "'outputs' entry must contain a dictionary or list of "
        msg += 'dictionaries.'
        assert isinstance(mesh, (list, dict)), msg
        # cast dict input to list of dict
        mesh = [mesh] if isinstance(mesh, dict) else mesh
        # Check entries provided in json file
        prefix = "Dictionary members of 'outputs.mesh[]' must contain at "
        prefix += "least one"
        suffix = "key. This key must correspond to the desired "
        output_collection = []
        for output in mesh:

            # check 'name'
            msg = f"{prefix} 'name' {suffix}"
            msg += "output file path. Paths can be full or relative and "
            msg += 'environment variables can be used.'
            assert 'name' in output.keys(), msg
            name = pathlib.Path(
                os.path.expandvars(str(output.get('name')))).resolve()

            # check for overwrite
            if name.is_file() and self._args.overwrite is False:
                msg = f"File {name} exists and overwrite is set to False "
                msg += "(default)."
                raise Exception(msg)

            # check 'crs'
            msg = f"{prefix} 'crs' {suffix}"
            msg += "output coordinate reference system."
            assert 'crs' in output.keys(), msg
            crs = CRS.from_user_input(str(output['crs']))

            # check 'format'
            msg = f"{prefix} 'format' {suffix}"
            msg += 'file format. This can be 2dm or grd.'
            assert 'format' in output.keys(), msg

            # check format types
            format = str(output['format'])
            msg = "'format' must be one of 2dm or grd."
            assert format.lower() in ['2dm', 'grd'], msg

            # collect parameters
            output_collection.append(namedtuple(
                "mesh_output_configuration",
                ['name', 'crs', 'format']
                )(
                name=name,
                crs=crs,
                format=format
                ))

        return output_collection

    def _get_config_outputs_boundaries(self, outputs):
        boundaries = outputs.get('boundaries')
        if boundaries is None:
            return
        raise NotImplementedError

    # def _get_config_raster_data(self, data):

        # We need to figure out if this URI is a URL or a local path.
        # if 'http' in data['uri'] or 'ftp' in data['uri']:
        #     msg = "URI provided is an internet address."
        #     raise NotImplementedError(msg)

        # Try to resolve path relative to config file.
        data.update({
            'uri': (
                pathlib.Path(self._args.config_file).parent /
                pathlib.Path(os.path.expandvars(data['uri']))
                )
            })

        if not data['uri'].is_file():
            msg = f"No file with path: {data['uri']}"
            raise FileNotFoundError(msg)

        try:
            return self._get_config_rasters_tile_index(data)
        except Exception:  # DriverError exception
            pass

        return self._get_config_raster_local_file(data)

        # msg = "\nThe URI object is not readable by rasterio nor fiona."
        # # msg += f"\nfiona returned {ferr} "
        # msg += f"\nrasterio returned {rerr}"
        # raise Exception(msg)

    def _get_config_rasters_tile_index(self, data):
        with fiona.open(data['uri'], 'r') as f:
            pass
        return RasterCollection()

    def _get_config_raster_local_file(self, data):
        self._logger.debug(f'_get_config_raster_local_file:{data["uri"]}')
        kwargs = {
            "path": data['uri'],
            "src_crs": data.get("src_crs"),
            "chunk_size": data.get("chunk_size"),
            "overlap": data.get("overlap")
        }
        raster = Raster(**kwargs)
        self._update_raster_opts(
            raster,
            # {x: data[x] for x in data if x != 'uri'}
            data
            )
        data.update({"_obj": raster})
        return data["_obj"]

    # def _get_geom_raster_multipolygon(self):
    #     polygon_collection = []
    #     for id, data in self._config.geom.rasters.values():
    #         for polygon in data['_obj'].get_multipolygon(
    #             zmin=data.get(id).get("zmin"),
    #             zmax=data.get(id).get("zmax")
    #                 ):
    #             polygon_collection.append(polygon)
    #     return MultiPolygon(polygon_collection).buffer(0)

    # def _resolve_tile_index(self, id):
    #     with fiona.open(self._conf_rasters[id], 'r') as src:
    #         uris = []
    #         for feature in src:
    #             url = feature.get('properties', {}).get('URL', None)
    #             if url is None:
    #                 msg = f'No URL given for feature with id {id}'
    #                 raise Exception(msg)
    #             fname = self._cache / url.split('/')[-1]
    #             if not fname.is_file() or fname.stat().st_size == 0:
    #                 open(fname, 'wb').write(
    #                     requests.get(
    #                         url, allow_redirects=True).content)
    #             uris.append(fname)
    #     return uris

    def _update_raster_opts(self, raster, opts):
        for key, opt in opts.items():
            if key == "resample":
                raster.resample(opt['scaling_factor'])
            if key == 'warp':
                raster.warp(opt)
            if key == 'fill_nodata':
                raster.fill_nodata()
            if key == 'clip':
                if isinstance(opt, dict):
                    raster.clip(
                        MultiPolygon(
                            [box(
                                opt['xmin'],
                                opt['ymin'],
                                opt['xmax'],
                                opt['ymax'])]
                            )
                        )
                else:
                    msg = "clip by geometry"
                    raise NotImplementedError(msg)

    def _update_hfun_raster_criteria(self, hfun, config):

        for id, criteria in config['hfun']['rasters'].items():
            print(id, criteria)
            raise NotImplementedError

        # contours = opts.get('contours', [])
        # if isinstance(contours, dict):
        #     contours = [contours]
        # for opt in contours:
        #     hfun.add_contour(
        #         opt['level'],
        #         opt['target_size'],
        #         opt['expansion_rate'],
        #         hmin=opt['hmin'],
        #         hmax=opt['hmax'],
        #         n_jobs=opt['n_jobs']
        #         )

    def _load_geom(self):
        print(self._config)
        # self._load_rasters(self._config)
        # self._get_config_geom(self._config)
        # query = self._db.query(db.Geom).filter_by(
        #     uri=self._get_current_config_uri(),
        #     md5=self._get_current_config_md5()
        #     )
        raise NotImplementedError

    @property
    def geom(self):
        return self._load_geom()

    @property
    def hfun(self):
        return self._load_hfun()

    @property
    @lru_cache(maxsize=None)
    def _config_path(self):
        return pathlib.Path(self._args.config_file)

    @property
    @lru_cache(maxsize=None)
    def _config(self):
        return cmd.config.read_config(self._config_path)

    @property
    def _args(self):
        return self.__args

    @_args.setter
    def _args(self, args):
        args.config_file = pathlib.Path(args.config_file)
        self.__args = args

    @property
    @lru_cache(maxsize=None)
    def _logger(self):
        return logging.getLogger(__name__ + '.' + self.__class__.__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file",
        help="Path to configuration file.",
        # nargs='?'
        )
    parser.add_argument("--log-level", choices=["info", "debug", "warning"])
    # parser.add_argument("--verbosity", type=int, default=0)
    # parser.add_argument("--cache-dir")
    parser.add_argument("--clear-cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.init(args.log_level)
    Geomesh(args)


if __name__ == '__main__':
    main()




























































    # def _get_feature_collection_by_id(self, id, dst_crs):
    #     uri = self._conf_features[id].get('uri', None)
    #     if uri is None:
    #         msg = f'Must specify at least one "uri" for feature with id {id}.'
    #         raise Exception(msg)
    #     feature_collection = []
    #     shp = fiona.open(uri)
    #     shp_crs = CRS.from_user_input(shp.crs)
    #     dst_crs = CRS.from_user_input(dst_crs)
    #     if shp_crs.srs != dst_crs.srs:
    #         transformer = Transformer.from_crs(
    #             shp.crs, dst_crs, always_xy=True)
    #         _tflag = True
    #     else:
    #         _tflag = False
    #     for feature in shp:
    #         geometry = shape(feature['geometry'])
    #         if _tflag:
    #             geometry = transform(transformer.transform, geometry)
    #         feature_collection.append(geometry)
    #     shp.close()
    #     return feature_collection

    # def _resolve_raster_data(self, id, opts):
    #     pass

    # @property
    # @lru_cache(maxsize=None)
    # def _conf_rasters(self):
    #     rasters = self._conf.get("rasters", {})
    #     msg = '"rasters" entry must be a dictionary, not '
    #     msg += f'{type(rasters)}. The dictionary must consist of a unique key '
    #     msg += 'for each raster, and each key contains a dictionary with '
    #     msg += 'raster options. At least the "uri" entry is required.'
    #     assert isinstance(rasters, dict), msg
    #     for id, data in rasters.items():
    #         msg = f'No "uri" entry for raster with id: {id}'
    #         assert 'uri' in data, msg
    #     rasters = {}
    #     for id, opts in self._conf["rasters"].items():
    #         rasters.update({id: self._resolve_raster_data(id, opts)})
    #     return rasters

    # @property
    # @lru_cache(maxsize=None)
    # def _conf_features(self):
    #     features = self._conf.get("features", {})
    #     msg = '"features" entry must be a dictionary, not '
    #     msg += f'{type(features)}.'
    #     assert isinstance(features, dict), msg
    #     for id, data in features.items():
    #         msg = f'No "uri" entry for raster with id: {id}'
    #         assert 'uri' in data, msg
    #     return features

    # @property
    # @lru_cache(maxsize=None)
    # def _conf_geom(self):
    #     geom = self._conf.get('geom', None)
    #     if geom is None:
    #         msg = 'No "geom" data in configuration file.'
    #         raise Exception(msg)
    #     msg = '"geom" entry must be a dictionary.'
    #     assert isinstance(geom, dict), msg
    #     return geom

    # @property
    # @lru_cache(maxsize=None)
    # def _conf_geom_rasters(self):
    #     # certify rasters
    #     rasters = self._conf_geom.get("rasters", None)
    #     if rasters is not None:
    #         msg = '"rasters" must be a list of rasters id\'s.'
    #         assert isinstance(rasters, list), msg
    #         msg = 'raster specified to geom is not in not in "rasters" key'
    #         for key in rasters:
    #             assert key in self._conf_rasters.keys(), msg
    #     else:
    #         rasters = []
    #     raster_collection = RasterCollection()
    #     for id in rasters:
    #         for fname in self._resolve_raster_id(id):
    #             raster_collection.append(fname)
    #     return raster_collection

    # @property
    # @lru_cache(maxsize=None)
    # def _conf_geom_features(self):
    #     # certify features
    #     features = self._conf_geom.get("features", None)
    #     if features is not None:
    #         if isinstance(features, dict):
    #             if 'clip' in features:
    #                 msg = '"clip" must be a list of features id\'s.'
    #                 assert isinstance(features['clip'], list), msg
    #                 msg = 'feature specified to geom is not in not in '
    #                 msg += '"features" key'
    #                 for key in features['clip']:
    #                     assert key in self._conf_features.keys(), msg
    #             if 'buffer' in features:
    #                 msg = '"buffer" must be a list of features id\'s.'
    #                 assert isinstance(features['buffer'], list), msg
    #                 msg = 'feature specified to geom is not in not in '
    #                 msg += '"features" key'
    #                 for key in features['buffer']:
    #                     assert key in self._conf_features.keys(), msg
    #         if 'clip' not in features and 'buffer' not in features:
    #             raise Exception("need to specified clip, buffer or both")
    #     else:
    #         features = []
    #     return features

    # @property
    # def _geom_crs(self):
    #     return self._conf_geom.get("crs", "EPSG:3395")

    # @property
    # @lru_cache(maxsize=None)
    # def _conf_geom_clip(self):
    #     # certify features
    #     features = self._conf_geom.get("clip", None)
    #     if features is None:
    #         return
    #     raise NotImplementedError('_conf_geom_clip')
    #     clip = features.get("clip", None)
    #     if clip is not None:
    #         msg = '"clip" must be a list of features id\'s.'
    #         assert isinstance(clip, list), msg
    #     msg = 'feature specified to geom is not in not in "features" key'
    #     fc = []
    #     for id in features.get("clip", []):
    #         assert id in self._conf_features.keys(), msg
    #         feats = self._get_feature_collection_by_id(id, self._geom_crs)
    #         for feat in feats:
    #             msg = '"clip" must be Polygon or MultiPolygon'
    #             assert isinstance(feat, (Polygon, MultiPolygon)), msg
    #             if isinstance(feat, Polygon):
    #                 fc.append(feat)
    #             else:
    #                 for polygon in feat:
    #                     fc.append(feat)
    #     return MultiPolygon(fc).buffer(0)

    # @property
    # @lru_cache(maxsize=None)
    # def _conf_geom_buffer(self):
    #     # certify features
    #     features = self._conf_geom.get("features", None)
    #     if features is None:
    #         return
    #     raise NotImplementedError()
    #     buffer = features.get("buffer", None)
    #     if buffer is not None:
    #         msg = '"buffer" must be a list of features id\'s.'
    #         assert isinstance(buffer, list), msg
    #     msg = 'feature specified to geom is not in not in "features" key'
    #     for key in features.get("buffer", {}):
    #         assert key in self._conf_features.keys(), msg
    #     return buffer

    # @property
    # @lru_cache(maxsize=None)
    # def _conf_hfun(self):
    #     hfun = self._conf.get('hfun', None)
    #     if hfun is not None:
    #         msg = '"hfun" entry must be a constant or a dictionary.'
    #         assert isinstance(hfun, (float, int, dict)), msg
    #     # if isinstance(hfun, dict):
    #     #     msg = '"hfun" must contain a "raster" dictionary of raster id '
    #     #     msg += 'keys.'
    #     #     assert "rasters" in hfun, msg
    #     return hfun

    # @property
    # @lru_cache(maxsize=None)
    # def _conf_levees(self):
    #     features = self._conf.get("levees", None)
    #     if features is not None:
    #         msg = '"levees" entry must be a dictionary, not '
    #         msg += f'{type(features)}.'
    #         assert isinstance(features, dict), msg
    #         msg = 'feature specified as levee is not in not in "features" keys'
    #         for id in features:
    #             assert id in self._conf_features.keys(), msg
    #     return features

    # @property
    # @lru_cache(maxsize=None)
    # def _conf_driver(self):
    #     pass
    #     # outputs = self._conf.get("outputs", None)
    #     # if outputs is None:
    #     #     warnings.warn('No outputs set in configuration file.')
    #     # else:
    #     #     msg = '"outputs" entry must be a dictionary.'
    #     #     assert isinstance(outputs, dict), msg
    #     # return outputs
    #     return {}

    # @property
    # @lru_cache(maxsize=None)
    # def _conf_hfun_rasters(self):
    #     rasters = self._conf_hfun.get("rasters", None)
    #     if rasters is None:
    #         msg = '""hfun" must contain a "rasters" entry and it must be '
    #         msg += 'a dictionary of raster id\'s.'
    #         assert isinstance(rasters, dict), msg
    #         msg = 'raster specified to geom is not in not in "rasters" key'
    #         for key in self._conf_hfun["rasters"]:
    #             assert key in self._conf_rasters.keys(), msg
    #     raster_collection = geomesh.raster.RasterCollection()
    #     for id in rasters:
    #         for fname in self._resolve_raster_id(id):
    #             raster_collection.append(fname)
    #     return raster_collection

    # @property
    # @lru_cache(maxsize=None)
    # def _conf_hfun_features(self):
    #     features = self._conf_hfun.get("features", {})
    #     # certify Mapping type
    #     msg = '"features" must be a dictionray of features id\'s.'
    #     assert isinstance(features, dict), msg
    #     # certify contents
    #     msg = 'feature specified to hfun is not in not in "features" key'
    #     for id in features:
    #         assert id in self._conf_features.keys(), msg
    #     return features

    # @property
    # @lru_cache(maxsize=None)
    # def _interp_rasters(self):
    #     raster_collection = geomesh.raster.RasterCollection()
    #     for id in self._conf_rasters:
    #         for fname in self._resolve_raster_id(id):
    #             raster_collection.append(fname)
    #     return raster_collection


# -------- interface
# def print_conf_file_template():
#     """
#     Prints extra help for creating a configuration file.
#     """
#     config = dict()
#     print(config)
#     raise NotImplementedError


        # hfun is optional. Return if no inputs are given.
    #     if self._conf_hfun is None:
    #         return

    #     # input for SizeFunction is currently restricted to self.geom
    #     # if geom has a raster collection use the geom.
    #     if len(self.geom.raster_collection) > 0:
    #         hfun = self.geom
    #     else:
    #         msg = 'geom has no rasters, therefore SizeFunction cannot be '
    #         msg += 'instantiated.'
    #         raise NotImplementedError(msg)
    #         hfun = self._hfun_raster_collection

    #     hfun = Hfun(
    #         hfun,
    #         hmin=self._conf_hfun.get("hmin", None),
    #         hmax=self._conf_hfun.get("hmax", None),
    #         crs=self._conf_hfun.get("crs", None),
    #         verbosity=self._args.verbosity
    #     )
    # # ----------- Add hfun requests
    #     # contour level
    #     for id, opts in self._conf_hfun['rasters'].items():
    #         # add contour requests
    #         contours = opts.get("contours", None)
    #         if contours is not None:
    #             msg = '"contours" entry must be a list.'
    #             assert isinstance(contours, list), msg
    #             for data in contours:
    #                 hfun.add_contour(
    #                     level=data['level'],
    #                     expansion_rate=data.get("expansion_rate", None),
    #                     target_size=data.get("target_size", None)
    #                     )

    #     # subtidal flow limiter
    #     sfl = self._conf_hfun.get("subtidal_flow_limiter", False)
    #     if sfl:
    #         if isinstance(sfl, bool):
    #             hfun.add_subtidal_flow_limiter()
    #         elif isinstance(sfl, dict):
    #             hfun.add_subtidal_flow_limiter(**sfl)

    #     # constant floodplain size
    #     floodplain = self._conf_hfun.get("floodplain_size", False)
    #     if floodplain is not False:
    #         hfun.floodplain_size(floodplain)

    #     # custom features
    #     for id, opts in self._conf_hfun_features.items():
    #         features = self._get_feature_collection_by_id(id, hfun._dst_crs)
    #         msg = '"features" entry must contain a "target_size".'
    #         assert "target_size" in opts, msg
    #         # It seems that we can handle MultiLineString more
    #         # efficiently than LineString and it seems to be natively
    #         # supported by add_feature method. There seemsno obvious reason to
    #         # innput each feature individually, but was consider
    #         hfun.add_feature(MultiLineString(features), **opts)

    #     # add gaussian filter requests
    #     for kwargs in self._conf_hfun.get("gaussian_filter", {}).values():
    #         hfun.add_gaussian_filter(**kwargs)

    #     return hfun