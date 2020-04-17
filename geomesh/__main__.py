#! /usr/bin/env python
import argparse
from functools import lru_cache
from copy import deepcopy
import fiona
import rasterio
import pathlib
import json
import tempfile
import logging
import requests
import os
import warnings
from shapely.ops import transform
from scipy.spatial import cKDTree
import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import (
    shape,
    LineString,
    MultiLineString,
    Polygon,
    MultiPolygon
    )
import geomesh


class Geomesh:

    def __init__(self, args):
        self.args = args

    def run(self):
        self.check_overwrite()
        self.generate_mesh()
        self.interpolate_mesh()
        self.generate_boundaries()
        self.fix_levee_elevations()
        self.save_mesh()

    def check_overwrite(self):
        outputs = self._outputs_config.get("mesh", [])
        for output in outputs:
            path = pathlib.Path(os.path.expandvars(output['name']))
            if path.is_file() and self.args.overwrite is False:
                msg = "File exists and overwrite is False."
                raise Exception(msg)

    def generate_mesh(self):
        driver = geomesh.driver.JigsawDriver(
            self.geom,
            self.hfun,
            self.hfun.hfun,
            )
        driver.optm_qlim = self._driver_config.get("optm_qlim", 0.975)
        driver.verbosity = self.args.verbosity
        self.mesh = driver.run()

    def interpolate_mesh(self):
        for raster in self._interp_rasters:  # noqa: .iter(priority=self.raster_conf.get("priority", "auto")): 
            self.mesh.interpolate(raster)

    def generate_boundaries(self):
        if self._boundaries_config is not None:
            if self._boundaries_config is True:
                self.mesh.generate_boundaries()
            else:
                self.mesh.generate_boundaries(**self.boundaries_config)

    def save_mesh(self):
        outputs = self._outputs_config.get("mesh", [])
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
                overwrite=self.args.overwrite,
                fmt=fmt
                )

    def fix_levee_elevations(self):
        levees = self._levees_config
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

    @property
    @lru_cache
    def geom(self):
        """
        This property calls the geomesh Geom generator based on the json inputs
        """
        geom_rasters = self._geom_raster_collection
        # geom_features = self._
        self._geom_config.get("features", None)
        if len(geom_rasters) > 0:
            geom = geomesh.geom.Geom(
                geom_rasters,
                zmin=self._geom_config.get('zmin', None),
                zmax=self._geom_config.get('zmax', None),
                radii=self._geom_config.get('radii', None),
                clip=self._geom_clip,
                buffer=self._geom_buffer,
                )

        else:
            raise NotImplementedError('Must instantiate geom using a polygon')

        return geom

    @property
    @lru_cache
    def hfun(self):
        """
        hfun is an optional parameter, so this property can return None,
        otherwise it returns a geomesh.SizeFunction instance.

        The geomesh.SizeFunction instance is initialized with the parameters
        in the config.json section called "hfun". See the json template for
        more details.
        """

        # hfun is optional. Return if no inputs are given.
        if self._hfun_config is None:
            return

        # input for SizeFunction is currently restricted to self.geom
        # if geom has a raster collection use the geom.
        if len(self.geom.raster_collection) > 0:
            hfun = self.geom
        else:
            msg = 'geom has no rasters, therefore SizeFunction cannot be '
            msg += 'instantiated.'
            raise NotImplementedError(msg)
            hfun = self._hfun_raster_collection

        hfun = geomesh.hfun.SizeFunction(
            hfun,
            hmin=self._hfun_config.get("hmin", None),
            hmax=self._hfun_config.get("hmax", None),
            crs=self._hfun_config.get("crs", None),
            verbosity=self.args.verbosity
        )
    # ----------- Add hfun requests
        # contour level
        for id, opts in self._hfun_config['rasters'].items():
            # add contour requests
            contours = opts.get("contours", None)
            if contours is not None:
                msg = '"contours" entry must be a list.'
                assert isinstance(contours, list), msg
                for data in contours:
                    hfun.add_contour(
                        level=data['level'],
                        expansion_rate=data.get("expansion_rate", None),
                        target_size=data.get("target_size", None)
                        )

        # subtidal flow limiter
        sfl = self._hfun_config.get("subtidal_flow_limiter", False)
        if sfl:
            if isinstance(sfl, bool):
                hfun.add_subtidal_flow_limiter()
            elif isinstance(sfl, dict):
                hfun.add_subtidal_flow_limiter(**sfl)

        # constant floodplain size
        floodplain = self._hfun_config.get("floodplain_size", False)
        if floodplain is not False:
            hfun.floodplain_size(floodplain)

        # custom features
        for id, opts in self._hfun_features.items():
            features = self._get_feature_collection_by_id(id, hfun._dst_crs)
            msg = '"features" entry must contain a "target_size".'
            assert "target_size" in opts, msg
            # It seems that we can handle MultiLineString more
            # efficiently than LineString and it seems to be natively
            # supported by add_feature method. There seemsno obvious reason to
            # innput each feature individually, but was consider
            hfun.add_feature(MultiLineString(features), **opts)

        # add gaussian filter requests
        for kwargs in self._hfun_config.get("gaussian_filter", {}).values():
            hfun.add_gaussian_filter(**kwargs)

        return hfun

    @property
    def verbosity(self):
        return self.args.verbosity

# ------- auxilliary functions

    def _resolve_raster_id(self, id):

        def is_tile_index(uri):
            try:
                fiona.open(uri, 'r')
                return True
            except Exception:  # DriverError exception
                return False

        def is_raster(uri):
            try:
                rasterio.open(uri, 'r')
                return True
            except Exception:  # RasterioIOError exception
                return False

        raster = self._rasters_config.get(id, None)
        if raster is None:
            raise Exception(f'No raster defined with id {id}')

        uri = raster.get('uri', None)

        if uri is None:
            msg = f'URI not provided for raster with ID {id}'
            raise Exception(msg)

        is_tile_index = is_tile_index(uri)
        is_raster = is_raster(uri)

        if is_tile_index:
            with fiona.open(uri, 'r') as src:
                uris = []
                for feature in src:
                    url = feature.get('properties', {}).get('URL', None)
                    if url is None:
                        msg = f'No URL given for feature with id {id}'
                        raise Exception(msg)
                    fname = self._cache / url.split('/')[-1]
                    if not fname.is_file() or fname.stat().st_size == 0:
                        open(fname, 'wb').write(
                            requests.get(
                                url, allow_redirects=True).content)
                    uris.append(fname)
            return uris

        elif is_raster:
            return [uri]

    def _get_feature_collection_by_id(self, id, dst_crs):
        uri = self._features_config[id].get('uri', None)
        if uri is None:
            msg = f'Must specify at least one "uri" for feature with id {id}.'
            raise Exception(msg)
        feature_collection = []
        shp = fiona.open(uri)
        shp_crs = CRS.from_user_input(shp.crs)
        dst_crs = CRS.from_user_input(dst_crs)
        if shp_crs.srs != dst_crs.srs:
            transformer = Transformer.from_crs(
                shp.crs, dst_crs, always_xy=True)
            _tflag = True
        else:
            _tflag = False
        for feature in shp:
            geometry = shape(feature['geometry'])
            if _tflag:
                geometry = transform(transformer.transform, geometry)
            feature_collection.append(geometry)
        shp.close()
        return feature_collection

# ---------- auxilliary variables
    @property
    @lru_cache
    def _config(self):
        with open(pathlib.Path(self.args.config_file), 'r') as js_file:
            return json.loads(js_file.read())

    @property
    @lru_cache
    def _rasters_config(self):
        rasters = self._config.get("rasters", {})
        msg = '"rasters" entry must be a dictionary, not '
        msg += f'{type(rasters)}.'
        assert isinstance(rasters, dict), msg
        for id, data in rasters.items():
            msg = f'No "uri" entry for raster with id: {id}'
            assert 'uri' in data, msg
        return rasters

    @property
    @lru_cache
    def _features_config(self):
        features = self._config.get("features", {})
        msg = '"features" entry must be a dictionary, not '
        msg += f'{type(features)}.'
        assert isinstance(features, dict), msg
        for id, data in features.items():
            msg = f'No "uri" entry for raster with id: {id}'
            assert 'uri' in data, msg
        return features

    @property
    @lru_cache
    def _geom_config(self):
        geom = self._config.get('geom', None)
        if geom is None:
            msg = 'No "geom" data in configuration file.'
            raise Exception(msg)
        msg = '"geom" entry must be a dictionary.'
        assert isinstance(geom, dict), msg
        return geom

    @property
    @lru_cache
    def _hfun_config(self):
        hfun = self._config.get('hfun', None)
        if hfun is not None:
            msg = '"hfun" entry must be a constant or a dictionary.'
            assert isinstance(hfun, (float, int, dict)), msg
        # if isinstance(hfun, dict):
        #     msg = '"hfun" must contain a "raster" dictionary of raster id '
        #     msg += 'keys.'
        #     assert "rasters" in hfun, msg
        return hfun

    @property
    @lru_cache
    def _levees_config(self):
        features = self._config.get("levees", None)
        if features is not None:
            msg = '"levees" entry must be a dictionary, not '
            msg += f'{type(features)}.'
            assert isinstance(features, dict), msg
            msg = 'feature specified as levee is not in not in "features" keys'
            for id in features:
                assert id in self._features_config.keys(), msg
        return features

    @property
    @lru_cache
    def _driver_config(self):
        pass
        # outputs = self._config.get("outputs", None)
        # if outputs is None:
        #     warnings.warn('No outputs set in configuration file.')
        # else:
        #     msg = '"outputs" entry must be a dictionary.'
        #     assert isinstance(outputs, dict), msg
        # return outputs
        return {}

    @property
    @lru_cache
    def _outputs_config(self):
        outputs = self._config.get("outputs", None)
        if outputs is None:
            warnings.warn('No outputs set in configuration file.')
        else:
            msg = '"outputs" entry must be a dictionary.'
            assert isinstance(outputs, dict), msg
        return outputs

    @property
    @lru_cache
    def _boundaries_config(self):
        boundaries = self._config.get("boundaries", None)
        if boundaries is not None:
            if boundaries is True:
                pass
            else:
                raise Exception('Certify boundaries inputs')
        return boundaries

    @property
    @lru_cache
    def _geom_raster_collection(self):
        # certify rasters
        rasters = self._geom_config.get("rasters", None)
        if rasters is not None:
            msg = '"rasters" must be a list of rasters id\'s.'
            assert isinstance(rasters, list), msg
            msg = 'raster specified to geom is not in not in "rasters" key'
            for key in rasters:
                assert key in self._rasters_config.keys(), msg
        else:
            rasters = []
        raster_collection = geomesh.raster.RasterCollection()
        for id in rasters:
            for fname in self._resolve_raster_id(id):
                raster_collection.append(fname)
        return raster_collection

    @property
    @lru_cache
    def _geom_features(self):
        # certify features
        features = self._geom_config.get("features", None)
        if features is not None:
            if isinstance(features, dict):
                if 'clip' in features:
                    msg = '"clip" must be a list of features id\'s.'
                    assert isinstance(features['clip'], list), msg
                    msg = 'feature specified to geom is not in not in '
                    msg += '"features" key'
                    for key in features['clip']:
                        assert key in self._features_config.keys(), msg
                if 'buffer' in features:
                    msg = '"buffer" must be a list of features id\'s.'
                    assert isinstance(features['buffer'], list), msg
                    msg = 'feature specified to geom is not in not in '
                    msg += '"features" key'
                    for key in features['buffer']:
                        assert key in self._features_config.keys(), msg
            if 'clip' not in features and 'buffer' not in features:
                raise Exception("need to specified clip, buffer or both")
        else:
            features = []
        return features

    @property
    def _geom_crs(self):
        return self._geom_config.get("crs", "EPSG:3395")

    @property
    @lru_cache
    def _geom_clip(self):
        # certify features
        features = self._geom_config.get("features", None)
        if features is None:
            return
        clip = features.get("clip", None)
        if clip is not None:
            msg = '"clip" must be a list of features id\'s.'
            assert isinstance(clip, list), msg
        msg = 'feature specified to geom is not in not in "features" key'
        fc = []
        for id in features.get("clip", []):
            assert id in self._features_config.keys(), msg
            feats = self._get_feature_collection_by_id(id, self._geom_crs)
            for feat in feats:
                msg = '"clip" must be Polygon or MultiPolygon'
                assert isinstance(feat, (Polygon, MultiPolygon)), msg
                if isinstance(feat, Polygon):
                    fc.append(feat)
                else:
                    for polygon in feat:
                        fc.append(feat)
        return MultiPolygon(fc).buffer(0)

    @property
    @lru_cache
    def _geom_buffer(self):
        # certify features
        features = self._geom_config.get("features", None)
        if features is not None:
            buffer = features.get("buffer", None)
            if buffer is not None:
                msg = '"buffer" must be a list of features id\'s.'
                assert isinstance(buffer, list), msg
            msg = 'feature specified to geom is not in not in "features" key'
            for key in features.get("buffer", {}):
                assert key in self._features_config.keys(), msg
            return buffer

    @property
    @lru_cache
    def _hfun_rasters(self):
        rasters = self._hfun_config.get("rasters", None)
        if rasters is None:
            msg = '""hfun" must contain a "rasters" entry and it must be '
            msg += 'a dictionary of raster id\'s.'
            assert isinstance(rasters, dict), msg
            msg = 'raster specified to geom is not in not in "rasters" key'
            for key in self._hfun_config["rasters"]:
                assert key in self._rasters_config.keys(), msg
        raster_collection = geomesh.raster.RasterCollection()
        for id in rasters:
            for fname in self._resolve_raster_id(id):
                raster_collection.append(fname)
        return raster_collection

    @property
    @lru_cache
    def _hfun_features(self):
        features = self._hfun_config.get("features", {})
        # certify Mapping type
        msg = '"features" must be a dictionray of features id\'s.'
        assert isinstance(features, dict), msg
        # certify contents
        msg = 'feature specified to hfun is not in not in "features" key'
        for id in features:
            assert id in self._features_config.keys(), msg
        return features

    @property
    @lru_cache
    def _interp_rasters(self):
        raster_collection = geomesh.raster.RasterCollection()
        for id in self._rasters_config:
            for fname in self._resolve_raster_id(id):
                raster_collection.append(fname)
        return raster_collection

# ------- database calls
    @property
    @lru_cache
    def _cache(self):
        if self.args.cache_dir is not None:
            self.__cache = pathlib.Path(self.args.cache_dir)
            self.__cache.mkdir(parents=True, exist_ok=True)
            return self.__cache
        else:
            self.__cache = tempfile.TemporaryDirectory()
            return pathlib.Path(self.__cache.name)


# -------- interface
def print_config_file_template():
    """
    Prints extra help for creating a configuration file.
    """
    config = dict()
    print(config)
    raise NotImplementedError


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file",
        help="Path to configuration file.",
        # nargs='?'
        )
    parser.add_argument("--log-level", choices=["info", "debug", "warning"])
    parser.add_argument("--verbosity", type=int, default=0)
    parser.add_argument("--cache-dir")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def init_logger(args):
    if args.log_level is not None:
        logging.basicConfig(level={
            "info": logging.INFO,
            "debug": logging.DEBUG,
            "warning": logging.WARNING,
        }[args.log_level])
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('fiona').setLevel(logging.WARNING)
        logging.getLogger('rasterio').setLevel(logging.WARNING)


def main():
    args = parse_args()
    init_logger(args)
    Geomesh(args).run()


if __name__ == '__main__':
    main()
