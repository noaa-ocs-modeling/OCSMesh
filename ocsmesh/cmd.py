import logging
import json
import pathlib
import os
from functools import lru_cache
import hashlib
from multiprocessing import Pool

from pyproj import CRS
from shapely import ops
from shapely.geometry import MultiPolygon, box
import fiona
import geoalchemy2

from ocsmesh import Hfun, Geom, Raster, db


class _ConfigManager:

    def __init__(self, args, session):
        self._args = args
        self._session = session
        self._certify_config()

    def get_geom(self):
        self._logger.debug('get_geom()')

        geom_collection = []

        if self._geom_nprocs is not None:
            job_args = []
            hashes = []
            geom = None

        for id, geom_opts in self._geom.items():
            self._logger.debug(f'get_geom(): processsing group id={id}')
            zmin = geom_opts.get("zmin")
            zmax = geom_opts.get("zmax")
            driver = geom_opts.get("driver", "matplotlib")

            for raster_path, raster_opts in self._get_raster_by_id(id):
                self._logger.debug(
                    f'get_geom(): appending raster {raster_path} for '
                    'parallel processing.')
                hash = _geom_identifier(
                    zmin, zmax, driver, Raster(raster_path).md5)
                query = self._session.query(db.GeomCollection).get(hash)
                if query is None:
                    chunk_size = raster_opts.get("chunk_size")
                    if self._geom_nprocs is not None:
                        if chunk_size == 0:
                            job_args.append((
                                raster_path, raster_opts, zmin, zmax,
                                geom_opts.get("join_method"),
                                driver,
                                chunk_size,
                                raster_opts.get("overlap", 2)))
                            hashes.append(hash)
                        else:
                            geom = _geom_raster_processing_worker(
                                raster_path, raster_opts, zmin, zmax,
                                geom_opts.get("join_method"),
                                driver, chunk_size,
                                raster_opts.get("overlap"))
                    else:
                        geom = _geom_raster_processing_worker(
                            raster_path, raster_opts, zmin, zmax,
                            geom_opts.get("join_method"),
                            driver, chunk_size,
                            raster_opts.get("overlap"))
                        self._save_geom_to_db(
                            geom, raster_path.name, zmin, zmax, driver, hash)
                        self._session.commit()
                else:
                    geom = Geom(
                        geoalchemy2.shape.to_shape(query.geom),
                        crs=self._crs)

                if geom is not None:
                    geom_collection.append(geom)

        if self._geom_nprocs is not None:
            self._logger.debug(
                'get_geom(): executing parallel geom computations...')
            with Pool(processes=self._geom_nprocs) as pool:
                res = pool.starmap(_geom_raster_processing_worker, job_args)
            pool.join()
            for i, geom in enumerate(res):
                geom_collection.append(geom)
                self._save_geom_to_db(
                    geom, job_args[i][0].name, job_args[i][2],
                    job_args[i][3], job_args[i][5], hashes[i])
            self._session.commit()
            del res

        for feature in self._features:
            raise NotImplementedError('features')

        mpc = []
        for geom in geom_collection:
            mpc.append(geom.multipolygon)
        self._logger.debug('get_geom(): apply unary_union...')
        mp = ops.unary_union(mpc)

        return Geom(mp, crs=self._crs)

    def get_hfun(self, geom=None):
        self._logger.debug('get_hfun()')

        if 'rasters' in self._hfun:
            for id, hfun_raster_opts in self._hfun['rasters'].items():
                self._logger.debug(f'get_hfun(): processsing group id={id}')

                for raster_path, raster_opts in self._get_raster_by_id(id):
                    self._logger.debug(
                        f'get_hfun(): appending raster {raster_path} for '
                        'parallel processing.')
                    raster = Raster(raster_path, crs=raster_opts.get('crs'))
                    _apply_raster_opts(raster, raster_opts)
                    hfun = Hfun(
                        raster,
                        hmin=self._hfun_hmin,
                        hmax=self._hfun_hmax,
                        nprocs=self._hfun_nprocs)
                    _apply_hfun_raster_opts(hfun, hfun_raster_opts)
                    mesh = hfun.get_mesh(geom=geom)

        if 'features' in self._hfun:
            raise NotImplementedError("config.hfun.features not implemented")

        return Hfun(mesh, crs=self._crs)

    def _certify_config(self):
        # pylint: disable=W0104

        self._config
        self._rasters
        self._features
        self._geom
        self._hfun
        self._outputs
        self._logger.debug(" done checking configuration file")

    def _get_raster_by_id(self, rast_id):

        def check_if_uri_is_tile_index(uri):
            try:
                fiona.open(uri, 'r')
                return True
            except fiona.errors.DriverError:
                return False

        raster_opts = self._rasters[rast_id]

        if 'http' in raster_opts['uri'] or 'ftp' in raster_opts['uri']:
            raise NotImplementedError("URI is internet address")

        uri = pathlib.Path(os.path.expandvars(raster_opts['uri']))
        uri = pathlib.Path(self._path).parent / uri

        if not uri.is_file():
            raise FileNotFoundError(f"No file with path: {str(uri.resolve())}")

        if check_if_uri_is_tile_index(uri):
            raise NotImplementedError('URI is a tile index')

        chunk_size = raster_opts.get("chunk_size")
        if chunk_size is None:
            chunk_size = self._config.get("chunk_size")
        if chunk_size is None:
            chunk_size = self._args.chunk_size
        raster_opts.update({"chunk_size": chunk_size})
        return [(uri, raster_opts)]

    def _save_geom_to_db(self, geom, source, zmin, zmax, driver, key):
        self._logger.debug("_save_geom_to_db()")
        _original_crs = geom.crs
        if not _original_crs.equals(CRS.from_epsg(4326)):
            self._logger.debug(f"tranforming from {geom.crs} to EPSG:4326")
            geom.transform_to('EPSG:4326')
        self._session.add(
            db.GeomCollection(
                geom=geoalchemy2.shape.from_shape(geom.multipolygon),
                source=source,
                zmin=zmin,
                zmax=zmax,
                driver=driver,
                id=key))
        if not geom.crs.equals(_original_crs):
            geom.transform_to(_original_crs)

    @property
    @lru_cache(maxsize=None)
    def _config(self):
        with open(self._path, 'r') as f:
            config = json.loads(f.read())

        if not isinstance(config, dict):
            raise TypeError('config json must be a dictionary.')

        if not any(x in config.keys() for x in ['rasters', 'features']):
            raise KeyError(
                "Configuration file must contain at least one of 'rasters' "
                "or 'features' keys.")

        return config

    @property
    def _path(self):
        return self._args.config_file

    @property
    @lru_cache(maxsize=None)
    def _crs(self):
        return CRS.from_user_input(self._config.get("crs", "EPSG:4326"))

    @property
    @lru_cache(maxsize=None)
    def _rasters(self):

        config_rasters = self._config.get('rasters')

        config_rasters = {} if config_rasters is None else config_rasters

        if not isinstance(config_rasters, (dict, list)):
            raise TypeError(
                "config.rasters must be a dictionary or list of "
                "dictionaries containing at least one 'id' and one 'uri' "
                "key.")

        if isinstance(config_rasters, dict):
            config_rasters = [config_rasters]

        _config_rasters = {}

        for i, config_raster in enumerate(config_rasters):

            if not isinstance(config_raster, dict):
                raise TypeError(
                    "config.rasters must be a dictionary or list of "
                    "dictionaries containing at least one 'id' and one "
                    "'uri' key.")

            config_raster_id = config_raster.get('id')

            if config_raster_id is None:
                raise KeyError(
                    "config.rasters entry must contain a unique 'id' key.")

            if config_raster_id in _config_rasters:
                raise KeyError(
                    "'id' entry in config.raster must be unique. "
                    f"repeated key: {config_raster_id}")

            config_raster_uri = config_raster.get('uri')

            if config_raster_uri is None:
                raise KeyError(
                    "config.rasters entry must contain a 'uri' key "
                    "(mutually exclusive).")

            if 'http' in config_raster_uri or 'ftp' in config_raster_uri:
                raise NotImplementedError(
                    "URI provided is an internet address.")

            opts = config_rasters[i].copy()
            opts.pop('id')
            _config_rasters.update({config_raster_id: opts})

        return _config_rasters

    @property
    @lru_cache(maxsize=None)
    def _features(self):
        config_features = self._config.get('features')
        # _features = [] if config_features is None else config_features
        if config_features is not None:
            raise NotImplementedError("config.features is not yet implemented")
        return []

    @property
    @lru_cache(maxsize=None)
    def _geom(self):
        config_geom = self._config.get("geom")
        if config_geom is None:
            raise KeyError("Configuration file must contain a 'geom' key.")
        _geom = {}
        if not isinstance(config_geom, dict):
            raise TypeError(
                "config.geom must be a dictionary containing 'rasters' or "
                "'features' keys.")

        if not any(
                x in config_geom.keys() for x in ['rasters', 'features']):
            raise TypeError(
                "config.geom must be a dictionary containing 'rasters' or "
                "'features' keys.")

        if 'rasters' in config_geom:
            config_geom_rasters = config_geom["rasters"].copy()
            if not isinstance(config_geom_rasters, (dict, list)):
                raise TypeError(
                    "geom.rasters must be a dictionary or list of dictionaries"
                    " containing at least one 'id' key that matches an some "
                    "id on the 'rasters' key.")
            if isinstance(config_geom_rasters, dict):
                config_geom_rasters = [config_geom_rasters]
            for geom_raster in config_geom_rasters:
                geom_raster_id = geom_raster.pop('id')
                if geom_raster_id in self._rasters:
                    _geom.update({geom_raster_id: geom_raster})
                else:
                    raise KeyError(
                        f'No raster with id={geom_raster_id} specified in '
                        'config.rasters')

        if 'features' in config_geom:
            raise NotImplementedError(
                'config.geom.features not yet implemented')

        return _geom

    @property
    @lru_cache(maxsize=None)
    def _hfun(self):
        config_hfun = self._config.get("hfun")
        if config_hfun is None:
            raise KeyError("Configuration file must contain a 'hfun' key.")

        if not isinstance(config_hfun, (int, float, dict)):
            raise TypeError(
                "config.geom must be an scalar value (constant size) or a "
                "dictionary containing either the 'rasters' or 'features' "
                "keys, or both.")

        if isinstance(config_hfun, (int, float)):
            raise NotImplementedError('Constant size funtion.')

        _config_hfun = {}
        _config_hfun.update({"hmin": config_hfun.get("hmin")})
        _config_hfun.update({"hmax": config_hfun.get("hmax")})

        _config_hfun_rasters = config_hfun.get("rasters").copy()
        if _config_hfun_rasters is not None:
            _config_hfun.update({"rasters": {}})
            if not isinstance(_config_hfun_rasters, (list, dict)):
                raise TypeError(
                    'config.hfun.rasters must be a dictionary or list of '
                    'dictionaries that contain an "id" key that matches a '
                    'key from the config.rasters entry.')
            if isinstance(_config_hfun_rasters, dict):
                _config_hfun_rasters = [_config_hfun_rasters]

            for hfun_raster_opts in _config_hfun_rasters:
                hfun_raster_id = hfun_raster_opts.pop('id')
                if hfun_raster_id not in self._rasters.keys():
                    raise KeyError(
                        f'No raster with id={hfun_raster_id} specified in '
                        'config.rasters')
                contours = hfun_raster_opts.get("contours", [])
                if isinstance(contours, dict):
                    contours = [contours]
                hfun_raster_opts.update({"contours": contours})
                _config_hfun["rasters"].update(
                        {hfun_raster_id: hfun_raster_opts})

                # _geom.update({geom_raster_id: geom_raster})

        if 'features' in config_hfun:
            raise NotImplementedError(
                'config.hfun.features not yet implemented')

        return _config_hfun

    @property
    def _outputs(self):
        config_outputs = self._config.get("outputs")
        _outputs = {}
        return _outputs

    @property
    @lru_cache(maxsize=None)
    def _geom_nprocs(self):
        nprocs = self._config["geom"].get("nprocs")
        if nprocs is None:
            nprocs = self._config.get("nprocs")
        if nprocs is None:
            nprocs = self._args.geom_nprocs
        if nprocs is None:
            nprocs = self._args.nprocs
        return nprocs

    @property
    @lru_cache(maxsize=None)
    def _hfun_nprocs(self):
        nprocs = self._config["hfun"].get("nprocs")
        if nprocs is None:
            nprocs = self._config.get("nprocs")
        if nprocs is None:
            nprocs = self._args.hfun_nprocs
        if nprocs is None:
            nprocs = self._args.nprocs
        return nprocs

    @property
    def _hfun_hmin(self):
        return self._config["hfun"].get("hmin")

    @property
    def _hfun_hmax(self):
        return self._config["hfun"].get("hmax")

    @property
    def _hfun_raster_opts(self):
        opts = self._config["hfun"]["rasters"]

    @property
    @lru_cache(maxsize=None)
    def _logger(self):
        return logging.getLogger(__name__ + '.' + self.__class__.__name__)


def _geom_identifier(zmin, zmax, driver, salt):
    zmin = "" if zmin is None else f"{zmin:G}"
    zmax = "" if zmax is None else f"{zmax:G}"
    return hashlib.md5(
            f"{zmin}{zmax}{driver}{salt}".encode('utf-8')
            ).hexdigest()


def _hfun_identifier(config_hfun):
    pass


def _apply_raster_opts(raster, raster_opts):
    for key, opt in raster_opts.items():
        if key == "resample":
            if isinstance(opt, (float, int)):
                raster.resample(opt)
            else:
                raster.resample(**opt)
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
                            opt['ymax'])]))
            else:
                raise NotImplementedError("clip by geometry")
        if key == 'chunk_size':
            raster.chunk_size = opt

        if key == 'overlap':
            raster.overlap = opt


def _apply_hfun_raster_opts(hfun, hfun_raster_opts):
    for key, opts in hfun_raster_opts.items():
        if key == 'contours':
            for kwargs in opts:
                hfun.add_contour(**kwargs)
        if key == 'features':
            for kwargs in opts:
                hfun.add_features(**kwargs)
        if key == 'subtidal_flow_limiter':
            if not isinstance(opts, (bool, dict)):
                raise TypeError(
                    "subtidal_flow_limiter options must be "
                    "a boolean or dict.")
            if opts is True:
                hfun.add_subtidal_flow_limiter()
            else:
                hfun.add_subtidal_flow_limiter(**opts)


def _geom_raster_processing_worker(
        raster_path,
        raster_opts,
        zmin,
        zmax,
        join_method,
        driver,
        chunk_size,
        overlap,
):
    raster = Raster(raster_path)
    _apply_raster_opts(raster, raster_opts)
    geom = Geom(raster.get_multipolygon(
        zmin=zmin,
        zmax=zmax),
        join_method=join_method,
        driver=driver,
        nprocs=1)

    return geom
