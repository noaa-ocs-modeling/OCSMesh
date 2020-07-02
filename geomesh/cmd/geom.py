import hashlib
import json
from shapely import ops
import geoalchemy2
from geomesh import cmd
from geomesh import Geom
from geomesh.cmd import db


def sanitize_config(config):
    geom = config.get('geom')
    if geom is None:
        msg = "config file must contain a 'geom' key."
        raise AttributeError(msg)

    rasters = geom.get("rasters")
    features = geom.get("features")

    if rasters is None and features is None:
        msg = "'geom' must contain either a 'rasters' or 'features' entry or "
        msg += "both."
        raise AttributeError(msg)

    if rasters is not None:
        _validate_rasters(config)

    if features is not None:
        _validate_features(config)

    _load_geom(config)


def _validate_rasters(config):
    irasters = {iraster['id']: iraster for iraster in config['rasters']}
    grasters = {
        graster['id']: graster for graster in config["geom"].get("rasters", [])
        }
    for i, (id, opts) in enumerate(grasters.items()):
        msg = f"config.geom.rasters.id position {i} with "
        msg += f"id '{id}' does not match any entry on config.rasters.id"
        assert id in irasters, msg


def _validate_features(config):
    raise NotImplementedError("_validate_features")


def _load_geom(config):
    geom_collection = _load_raster_geom_collection(config)
    raise Exception('debug')
    multipolygon = _unify_geom_collection(geom_collection)
    s = db.session(config["cache"])
    s.add(db.Geom(
        geom=geoalchemy2.shape.from_shape(multipolygon, srid=-1),
        # config=None,
        id=0
        ))
    s.commit()


def _unify_geom_collection(geom_collection):
    mpc = []
    for geom in geom_collection:
        mpc.append(geom.multipolygon)
    return ops.unary_union(mpc)


def _load_raster_geom_collection(config):
    geom_collection = []
    irasters = {iraster['id']: iraster for iraster in config['rasters']}
    grasters = {
        graster['id']: graster for graster in config["geom"].get("rasters", [])
        }
    session = db.session(config['cache'])
    for id, graster in grasters.items():
        source = irasters[id]["_src_path"].name
        zmin = graster.get("zmin")
        zmax = graster.get("zmax")
        driver = graster.get("driver", "matplotlib")
        salt = irasters[id]["_md5"]
        key = _get_geom_key(
                zmin,
                zmax,
                driver,
                salt,
                )
        geom = session.query(db.GeomCollection).get(key)
        if geom is None:
            geom = _get_raster_geom(irasters[id], graster)
            _put_raster_geom(session, geom, source, zmin, zmax, driver, key)
        else:
            geom = Geom(
                geoalchemy2.shape.to_shape(geom.geom),
                src_crs=4326
                )
        geom_collection.append(geom)
    return geom_collection


def _get_geom_key(zmin, zmax, driver, salt):
    zmin = "" if zmin is None else f"{zmin:G}"
    zmax = "" if zmax is None else f"{zmax:G}"
    return hashlib.md5(
        f"{zmin}{zmax}{driver}{salt}".encode('utf-8')
        ).hexdigest()


def _put_raster_geom(session, geom, source, zmin, zmax, driver, key):
    epsg = geom.crs.to_epsg()
    if epsg is not None:
        epsg = -1

    session.add(db.GeomCollection(
        geom=geoalchemy2.shape.from_shape(geom.multipolygon, srid=epsg),
        source=source,
        zmin=zmin,
        zmax=zmax,
        driver=driver,
        id=key
        ))
    session.commit()


def _get_raster_geom(raster, opts):
    rast = cmd.rasters._load_rast(raster)
    cmd.rasters._unload_rast(raster)
    return rast.get_geom(
        zmin=opts.get('zmin'),
        zmax=opts.get('zmax'),
        driver=opts.get('driver'),
        join_method=opts.get('join_method'),
        overlap=opts.get("overlap")
        )
