import os
import pathlib
import hashlib
import fiona
from shapely.geometry import MultiPolygon, box
from geomesh import Raster


def sanitize_config(config):
    rasters = config.get("rasters")

    # validate top level: raise if no "rasters" entry is found
    if rasters is None:
        msg = "config file must contain a 'rasters' key with a list of "
        msg += "ditionaries."
        raise AttributeError(msg)

    # cast dictionary to list
    if isinstance(rasters, dict):
        rasters = [rasters]

    # validate top level input type
    for raster in rasters:
        msg = "'rasters' entry must be a dictionary."
        assert isinstance(raster, dict), msg

    # validate fields
    _validate_ids(rasters)
    for i, raster in enumerate(rasters):
        _validate_uri(i, raster, parent=config["_path"].parent)
        _validate_clip(raster)
        _insert_hash(raster)


def _validate_ids(rasters):
    # check that each raster entry contains 'id' field.
    for i, raster in enumerate(rasters):
        id = raster.get('id')
        if id is None:
            msg = f"'raster' in position {i} does not contain an id entry. "
            msg += "All raster entries must contain a unique id."
            raise AttributeError(msg)

    # check that inputs do not contain duplicate id's
    input_ids = [raster['id'] for raster in rasters]
    id_duplicates = set([i for i in input_ids if input_ids.count(i) > 1])
    if len(id_duplicates) > 0:
        id_duplicates = " ".join(list(id_duplicates))
        msg = f"Duplicate id found: {id_duplicates}\n"
        msg += "config.rasters entries must not contain duplicate ids."
        raise AttributeError(msg)


def _validate_uri(i, raster, parent=None):
    uri = raster.get('uri')
    if uri is None:
        msg = f"'raster' in position {i} does not contain an uri entry."
        msg += "All raster entries must contain a unique URI."
        raise AttributeError(msg)

    msg = "URI must be a string pointing to an internet resource or local "
    msg += "file path."
    assert isinstance(uri, str), msg
    if 'http' in uri or 'ftp' in uri:
        msg = "TODO: URI provided is an internet address."
        raise NotImplementedError(msg)

    else:
        _path = pathlib.Path(os.path.expandvars(uri))

        if _is_tile_index(_path):
            raise NotImplementedError

        if parent is not None:
            _path = parent / _path
        raster.update({
            "_src_path": _path.resolve()
            })


def _validate_tile_index(path):
    with fiona.open(path) as src:
        for feature in src:
            url = feature['properties'].get("URL")
            if url is None:
                msg = f"No 'URL' entry for feature with id {feature['id']} "
                msg += f"on file {path}"
                raise AttributeError(msg)
            print(url)
    exit()


def _is_tile_index(path):

    try:
        fiona.open(path)
        return True

    except fiona.errors.DriverError:
        return False


def _validate_clip(raster):
    clip = raster.get('clip')
    if clip is None:
        return
    msg = "'clip' entry must be a URI to a clipping geometry or a "
    msg += "dictionary with keys xmin, xmax, ymin, ymax."
    assert isinstance(clip, (str, dict)), msg

    if isinstance(clip, str):
        msg = 'Geometry based clipping not yet implemented.'
        raise NotImplementedError(msg)

    if isinstance(clip, dict):
        raster["clip"].update({"_multipolygon": MultiPolygon(
                [box(
                    clip['xmin'],
                    clip['ymin'],
                    clip['xmax'],
                    clip['ymax'])])})


def _insert_hash(raster):
    raster.update({
        "_md5": _get_raster_hash(raster)
        })


def _load_rast(raster):
    rast = Raster(raster["_src_path"], src_crs=raster.get("src_crs"))
    for key, opt in raster.items():
        if key == "resample":
            rast.resample(opt['scaling_factor'])
        if key == 'warp':
            rast.warp(opt)
        if key == 'fill_nodata':
            rast.fill_nodata()
        if key == 'clip':
            if isinstance(opt, dict):
                rast.clip(
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

    # if clip is not None:
    #     rast.clip(clip)
    # warp = raster.get("warp")
    # if warp is not None:
    #     rast.warp(warp)

    raster.update({"_rast": rast})
    return rast


def _get_raster_hash(raster):
    """
    https://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file
    """
    _load_rast(raster)
    hash_md5 = hashlib.md5()
    with open(raster["_rast"]._tmpfile.resolve(), "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    _unload_rast(raster)
    return hash_md5.hexdigest()


def _unload_rast(raster):
    raster.pop("_rast")
