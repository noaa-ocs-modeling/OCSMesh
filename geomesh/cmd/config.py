import pathlib
import json
from geomesh import cmd


def read_config(path):

    with open(pathlib.Path(path).resolve(), 'r') as f:
        config = json.loads(f.read())

    _append_file_path(path, config)
    _update_cache_dir(config)

    cmd.rasters.sanitize_config(config)
    cmd.geom.sanitize_config(config)

    return config


def _append_file_path(path, config):
    config.update({"_path": pathlib.Path(path)})


def _update_cache_dir(config):
    cache = config.get('cache')
    if cache is None:
        cache = config["_path"].parent / 'cache.db'
    else:
        cache = pathlib.Path(cache)
    config.update({'cache': cache})
