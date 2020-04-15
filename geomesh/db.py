import pathlib
import sys
import sqlite3

PYENV_PREFIX = pathlib.Path("/".join(sys.executable.split('/')[:-2]))
CACHE = PYENV_PREFIX / ".cache"
CACHE.mkdir(exist_ok=True)
DBPATH = CACHE / '._db.sqlite'


def get_conn():
    return sqlite3.connect(DBPATH)


def init_db():
    if not DBPATH.is_file():
        with sqlite3.connect(DBPATH) as conn:
            conn.enable_load_extension(True)
            conn.load_extension("mod_spatialite")
