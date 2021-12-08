import pathlib
from importlib import util
import tempfile
import os
import sys
import platform


try:
    import jigsawpy  # noqa: F401
except OSError as e:
    pkg = util.find_spec("jigsawpy")
    libjigsaw = {
            "Windows": "jigsaw.dll",
            "Linux": "libjigsaw.so",
            "Darwin": "libjigsaw.dylib"
            }[platform.system()]
    tgt_libpath = pathlib.Path(pkg.origin).parent / "_lib" / libjigsaw
    pyenv = pathlib.Path("/".join(sys.executable.split('/')[:-2]))
    src_libpath = pyenv / 'lib' / libjigsaw
    if not src_libpath.is_file():
        raise e

    os.symlink(src_libpath, tgt_libpath)


from .raster import Raster
from .mesh import Mesh
from .geom import Geom
from .hfun import Hfun
from .driver import JigsawDriver

if util.find_spec("colored_traceback") is not None:
    import colored_traceback
    colored_traceback.add_hook(always=True)

tmpdir = str(pathlib.Path(tempfile.gettempdir()+'/ocsmesh'))+'/'
os.makedirs(tmpdir, exist_ok=True)

__all__ = [
    "Geom",
    "Hfun",
    "Raster",
    "Mesh",
    "JigsawDriver",
]

# mpl.rcParams['agg.path.chunksize'] = 10000
