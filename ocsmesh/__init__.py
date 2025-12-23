import pathlib
from importlib import util
import tempfile
import os
import sys
import platform
import warnings

# Major API Change Warning
warnings.warn(
"\n\n"
"********************************************************************************\n"
"                  MAJOR UPDATE ALERT - Upcoming OCSMesh 2.0.0                   \n"
"********************************************************************************\n"
"OCSMesh is going through significant architectural changes.\n"
"The current architecture will be depracated soon, among the upcoming changes we highlight:\n"
"1. Mesh Engines (Jigsaw and Triangle) will become OPTIONAL dependencies.\n"
"   After version 2.0.0 you will have to install them manually.\n"
"2. A new default mesh engine will be adopted to replace Jigsaw.\n"
"3. 'jigsaw_msh_t' will be replaced by a new custom Python class called 'MeshData'.\n"
"4. Functions operating on msh_t objects will be updated to support MeshData.\n"
"This v1.7.0 is the last release version before OCSMesh 2.0.0.\n"
"********************************************************************************\n",
    FutureWarning,
    stacklevel=2
)

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
