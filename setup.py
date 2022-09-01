#! /usr/bin/env python
import setuptools
import subprocess
import setuptools.command.build_py
import distutils.cmd
import distutils.util
import shutil
import platform
from multiprocessing import cpu_count
from pathlib import Path
import sys
import os

from dunamai import Version

PARENT = Path(__file__).parent.absolute()
PYENV_PREFIX = Path("/".join(sys.executable.split('/')[:-2]))
SYSLIB = {
    "Windows": "jigsaw.dll",
    "Linux": "libjigsaw.so",
    "Darwin": "libjigsaw.dylib"}

if "install_jigsaw" not in sys.argv:
    if "develop" not in sys.argv:
        if "install" in sys.argv:
            libsaw = PYENV_PREFIX / 'lib' / SYSLIB[platform.system()]
            if not libsaw.is_file():
                subprocess.check_call(
                    [sys.executable, "setup.py", "install_jigsaw"])


class InstallJigsawCommand(distutils.cmd.Command):
    """Custom build command."""

    user_options = []

    def initialize_options(self): pass

    def finalize_options(self): pass

    def run(self):
        self.announce('Loading JIGSAWPY from GitHub', level=3)
        # init jigsaw-python submodule
        subprocess.check_call(
            ["git", "submodule", "update",
             "--init", "submodules/jigsaw-python"])
        # install jigsawpy
        os.chdir(PARENT / 'submodules/jigsaw-python')
        subprocess.check_call(["git", "checkout", "master"])
        self.announce('INSTALLING JIGSAWPY', level=3)
        subprocess.check_call(["python", "setup.py", "install"])
        # install jigsaw
        self.announce(
            'INSTALLING JIGSAW LIBRARY AND BINARIES FROM '
            'https://github.com/dengwirda/jigsaw-python', level=3)
        os.chdir("external/jigsaw")
        os.makedirs("build", exist_ok=True)
        os.chdir("build")
        gcc, cpp = self._check_gcc_version()
        subprocess.check_call(
            ["cmake", "..",
             "-DCMAKE_BUILD_TYPE=Release",
             f"-DCMAKE_INSTALL_PREFIX={PYENV_PREFIX}",
             f"-DCMAKE_C_COMPILER={gcc}",
             f"-DCMAKE_CXX_COMPILER={cpp}",
             ])
        subprocess.check_call(["make", f"-j{cpu_count()}", "install"])
        libsaw_prefix = list(PYENV_PREFIX.glob("**/*jigsawpy*")).pop() / '_lib'
        os.makedirs(libsaw_prefix, exist_ok=True)
        envlib = PYENV_PREFIX / 'lib' / SYSLIB[platform.system()]
        os.symlink(envlib, libsaw_prefix / envlib.name)
        os.chdir(PARENT)
        subprocess.check_call(
          ["git", "submodule", "deinit", "-f", "submodules/jigsaw-python"])

    def _check_gcc_version(self):
        cpp = shutil.which("c++")
        major, minor, patch = subprocess.check_output(
            [cpp, "--version"]
            ).decode('utf-8').split('\n')[0].split()[-1].split('.')
        current_version = float(f"{major}.{minor}")
        if current_version < 7.:
            raise Exception(
                'JIGSAW requires GCC version 7 or later, got '
                f'{major}.{minor}.{patch} from {cpp}')
        return shutil.which("gcc"), cpp


conf = setuptools.config.read_configuration(PARENT / 'setup.cfg')
meta = conf['metadata']
setuptools.setup(
    name=meta['name'],
    version=Version.from_any_vcs().serialize(),
    author=meta['author'],
    author_email=meta['author_email'],
    description=meta['description'],
    long_description=meta['long_description'],
    long_description_content_type="text/markdown",
    url=meta['url'],
    packages=setuptools.find_packages(),
    cmdclass={
        'install_jigsaw': InstallJigsawCommand,
        },
    python_requires='>=3.7, <3.10',
    setup_requires=['wheel', 'numpy'],
    install_requires=[
                      "colored-traceback",
                      "fiona",
                      "geoalchemy2",
                      "geopandas",
                      "jigsawpy",
                      "matplotlib",
                      "netCDF4",
                      "numba",
                      "numpy>=1.21", # introduce npt.NDArray
                      "pyarrow",
                      "pygeos",
                      "pyproj>=3.0",
                      "rasterio",
                      "requests",
                      "scipy<1.8",   # dropping python 3.7
                      "shapely>=1.8",
                      "tqdm",
                      "typing_extensions",
                      "utm",
                      ],
    entry_points={
        'console_scripts': [
            "ocsmesh=ocsmesh.__main__:main",
            "interp=ocsmesh.interp:main"
        ]
    },
    extras_require={
        'testing': ['pylint>=2.14'],
        'documentation': [
            'sphinx',
            'sphinx-rtd-theme',
            'sphinx-argparse',
            'dunamai',
            'mistune==0.8.4',
            'm2r2',
            'numpydoc'
        ]
    },
    tests_require=['nose'],
    test_suite='nose.collector',
)
