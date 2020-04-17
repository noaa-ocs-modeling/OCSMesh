#! /usr/bin/env python
import setuptools
import subprocess
import setuptools.command.build_py
import distutils.cmd
import distutils.util
import shutil
from pathlib import Path
import sys
import os

PARENT = Path(__file__).parent.absolute()
PYENV_PREFIX = "/".join(sys.executable.split('/')[:-2])


class BootstrapJigsawCommand(distutils.cmd.Command):
    """Custom build command."""

    user_options = []

    def initialize_options(self): pass

    def finalize_options(self): pass

    def run(self):
        # init jigsaw-python submodule
        subprocess.check_call(
            ["git", "submodule", "update",
             "--init", "submodules/jigsaw-python"])
        # install jigsawpy
        os.chdir(PARENT / 'submodules/jigsaw-python')
        subprocess.check_call(["python", "setup.py", "install"])
        # install jigsaw
        os.chdir("external/jigsaw")
        os.makedirs("build", exist_ok=True)
        os.chdir("build")
        subprocess.check_call(
            ["cmake", "..",
             "-DCMAKE_BUILD_TYPE=Release",
             f"-DCMAKE_INSTALL_PREFIX={PYENV_PREFIX}",
             ])
        subprocess.check_call(["make", "install"])
        # push the shell to the parent dir
        os.chdir(PARENT)


class BootstrapJigsawPreviousVersionCommand(distutils.cmd.Command):
    """Custom build command."""

    user_options = []

    def initialize_options(self): pass

    def finalize_options(self): pass

    def run(self):
        # init jigsaw-python submodule
        tgt = PARENT / 'submodules/jigsaw-geo-python'
        if tgt.is_dir():
            shutil.rmtree(tgt)
        subprocess.check_call(
            ["git",
             "clone",
             "--single-branch",
             '--branch',
             'certify_values_fix',
             'https://github.com/jreniel/jigsaw-geo-python',
             'submodules/jigsaw-geo-python'])
        # install jigsawpy
        os.chdir(PARENT / 'submodules/jigsaw-geo-python')
        subprocess.check_call(["python", "setup.py", "install"])
        # install jigsaw
        os.chdir("_ext_/jigsaw")
        os.makedirs("build", exist_ok=True)
        os.chdir("build")
        subprocess.check_call(
            ["cmake", "..",
             "-DCMAKE_BUILD_TYPE=Release",
             f"-DCMAKE_INSTALL_PREFIX={PYENV_PREFIX}",
             ])
        subprocess.check_call(["make", "install"])
        # push the shell to the parent dir
        os.chdir(PARENT)


conf = setuptools.config.read_configuration(PARENT / 'setup.cfg')
meta = conf['metadata']
setuptools.setup(
    name=meta['name'],
    version=meta['version'],
    author=meta['author'],
    author_email=meta['author_email'],
    description=meta['description'],
    long_description=meta['long_description'],
    long_description_content_type="text/markdown",
    url=meta['url'],
    packages=setuptools.find_packages(),
    cmdclass={
        # 'bootstrap_jigsaw': BootstrapJigsawCommand,
        'bootstrap_jigsaw': BootstrapJigsawPreviousVersionCommand
        },
    python_requires='==3.8',
    setup_requires=['wheel', 'numpy'],
    install_requires=[
                      "jigsawpy",
                      "matplotlib",
                      "netCDF4",
                      "scipy",
                      "pyproj",
                      "fiona",
                      "rasterio",
                      "jsmin",
                      # "pysheds",
                      "colored_traceback",
                      "requests",
                      "shapely",
                      ],
    entry_points={
        'console_scripts': [
            "geomesh=geomesh.__main__:main",
            "levee_interp=geomesh.cmd.levee_interp:main"
        ]
    },
    tests_require=['nose'],
    test_suite='nose.collector',
)
