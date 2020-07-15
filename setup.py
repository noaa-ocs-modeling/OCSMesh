#! /usr/bin/env python
import setuptools
import subprocess
import setuptools.command.build_py
import distutils.cmd
import distutils.util
import shutil
from pathlib import Path
from multiprocessing import cpu_count
import sys
import os

CMAKE_OPTS = []
for i, argv in reversed(list(enumerate(sys.argv))):
    if "-DCMAKE" in argv:
        CMAKE_OPTS.append(argv)
        sys.argv.pop(i)


class InstallJigsawCommand(distutils.cmd.Command):
    """Custom build command."""

    user_options = []

    def initialize_options(self):
        """Set default values for options."""
        self.work_dir = Path(__file__).parent.absolute()
        self.pyenv_prefix = "/".join(sys.executable.split('/')[:-2])

    def finalize_options(self):
        """Post-process options."""
        self.work_dir = str(Path(self.work_dir))
        self.pyenv_prefix = Path(self.pyenv_prefix)

    def run(self):
        # need to install jigsawpy first, then jigsaw so that libjigsaw.so
        # ends up on the right path.
        self._install_jigsawpy()
        self._install_jigsaw()

    def _setup_step(f):
        def decorator(self):
            os.chdir(str(Path(str(self.work_dir) + "/contrib")))
            f(self)
            os.chdir(self.work_dir)
        return decorator

    @_setup_step
    def _install_jigsaw(self):
        os.chdir(str(Path("jigsawpy/_ext_/jigsaw")))
        os.makedirs("build", exist_ok=True)
        os.chdir(str(Path("build")))
        cmake_cmd = [
            "cmake",
            "..",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={self.pyenv_prefix}",
            ]
        cmake_cmd.extend(CMAKE_OPTS)
        subprocess.check_call(cmake_cmd)
        subprocess.check_call(["make", f"-j{cpu_count()}", "install"])
        libsaw_prefix = str(
            list(self.pyenv_prefix.glob("**/*jigsawpy")).pop()) + '/_lib'
        os.makedirs(libsaw_prefix, exist_ok=True)
        for libsaw in self.pyenv_prefix.glob("lib/*jigsaw*"):
            shutil.copy(libsaw, libsaw_prefix)
        os.chdir(self.work_dir + '/contrib')
        subprocess.check_call(["git", "submodule", "deinit", "-f", "jigsawpy"])

    @_setup_step
    def _install_jigsawpy(self):
        subprocess.check_call(
            ["git", "submodule", "update", "--init", "jigsawpy"])
        os.chdir(str(Path("jigsawpy")))
        subprocess.check_call(["python", "setup.py", "install"])


conf = setuptools.config.read_configuration(
    str(Path(__file__).parent.absolute()) + '/setup.cfg')
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
    cmdclass={'install_jigsaw': InstallJigsawCommand},
    python_requires='>=3.7',
    setup_requires=['wheel', 'numpy'],
    install_requires=[
                      "jigsawpy",
                      "matplotlib",
                      "netCDF4",
                      "scipy",
                      "pyproj",
                      "fiona",
                      "rasterio",
                      # "pysheds",
                      "shapely",
                      ],
    )
