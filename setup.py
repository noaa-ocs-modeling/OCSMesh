#! /usr/bin/env python
import setuptools
import subprocess
import setuptools.command.build_py
import distutils.cmd
import distutils.util
from pathlib import Path
import sys
import os


class InstallDepsCommand(distutils.cmd.Command):
    """Custom build command."""

    user_options = [  # The format is (long option, short option, description).
                    ('include-gdal=', None, 'Build and install gdal.')]

    def initialize_options(self):
        """Set default values for options."""
        self.include_gdal = 'False'
        self.work_dir = os.getcwd()
        self.cmake_prefix = "/".join(sys.executable.split('/')[:-2])

    def finalize_options(self):
        """Post-process options."""
        self.include_gdal = distutils.util.strtobool(self.include_gdal)
        self.work_dir = str(Path(self.work_dir))
        self.cmake_prefix = str(Path(self.cmake_prefix))

    def run(self):
        self._install_pymesh()
        self._install_jigsaw()
        self._install_jigsawpy()
        if self.include_gdal:
            self._install_proj()
            self._install_gdal()

    def _setup_step(f):
        def decorator(self):
            os.chdir(str(Path("third_party")))
            f(self)
            os.chdir(self.work_dir)
        return decorator

    @_setup_step
    def _install_pymesh(self):
        subprocess.check_call(["git", "submodule", "update", "--init",
                               "--recursive", "PyMesh"])
        os.chdir("PyMesh")
        subprocess.check_call(["./setup.py", "build"])
        subprocess.check_call(["./setup.py", "install"])
        subprocess.check_call(["git", "submodule", "deinit", "-f", "PyMesh"])

    @_setup_step
    def _install_jigsaw(self):
        subprocess.check_call(
            ["git", "submodule", "update", "--init", "jigsawpy"])
        os.chdir(str(Path("jigsawpy/_ext_/jigsaw")))
        os.makedirs("build", exist_ok=True)
        os.chdir(str(Path("build")))
        subprocess.check_call(
            ["cmake", "..",
             "-DCMAKE_BUILD_TYPE=Release",
             "-DCMAKE_INSTALL_PREFIX={}".format(self.cmake_prefix)])
        subprocess.check_call(["make", "install"])
        subprocess.check_call(["git", "submodule", "deinit", "-f", "jigsawpy"])

    @_setup_step
    def _install_proj(self):
        subprocess.check_call(
            ["git", "submodule", "update", "--init", "PROJ"])
        os.chdir("PROJ")
        subprocess.check_call("./autogen.sh")
        subprocess.check_call(
            ["./configure", "--prefix={}".format(self.cmake_prefix)])
        subprocess.check_call(["make", "install"])
        subprocess.check_call(["git", "submodule", "deinit", "-f", "PROJ"])

    @_setup_step
    def _install_gdal(self):
        subprocess.check_call(
            ["git", "submodule", "update", "--init", "gdal"])
        os.chdir(str(Path("gdal/gdal")))
        subprocess.check_call("./autogen.sh")
        subprocess.check_call(
            ["./configure", "--prefix={}".format(self.cmake_prefix)])
        subprocess.check_call(["make", "install"])
        os.chdir(str(Path("../..")))
        subprocess.check_call(["git", "submodule", "deinit", "-f", "gdal"])

    @_setup_step
    def _install_jigsawpy(self):
        os.chdir(str(Path("jigsawpy")))
        subprocess.check_call(["python", "setup.py", "install"])


conf = setuptools.config.read_configuration('./setup.cfg')
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
    cmdclass={'install_deps': InstallDepsCommand},
    python_requires='>3.6',
    install_requires=["jigsawpy",
                      "matplotlib",
                      "netCDF4",
                      "gdal",
                      "scipy"],
    )
