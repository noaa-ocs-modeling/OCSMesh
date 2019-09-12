#! /usr/bin/env python
import setuptools
import subprocess
import setuptools.command.build_py
import distutils.cmd
import distutils.util
from urllib import request
import hashlib
import io
import shutil
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
        self.work_dir = Path(__file__).parent.absolute()
        self.pyenv_prefix = "/".join(sys.executable.split('/')[:-2])

    def finalize_options(self):
        """Post-process options."""
        self.include_gdal = distutils.util.strtobool(self.include_gdal)
        self.work_dir = str(Path(self.work_dir))
        self.pyenv_prefix = Path(self.pyenv_prefix)

    def run(self):
        # self._install_topex()
        self._install_jigsawpy()
        self._install_jigsaw()
        if self.include_gdal:
            self._install_proj()
            self._install_gdal()

    def _setup_step(f):
        def decorator(self):
            os.chdir(str(Path(str(self.work_dir) + "/third_party")))
            f(self)
            os.chdir(self.work_dir)
        return decorator

    @_setup_step
    def _install_topex(self):
        def query_yes_no(question, default="yes"):
            """Ask a yes/no question via raw_input() and return their answer.
            "question" is a string that is presented to the user.
            "default" is the presumed answer if the user just hits <Enter>.
                It must be "yes" (the default), "no" or None (meaning
                an answer is required of the user).

            The "answer" return value is one of "yes" or "no".
            """
            valid = {"yes": True,   "y": True,  "ye": True,
                     "no": False,     "n": False}
            if default is None:
                prompt = " [y/n] "
            elif default == "yes":
                prompt = " [Y/n] "
            elif default == "no":
                prompt = " [y/N] "
            else:
                raise ValueError("invalid default answer: '%s'" % default)

            while 1:
                sys.stdout.write(question + prompt)
                choice = input().lower()
                if default is not None and choice == '':
                    return default
                elif choice in valid.keys():
                    return valid[choice]
                else:
                    sys.stdout.write("Please respond with 'yes' or 'no' "
                                     + "(or 'y' or 'n').\n")
        print('Checking the SRTM15+V2 bathymetry file.\n')
        if os.getenv('SRTM15_PATH') is not None:
            dataset = Path('SRTM15_PATH')
        else:
            dataset = Path(str(self.pyenv_prefix) + "/lib/SRTM15+V2.nc")
        if dataset.is_file():
            md5 = hashlib.md5()
            with io.open(dataset, mode="rb") as fd:
                for chunk in iter(
                        lambda: fd.read(io.DEFAULT_BUFFER_SIZE), b''):
                    md5.update(chunk)
            if md5.hexdigest() != '3c6c3793db338da30cf7d8f3842e87e3':
                print('{} checksum failed. File might be corrupt.'.format(
                    str(dataset)))
                sys.exit(1)
            else:
                print('SRT15+V2.nc checksum passed.\n')
                return
        print('****** NOTICE FOR USAGE OF SRTM15+V2 BATHYMETRY DATASET ******')
        print(request.urlopen(
            "ftp://topex.ucsd.edu/pub/srtm30_plus/README_PERMISSIONS.txt")
            .read(20000).decode('utf8'), end='')
        print("For more information see: https://topex.ucsd.edu/sandwell/"
              + "publications/180_Tozer_SRTM15+.pdf\n")
        print('******                  END NOTICE                      ******')
        url = "ftp://topex.ucsd.edu/pub/srtm15_plus/SRTM15+V2.nc"
        msg = '\nThis software uses the SRTM15+V2 dataset. '
        msg += 'This setup can dowload the SRTM15+V2.nc file automatically '
        msg += 'from the internet by answering yes below. '
        msg += 'You may also cancel this setup and provide '
        msg += 'the path to the SRTM15+V2.nc file using the SRTM15_PATH '
        msg += 'environment variable. '
        msg += 'Alternatively, you may also manually download the file from '
        msg += '{} and place it in the {}/lib directory.\n'.format(
            url, self.pyenv_prefix)
        msg += 'The SRTM15+V2 will take 2.94GB of space. \n'
        msg += 'Would you like setup to download and save the SRTM15+V2.nc '
        msg += 'file to your computer?\n'
        a = query_yes_no(msg)
        if a is False:
            print('No SRTM15+V2 file found. Aborting setup...')
            sys.exit(1)
        subprocess.check_call(
            ["wget", "--directory-prefix={}".format(
                Path(str(self.pyenv_prefix) + "/lib")), url])

    @_setup_step
    def _install_jigsawpy(self):
        subprocess.check_call(
            ["git", "submodule", "update", "--init", "jigsawpy"])
        os.chdir(str(Path("jigsawpy")))
        subprocess.check_call(["python", "setup.py", "install"])

    @_setup_step
    def _install_jigsaw(self):
        os.chdir(str(Path("jigsawpy/_ext_/jigsaw")))
        os.makedirs("build", exist_ok=True)
        os.chdir(str(Path("build")))
        subprocess.check_call(
            ["cmake", "..",
             "-DCMAKE_BUILD_TYPE=Release",
             "-DCMAKE_INSTALL_PREFIX={}".format(self.pyenv_prefix)])
        subprocess.check_call(["make", "install"])
        libsaw_prefix = str(
            list(self.pyenv_prefix.glob("**/*jigsawpy")).pop()) + '/_lib'
        os.makedirs(libsaw_prefix, exist_ok=True)
        for libsaw in self.pyenv_prefix.glob("lib/*jigsaw*"):
            shutil.copy(libsaw, libsaw_prefix)
        os.chdir(self.work_dir + '/third_party')
        subprocess.check_call(["git", "submodule", "deinit", "-f", "jigsawpy"])

    @_setup_step
    def _install_proj(self):
        subprocess.check_call(
            ["git", "submodule", "update", "--init", "PROJ"])
        os.chdir("PROJ")
        subprocess.check_call("./autogen.sh")
        subprocess.check_call(
            ["./configure", "--prefix={}".format(self.pyenv_prefix)])
        subprocess.check_call(["make", "install"])
        os.chdir(self.work_dir + '/third_party')
        subprocess.check_call(["git", "submodule", "deinit", "-f", "PROJ"])

    @_setup_step
    def _install_gdal(self):
        subprocess.check_call(
            ["git", "submodule", "update", "--init", "gdal"])
        os.chdir(str(Path("gdal/gdal")))
        subprocess.check_call("./autogen.sh")
        subprocess.check_call(
            ["./configure", "--prefix={}".format(self.pyenv_prefix)])
        subprocess.check_call(["make", "install"])
        os.chdir(self.work_dir + '/third_party')
        subprocess.check_call(["git", "submodule", "deinit", "-f", "gdal"])


try:
    gdal = "gdal=={}".format(subprocess.check_output(
        ["gdal-config", "--version"]).decode('utf8').strip('\n'))
except FileNotFoundError:
    if "--include-gdal=True" not in sys.argv:
        msg = 'GDAL was not found in the system.\n'
        msg += 'Run `setup.py install_deps --include-gdal=True` to build '
        msg += 'GDAL or alternatively install GDAL using your system\'s '
        msg += 'package manager.'
        raise Exception(msg)
    else:
        gdal = "gdal"

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
    cmdclass={'install_deps': InstallDepsCommand},
    python_requires='>=3.7',
    setup_requires=['wheel', 'numpy'],
    install_requires=[
                      "jigsawpy",
                      "matplotlib",
                      "netCDF4",
                      "scipy",
                      gdal
                      ],
    )
