# geomesh

## Requirements
A working C/CXX compiler <br/>
Python 3.7 (and development headers)<br/>
[GDAL (and development headers)](#notes-on-gdal)

### Notes on GDAL
It is recommended to use the system's package manager to satisfy this dependency, however, one can optionally pass the `--include-gdal` flag to the installation script, and it will build/install GDAL for you. This flag can also be used to compile a local version of GDAL that overrides the system's version inside the geomesh Python environment.<br/>

### Notes on operating system requirements
This package is intended to run on Linux.<br/>
It is compatible with the Windows Subsystem for Linux.<br/>
MacOSX is not supported. If you wish to test the software on MacOSX, use the [Vagrant](#vagrant) installation.

### Virtual memory requirements
Processing DEM datasets can be a memory expensive operation. The amount of memory required will depend on the resolution and quantity of the input datasets. It is recommended to use [systemd-swap](https://github.com/Nefelim4ag/systemd-swap) or create a large swap partition when processing large or very high resolution datasets if the physical memory is running out.


## Installation instructions
To install, just run
```bash
./install.sh
```
You may optionally pass the `--include-gdal` flag to include the installation of gdal to the Python environment. To complete the installation, re-source or restart the shell.

### Python environment
The installation script will create a Python environment in the project's directory called `.geomesh_env`. The setup script will also create an alias on the ~/.bashrc or ~/.zshrc files as follows:
```bash
alias geomesh="source <install_dir>/.geomesh_env/bin/activate"
```
To source the geomesh Python environment, simply run the command:
```bash
geomesh
```

### Notes for developers
If you plan to do development on the code, after running the normal installation you should execute
```bash
./setup.py develop
```
This will allow you to make changes to the source code and be able to test them immediately without having to rerun the installer.

## Vagrant
```sh
sh -c "$(curl -fsSL https://gist.github.com/jreniel/397df26f8b0c4aa71ea18e4a6baa012c/raw)"
cd geomesh
vagrant ssh
```
