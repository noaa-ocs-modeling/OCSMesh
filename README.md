# geomesh

## Requirements
CMake>= <br/>
Python 3.7 (and development headers)<br/>
GDAL (and development headers).<br/>
This package has been tested on the Windows Subsystem for Linux (Ubuntu) and Arch Linux. There is not Mac support at this time. See the Vagrant section below for a simple method for running this software on a mac computer.


## Installation instructions
This software is written in Python 3.7 and it should be installable on most systems. It has been tested on Ubuntu, Arch Linux and on the Windows Subsystem for Linux (Ubuntu version). MacOSX is currently not supported. See the [Running on Vagrant](#running-on-vagrant)

Compile dependencies and install:
```bash
./setup.py install_deps # --include-gdal=True 
./setup.py install  #   ./setup.py develop # if you are a developer
```
You may use the --include-gdal=True to compile GDAL if it's not available on the OS.


## Using the software through Vagrant:
The Vagrant configuration is provided here as a convenience for the users who might be having issues installing or running the software, but that still would like to run tests. You must have [Vagrant](https://www.vagrantup.com/), [VirtualBox](https://www.virtualbox.org/) and [git-lfs](https://git-lfs.github.com/) installed on your machine. These are the only dependencies to meet, as the vagrant image will take care of the rest. Once you have met these dependencies, simply execute to following command to bootstrap the geomesh project into an interactive vagrant box shell:
```sh
sh -c "$(curl -fsSL https://gist.github.com/jreniel/397df26f8b0c4aa71ea18e4a6baa012c/raw)"
```
Once the vagrant box has been created, simply `cd` into the downloaded geomesh directory and execute:
```sh
vagrant ssh
```
Now you can modify the source in real time using your OS editor, while running the geomesh inside the Vagrant shell.
