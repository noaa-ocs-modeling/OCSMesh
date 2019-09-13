# geomesh

## Requirements:
Python 3.7

GDAL

## Installation on Linux
Compile dependencies and install:
```bash
./setup.py install_deps # --include-gdal=True 
./setup.py install  #   ./setup.py develop # if you are a developer
```
You may use the --include-gdal=True to compile GDAL if it's not available on the OS.

## Windows installation:
Install the Windows Subsystem for Linux using any distro of your choice. Follow the Linux installation above.
Alternatively you can use Vagrant. See instructions below.

## MacOSX installation
Only through Vagrant. See instructions below.

## Using the software through Vagrant:
Vagrant is a way of deploying apps across all major platforms as long as Vagrant is installed on the OS.
The Vagrantfile on this project is configured to provide an enviroment where you can test and develop this software. It is recommended to use a native installation, but this is provided as an alternative so that the software can be accessible accross most platforms.
To use the Vagrant method, you must have [Vagrant](https://www.vagrantup.com/downloads.html) and [VirtualBox](https://www.virtualbox.org/wiki/Downloads) installed in your Operating System. You must also have an XServer installed on the Host machine (XMing on Windows or XQuartz on MacOSX).

Before bringing Vagrant up, the jigsawpy submodule has to be manually initialized or the vagrant provisionsing will fail. You only need to do this step once in your repo.

```bash
git submodule update --init third_party/jigsawpy
```

You should also initialize the example data as described in [Setting up the example data](#setting-up-the-example-data) section.

Once you have met these Vagrant dependencies, you may initialize the vagrant box and ssh into it:

```bash
vagrant up --provision
```
Once this is finished, you can ssh into the vagrant developer shell by running:

```bash
vagrant ssh
```

The source code on the host machine and on the vagrant shell is in real-time sync. This means you can do file editing on the host machine and run the program in the vagrant shell in order to do development/debbugging.

## Setting up the example data
In order to bring to bring up the example data you must run:
```bash
git lfs install  # example data is a git-lfs repo
git submodule update --init examples/data  # example data is provided as submodule
```
If you are using Vagrant, you must run this steps in the host machine (not in the vagrant shell).

## Running the examples
To run the examples you may execute:
```bash
examples/example_1.py
```
and
```bash
examples/example_2.py
```
The example_3.py requires external files. See [example_3.py](examples/example_3.py) header comments for a link to the files. Example 3 can take more than 7 hours to run and is computationally expensive. It is provided here as a demos of the capability to bulk process DEM's into meshes.
