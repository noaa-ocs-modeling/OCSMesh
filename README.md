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

## What is Vagrant:
Vagrant is a way of deploying the app across as long as Vagrant is properly installed.
To use the Vagrant method, you must have [Vagrant](https://www.vagrantup.com/downloads.html) and [VirtualBox](https://www.virtualbox.org/wiki/Downloads) installed in your Operating System. You must also have an XServer installed on the Host machine, that is XMing (Windows) or XQuartz(MacOSX).

Once you have met the Vagrant dependencies, `cd` into the cloned directory and run:

```bash
vagrant up --provision && vagrant ssh
```
Once this is finished, you will be on a Vagrant developer shell. You may edit the source code in the host machine and rerun/debug the code using the vagrant shell.


## External data sources:
If you are running from Vagrant skip this part and go directly to run the examples.
If you installed natively you must run at least once:
```bash
git lfs install  # example data is a git-lfs repo
git submodule update --init examples/data  # example data is provided as submodule
```

## Running the examples:
To run the examples you may execute:
```bash
examples/example_1.py
```
and
```bash
examples/example_2.py
```
The example_3.py requires external files and is not supported in the Vagrant demo.
