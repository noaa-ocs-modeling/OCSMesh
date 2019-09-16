# geomesh

## Requirements
CMake>=<br/>
Python 3.7 (and development headers)<br/>
GDAL (and development headers).<br/>
This package is intended to run on Linux.<br/>
It is compatible with the Windows Subsystem for Linux.<br/>
MacOSX is not supported. If you wish to test the softare on MacOSX, use the [Vagrant](#vagrant) installation.


## Installation instructions
```bash
./install.sh
```

## Vagrant
```sh
sh -c "$(curl -fsSL https://gist.github.com/jreniel/397df26f8b0c4aa71ea18e4a6baa012c/raw)" && cd geomesh && vagrant ssh
```
