FROM archlinux/base:latest

RUN pacman -Sy --noconfirm \
base-devel \
git \
python \
python-setuptools \
cmake

# RUN git clone --branch dev --recurse-submodules https://github.com/jreniel/geomesh
RUN mkdir geomesh
WORKDIR geomesh
COPY . .
# RUN git submodule update --init --recursive
# RUN ./setup.py build
RUN ./setup.py install_deps --include-gdal=True
RUN ./setup.py install
# RUN python -c ""
# RUN pip install -e .

# RUN svn co --username anonymous http://columbia.vims.edu/schism/tags/v5.7.0
# # build metis
# WORKDIR v5.7.0/src/ParMetis-3.1-Sep2010
# RUN export CC=mpicc; \
# export FC=mpif90; \
# make
# WORKDIR /


# RUN mkdir v5.7.0/src/build
# WORKDIR v5.7.0/src/build
# RUN cmake .. \
# -DNetCDF_Fortran_LIBRARY=/lib/libnetcdff.so \
# -DNetCDF_C_LIBRARY=/lib/libnetcdf.so \
# -DNetCDF_LIBRARY_DIR=/lib \
# -DNetCDF_INCLUDE_DIR=/usr/include \
# -DTVD_LIM=SB \
# -DMETIS_LIBRARY=../ParMetis-3.1-Sep2010/libmetis.a \
# -DPARMETIS_LIBRARY=../ParMetis-3.1-Sep2010/libparmetis.a
# RUN make
