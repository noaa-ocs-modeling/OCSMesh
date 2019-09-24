import numpy as np
import jigsawpy
import geomesh


class Jigsaw:

    def __init__(self, Geom, Hfun=None, Mesh=None):
        self._Geom = Geom
        self._Hfun = Hfun
        self._Mesh = Mesh

    def run(self):
        self.jigsaw(self._opts, self._geom, self._output_mesh,
                    self._input_mesh, self._hfun)
        return geomesh.UnstructuredMesh(
            self._output_mesh.vert2['coord'],
            self._output_mesh.tria3['index'],
            SpatialReference=self.Geom.SpatialReference)

    @property
    def Geom(self):
        return self._Geom

    @property
    def Hfun(self):
        return self._Hfun

    @property
    def Mesh(self):
        return self._Mesh

    @property
    def jigsaw(self):
        return jigsawpy.lib.jigsaw

    @property
    def verbosity(self):
        return self._opts.verbosity

    @property
    def _opts(self):
        try:
            return self.__opts
        except AttributeError:
            self._opts = jigsawpy.jigsaw_jig_t()
            return self.__opts

    @property
    def _geom(self):
        return self._Geom.geom

    @property
    def _input_mesh(self):
        if self.Mesh is not None:
            raise NotImplementedError
            return self.Mesh._msh

    @property
    def _hfun(self):
        if self.Hfun is not None:
            return self.Hfun.hfun

    @property
    def _output_mesh(self):
        try:
            return self.__output_mesh
        except AttributeError:
            self.__output_mesh = jigsawpy.jigsaw_msh_t()
            return self.__output_mesh

    @property
    def _Geom(self):
        return self.__Geom

    @property
    def _Hfun(self):
        return self.__Hfun

    @property
    def _Mesh(self):
        return self.__Mesh

    @verbosity.setter
    def verbosity(self, verbosity):
        self._opts.verbosity = verbosity

    @_Geom.setter
    def _Geom(self, Geom):
        assert isinstance(Geom, (geomesh.PlanarStraightLineGraph, ))
        self.__Geom = Geom

    @_Hfun.setter
    def _Hfun(self, Hfun):
        assert isinstance(Hfun, (geomesh.SizeFunction, type(None)))
        self.__Hfun = Hfun

    @_Mesh.setter
    def _Mesh(self, Mesh):
        assert isinstance(Mesh, (geomesh.UnstructuredMesh, type(None)))
        self.__Mesh = Mesh

    @_opts.setter
    def _opts(self, jigsaw_jig_t):
        jigsaw_jig_t.mesh_dims = self.Geom._ndim
        if self.Hfun is not None:
            jigsaw_jig_t.hfun_scal = self.Hfun.scaling
            # jigsaw_jig_t.hfun_hmin = np.min(self.Hfun.values)
            jigsaw_jig_t.hfun_hmax = np.max(self.Hfun.values)
        else:
            jigsaw_jig_t.hfun_scal = "absolute"
        self.__opts = jigsaw_jig_t
