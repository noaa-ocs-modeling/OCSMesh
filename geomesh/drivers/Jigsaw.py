import jigsawpy
import numpy as np
import geomesh


class Jigsaw(object):

    def __init__(self, Geom, Hfun=None, Mesh=None):
        self._Geom = Geom
        self._Hfun = Hfun
        self._Mesh = Mesh
        self.__init_opts()  # opts must be initialized at init
        self.__init_output_mesh()  # output_mesh must be initialized at init

    def run(self):
        self.jigsaw(self._opts, self._geom, self._output_mesh,
                    self._input_mesh, self._hfun)
        return geomesh.Mesh(self._output_mesh.vert2['coord'],
                            self._output_mesh.tria3['index'],
                            SpatialReference=self.Geom.SpatialReference)

    def __init_opts(self):
        opts = jigsawpy.jigsaw_jig_t()
        opts.mesh_dims = self.Geom.ndim
        opts.verbosity = 0
        if self.Hfun is not None:
            opts.hfun_scal = self.Hfun.hfun_scal
        self.__opts = opts

    def __init_output_mesh(self):
        self.__output_mesh = jigsawpy.jigsaw_msh_t()

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
    def opts(self):
        return self._opts

    @property
    def _Geom(self):
        return self.__Geom

    @property
    def _Hfun(self):
        return self.__Hfun

    @property
    def _Mesh(self):
        return self.__Mesh

    @property
    def _opts(self):
        return self.__opts

    @property
    def _geom(self):
        geom = jigsawpy.jigsaw_msh_t()
        geom.mshID = "euclidean-mesh"
        assert self.Geom.ndim == 2
        vert2 = list()
        for i, (x, y) in enumerate(self.Geom.vert2):
            vert2.append(((x, y), 0))   # why 0?
        geom.vert2 = np.asarray(vert2, dtype=jigsawpy.jigsaw_msh_t.VERT2_t)
        edge2 = list()
        for i, (e0, e1) in enumerate(self.Geom.edge2):
            edge2.append(((e0, e1), 0))   # why 0?
        geom.edge2 = np.asarray(edge2, dtype=jigsawpy.jigsaw_msh_t.EDGE2_t)
        return geom

    @property
    def _input_mesh(self):
        if not hasattr(self, "__input_mesh"):
            if self.Mesh is None:
                input_mesh = None
            else:
                raise NotImplementedError
                input_mesh = jigsawpy.jigsaw_msh_t()
            self.__input_mesh = input_mesh
        return self.__input_mesh

    @property
    def _hfun(self):
        if not hasattr(self, "__hfun"):
            if self.Hfun is None:
                hfun = None
            else:
                raise NotImplementedError
                hfun = jigsawpy.jigsaw_msh_t()
            self.__hfun = hfun
        return self.__hfun

    @property
    def _output_mesh(self):
        return self.__output_mesh

    @_Geom.setter
    def _Geom(self, Geom):
        assert isinstance(Geom, (geomesh.PlanarStraightLineGraph, ))
        self.__Geom = Geom

    @_Hfun.setter
    def _Hfun(self, Hfun):
        if Hfun is not None:
            assert isinstance(Hfun, geomesh.Hfun)
        self.__Hfun = Hfun

    @_Mesh.setter
    def _Mesh(self, Mesh):
        if Mesh is not None:
            assert isinstance(Mesh, geomesh.Mesh)
        self.__Mesh = Mesh
