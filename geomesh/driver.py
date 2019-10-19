import numpy as np
import jigsawpy
import geomesh


class Jigsaw:

    def __init__(self, geom, initial_mesh=None):
        """
        geom can be SizeFunction or PlanarStraightLineGraph instance.
        """
        self._geom = geom
        self._initial_mesh = initial_mesh

    def run(self):
        self.jigsaw(
            self.opts,
            self.geom,
            self.output_mesh,
            self.initial_mesh,
            self.hfun
        )
        # raise if empty mesh is returned
        msg = 'ERROR: Jigsaw returned empty mesh.'
        assert self.output_mesh.tria3['index'].shape[0] > 0, msg
        return geomesh.Mesh(
            self.output_mesh.vert2['coord'],
            self.output_mesh.tria3['index'],
            crs=self.crs)

    @property
    def geom(self):
        return self._geom

    @property
    def hfun(self):
        return self._hfun

    @property
    def initial_mesh(self):
        return self._initial_mesh

    @property
    def output_mesh(self):
        return self._output_mesh

    @property
    def jigsaw_jig_t(self):
        return self._jigsaw_jig_t

    @property
    def jigsaw(self):
        return jigsawpy.lib.jigsaw

    @property
    def opts(self):
        return self.jigsaw_jig_t

    @property
    def verbosity(self):
        return self.opts.verbosity

    @property
    def hfun_hmin(self):
        return self.opts.hfun_hmin

    @property
    def hfun_hmax(self):
        return self.opts.hfun_hmax

    @property
    def hfun_scal(self):
        return self.opts.hfun_scal

    @property
    def crs(self):
        return self._crs

    @property
    def _jigsaw_jig_t(self):
        try:
            return self.__jigsaw_jig_t
        except AttributeError:
            self.__jigsaw_jig_t = jigsawpy.jigsaw_jig_t()
            return self.__jigsaw_jig_t

    @property
    def _geom(self):
        return self.__geom

    @property
    def _hfun(self):
        return self.__hfun

    @property
    def _initial_mesh(self):
        return self.__initial_mesh

    @property
    def _crs(self):
        return self.__crs

    @property
    def _output_mesh(self):
        try:
            return self.__output_mesh
        except AttributeError:
            self.__output_mesh = jigsawpy.jigsaw_msh_t()
            return self.__output_mesh

    @verbosity.setter
    def verbosity(self, verbosity):
        assert verbosity in [0, 1, 2, 3]
        self.opts.verbosity = verbosity

    @hfun_hmin.setter
    def hfun_hmin(self, hfun_hmin):
        self.opts.hfun_hmin = float(hfun_hmin)

    @hfun_hmax.setter
    def hfun_hmax(self, hfun_hmax):
        self.opts.hfun_hmax = float(hfun_hmax)

    @hfun_scal.setter
    def hfun_scal(self, hfun_scal):
        assert hfun_scal in ["absolute", "relative"]
        self.opts.hfun_scal = hfun_scal

    @_geom.setter
    def _geom(self, geom):
        if isinstance(geom, geomesh.SizeFunction):
            self._hfun = geom
            geom = geom.geom
        elif isinstance(geom, geomesh.PlanarStraightLineGraph):
            self._hfun = None
        self.opts.mesh_dims = geom.ndim
        self._crs = geom.crs
        self.__geom = geom.geom

    @_hfun.setter
    def _hfun(self, hfun):
        if hfun is not None:
            assert isinstance(hfun, geomesh.SizeFunction)
            # set scaling
            self.hfun_scal = hfun.scaling
            # use hmin limits
            if hfun.hmin_is_absolute_limit is True:
                self.hfun_hmin = hfun.hmin
            elif hfun.hmin_is_absolute_limit is False:
                self.hfun_hmin = np.min(hfun.values)
            # set hmax limits
            if hfun.hmax_is_absolute_limit is True:
                self.hfun_hmax = hfun.hmax
            elif hfun.hmax_is_absolute_limit is False:
                self.hfun_hmax = np.max(hfun.values)
            # push jigsaw_msh_t object
            hfun = hfun.hfun
        self.__hfun = hfun

    @_initial_mesh.setter
    def _initial_mesh(self, initial_mesh):
        if initial_mesh is not None:
            assert isinstance(initial_mesh, geomesh.Mesh)
            initial_mesh = initial_mesh.mesh
        self.__initial_mesh = initial_mesh

    @_crs.setter
    def _crs(self, crs):
        self.__crs = crs






    # @property
    # def _input_mesh(self):
    #     if self.Mesh is not None:
    #         raise NotImplementedError
    #         return self.Mesh._msh

    # @property
    # def _hfun(self):
    #     if self.Hfun is not None:
    #         return self.Hfun.hfun

    # @property
    # def _mesh(self):
    #     try:
    #         return self.__output_mesh
    #     except AttributeError:
    #         self.__output_mesh = jigsawpy.jigsaw_msh_t()
    #         return self.__output_mesh

    # @property
    # def _Geom(self):
    #     return self.__Geom

    # @property
    # def _Hfun(self):
    #     return self.__Hfun

    # @property
    # def _Mesh(self):
    #     return self.__Mesh

    # @verbosity.setter
    # def verbosity(self, verbosity):
    #     self._opts.verbosity = verbosity

    # @_Geom.setter
    # def _Geom(self, Geom):
    #     assert isinstance(Geom, (geomesh.PlanarStraightLineGraph, ))
    #     self.__Geom = Geom

    # @_Hfun.setter
    # def _Hfun(self, Hfun):
    #     assert isinstance(Hfun, (geomesh.SizeFunction, type(None)))
    #     self.__Hfun = Hfun

    # @_Mesh.setter
    # def _Mesh(self, Mesh):
    #     assert isinstance(Mesh, (geomesh.UnstructuredMesh, type(None)))
    #     self.__Mesh = Mesh

    # @_opts.setter
    # def _opts(self, jigsaw_jig_t):
    #     jigsaw_jig_t.mesh_dims = self.Geom._ndim
    #     if self.Hfun is not None:
    #         jigsaw_jig_t.hfun_scal = self.Hfun.scaling
    #         if np.min(self.Hfun.values) == np.max(self.Hfun.values):
    #             # For a constant size function is better to use built-in
    #             jigsaw_jig_t.hfun_hmin = 0.
    #             jigsaw_jig_t.hfun_hmax = np.max(self.Hfun.values)
    #             self.__Hfun = None
    #         else:
    #             # jigsaw_jig_t.hfun_hmin = 0.
    #             jigsaw_jig_t.hfun_hmax = np.max(self.Hfun.values)
    #     else:
    #         jigsaw_jig_t.hfun_scal = "absolute"
    #     self.__opts = jigsaw_jig_t
