import numpy as np
import jigsawpy
import geomesh


class Jigsaw:

    def __init__(self, geom, hfun=None, initial_mesh=None):
        self._geom = geom
        self._hfun = hfun
        self._initial_mesh = initial_mesh

    def run(self):

        # set jigsaw hfun scaling
        hfun = self.hfun
        if hfun is not None:
            self.opts.hfun_scal = hfun.scaling
            self.opts.hfun_hmin = hfun.hmin
            self.opts.hfun_hmax = hfun.hmax
            hfun = hfun.hfun
        else:
            self.opts.hfun_scal = "absolute"

        # set initial mesh as init or None
        init = self.initial_mesh
        if init is not None:
            init = init.mesh

        # set jigsaw mesh dims
        self.opts.mesh_dims = self.geom.ndim

        # set additional jigsaw options
        # self.opts.mesh_top1 = True  # locally 1-manifold
        # self.opts.geom_feat = True  # for sharp feat's

        # call jigsaw
        self.jigsaw(
            opts=self.opts,
            geom=self.geom.geom,  # geom/pslg
            mesh=self.output_mesh,  # output mesh
            init=init,  # initial mesh
            hfun=hfun   # hfun
        )
        from matplotlib.pyplot import plt
        plt.triplot(
            self.output_mesh.vert2['coord'][:, 0],
            self.output_mesh.vert2['coord'][:, 1],
            self.output_mesh.tria3['index'])
        plt.show(block=False)
        breakpoint()
        return geomesh.TriMesh(
            self.output_mesh.vert2['coord'],
            self.output_mesh.tria3['index'],
            crs=self.pslg.crs)

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

    @_geom.setter
    def _geom(self, geom):
        assert isinstance(geom, geomesh.PlanarStraightLineGraph)
        self.__geom = geom

    @_hfun.setter
    def _hfun(self, hfun):
        if hfun is not None:
            assert isinstance(hfun, geomesh.SizeFunction)
        self.__hfun = hfun

    @_initial_mesh.setter
    def _initial_mesh(self, initial_mesh):
        if initial_mesh is not None:
            assert isinstance(initial_mesh, geomesh.Mesh)
        self.__initial_mesh = initial_mesh






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
