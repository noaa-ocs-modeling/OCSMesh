import numpy as np
from matplotlib.tri import Triangulation
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
        self._certify_output()
        return geomesh.Mesh(
            self.output_mesh.vert2['coord'],
            self.output_mesh.tria3['index'],
            crs=self.dst_crs)

    def _certify_output(self):
        # raise if empty mesh is returned
        msg = 'ERROR: Jigsaw returned empty mesh.'
        assert self.output_mesh.tria3['index'].shape[0] > 0, msg
        self._remove_isolates()

    def _remove_isolates(self):
        # cleanup isolated nodes
        node_indexes = np.arange(self.output_mesh.vert2['coord'].shape[0])
        used_indexes = np.unique(self.output_mesh.tria3['index'])
        vert2_idxs = np.where(
            np.isin(node_indexes, used_indexes, assume_unique=True))[0]
        tria3_idxs = np.where(
            ~np.isin(node_indexes, used_indexes, assume_unique=True))[0]
        tria3 = self.output_mesh.tria3['index'].flatten()
        for idx in reversed(tria3_idxs):
            _idx = np.where(tria3 >= idx)
            tria3[_idx] = tria3[_idx] - 1
        tria3 = tria3.reshape(self.output_mesh.tria3['index'].shape)
        _mesh = jigsawpy.jigsaw_msh_t()
        _mesh.ndims = 2
        _mesh.vert2 = self.output_mesh.vert2.take(vert2_idxs, axis=0)
        _mesh.tria3 = np.asarray(
            [(tuple(indices), self.output_mesh.tria3['IDtag'][i])
             for i, indices in enumerate(tria3)],
            dtype=jigsawpy.jigsaw_msh_t.TRIA3_t)
        self.__output_mesh = _mesh

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
    def jigsaw(self):
        return jigsawpy.lib.jigsaw

    @property
    def opts(self):
        try:
            return self.__opts
        except AttributeError:
            self.__opts = jigsawpy.jigsaw_jig_t()
            return self.__opts

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
    def optm_qlim(self):
        return self.opts.optm_qlim

    @property
    def mesh_top1(self):
        return self.opts.mesh_top1

    @property
    def geom_feat(self):
        return self.opts.geom_feat

    @property
    def dst_crs(self):
        return self._dst_crs

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
    def _dst_crs(self):
        return self.__dst_crs

    @property
    def _mesh_dims(self):
        return self.__mesh_dims

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

    @optm_qlim.setter
    def optm_qlim(self, optm_qlim):
        optm_qlim = float(optm_qlim)
        assert optm_qlim > 0 and optm_qlim < 1
        self.opts.optm_qlim = optm_qlim

    @mesh_top1.setter
    def mesh_top1(self, mesh_top1):
        assert isinstance(mesh_top1, bool)
        self.opts.mesh_top1 = mesh_top1

    @geom_feat.setter
    def geom_feat(self, geom_feat):
        assert isinstance(geom_feat, bool)
        self.opts.geom_feat = geom_feat

    @_geom.setter
    def _geom(self, geom):
        if isinstance(geom, geomesh.SizeFunction):
            self._hfun = geom
        elif isinstance(geom, geomesh.PlanarStraightLineGraph):
            self._hfun = None
        self._dst_crs = geom.dst_crs
        self._mesh_dims = geom.ndim
        self.__geom = geom.geom

    @_hfun.setter
    def _hfun(self, hfun):
        if hfun is not None:
            assert isinstance(hfun, geomesh.SizeFunction)

            # set scaling
            self.hfun_scal = hfun.scaling

            # use hmin limits
            if hfun.hmin_is_absolute_limit:
                self.hfun_hmin = hfun.hmin
            else:
                self.hfun_hmin = np.min(hfun.values)

            # set hmax limits
            if hfun.hmax_is_absolute_limit:
                self.hfun_hmax = hfun.hmax
            else:
                self.hfun_hmax = np.max(hfun.values)

            # push jigsaw_msh_t object
            hfun = hfun.hfun
        self.__hfun = hfun

    @_initial_mesh.setter
    def _initial_mesh(self, initial_mesh):
        if initial_mesh is not None:
            assert isinstance(initial_mesh, geomesh.Mesh)
            initial_mesh = initial_mesh.mesh
        # Seams between the tiles become noticeable when initial mesh
        # is provided. More testing should be done.
        # elif self.hfun is not None:
        #     initial_mesh = jigsawpy.jigsaw_msh_t()
        #     initial_mesh.vert2 = self.hfun.vert2
        #     initial_mesh.tria3 = self.hfun.tria3
        self.__initial_mesh = initial_mesh

    @_dst_crs.setter
    def _dst_crs(self, dst_crs):
        self.__dst_crs = dst_crs

    @_mesh_dims.setter
    def _mesh_dims(self, mesh_dims):
        self.opts.mesh_dims = mesh_dims
