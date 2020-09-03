import numpy as np
import pathlib
import logging
from collections import defaultdict
from itertools import permutations
import fiona
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import LineString, mapping
import matplotlib.pyplot as plt
from functools import lru_cache
from scipy.interpolate import RectBivariateSpline, griddata
from jigsawpy import jigsaw_msh_t
from ..raster import Raster
from .. import figures as fig
from .. import utils
from .euclidean_mesh_2d import EuclideanMesh2D


class Mesh(EuclideanMesh2D):

    def __init__(
        self,
        coords,
        triangles=None,
        quads=None,
        values=None,
        crs=None,
        description=None,
        boundaries=None,
    ):
        super().__init__(coords, triangles, quads, values, crs, description)
        self._boundaries = boundaries

    @classmethod
    def open_2dm(cls, path, crs=None):
        cls = super().open_2dm(path, crs)
        cls._values = -cls.values
        return cls

    @classmethod
    def open_grd(cls, path, crs=None):
        cls = super().open_grd(path, crs)
        cls._values = -cls.values
        return cls

    def add_boundary_type(self, ibtype):
        if ibtype not in self.boundaries:
            self.__boundaries[ibtype] = defaultdict()
        else:
            msg = f"Cannot add boundary_type={ibtype}: boundary type already "
            msg += "exists."
            raise Exception(msg)

    def set_boundary_data(self, ibtype, id, indexes, **properties):
        msg = "Indexes must be subset of node id's."
        for idx in indexes:
            assert idx in self.node_index.keys(), msg
        self.__boundaries[ibtype][id] = {
            'indexes': indexes,
            **properties
        }

    def clear_boundaries(self):
        self.__boundaries = {}

    def delete_boundary_type(self, ibtype):
        del self.__boundaries[ibtype]

    def delete_boundary_data(self, ibtype, id):
        del self.__boundaries[ibtype][id]

    def generate_boundaries(
        self,
        threshold=0.,
        land_ibtype=0,
        interior_ibtype=1,
    ):
        if np.any(np.isnan(self.values)):
            msg = "Mesh contains invalid values. Raster values must "
            msg += "be interpolated to the mesh before generating "
            msg += "boundaries."
            raise Exception(msg)

        # generate exterior boundaries
        for ring in self.outer_ring_collection.values():
            # find boundary edges
            edge_tag = np.full(ring.shape, 0)
            edge_tag[np.where(self.values[ring[:, 0]] < threshold)[0], 0] = -1
            edge_tag[np.where(self.values[ring[:, 1]] < threshold)[0], 1] = -1
            edge_tag[np.where(self.values[ring[:, 0]] >= threshold)[0], 0] = 1
            edge_tag[np.where(self.values[ring[:, 1]] >= threshold)[0], 1] = 1
            # sort boundary edges
            ocean_boundary = list()
            land_boundary = list()
            for i, (e0, e1) in enumerate(edge_tag):
                if np.any(np.asarray((e0, e1)) == -1):
                    ocean_boundary.append(tuple(ring[i, :]))
                elif np.any(np.asarray((e0, e1)) == 1):
                    land_boundary.append(tuple(ring[i, :]))
            ocean_boundaries = utils.sort_edges(ocean_boundary)
            land_boundaries = utils.sort_edges(land_boundary)
            # add ocean boundaries
            if None not in self.boundaries:
                self.add_boundary_type(None)
            _bnd_id = len(self.boundaries[None])
            for bnd in ocean_boundaries:
                e0, e1 = [list(t) for t in zip(*bnd)]
                e0 = list(map(self.get_node_id, e0))
                data = e0 + [self.get_node_id(e1[-1])]
                self.set_boundary_data(None, _bnd_id, data)
                _bnd_id += 1
            # add land boundaries
            if land_ibtype not in self.boundaries:
                self.add_boundary_type(land_ibtype)
            _bnd_id = len(self._boundaries[land_ibtype])
            for bnd in land_boundaries:
                e0, e1 = [list(t) for t in zip(*bnd)]
                e0 = list(map(self.get_node_id, e0))
                data = e0 + [self.get_node_id(e1[-1])]
                self.set_boundary_data(land_ibtype, _bnd_id, data)
                _bnd_id += 1
        # generate interior boundaries
        _bnd_id = 0
        _interior_boundaries = defaultdict()
        for interiors in self.inner_ring_collection.values():
            for interior in interiors:
                e0, e1 = [list(t) for t in zip(*interior)]
                if utils.signed_polygon_area(self.coords[e0, :]) < 0:
                    e0 = list(reversed(e0))
                    e1 = list(reversed(e1))
                e0 = list(map(self.get_node_id, e0))
                e0 += [e0[0]]
                _interior_boundaries[_bnd_id] = e0
                _bnd_id += 1
        self.add_boundary_type(interior_ibtype)
        for bnd_id, data in _interior_boundaries.items():
            self.set_boundary_data(interior_ibtype, bnd_id, data)

    def write_boundaries(self, path, overwrite=False):
        path = pathlib.Path(path)
        with fiona.open(
                    path.resolve(),
                    'w',
                    driver='ESRI Shapefile',
                    crs=self.crs.srs,
                    schema={
                        'geometry': 'LineString',
                        'properties': {
                            'id': 'int',
                            'ibtype': 'str',
                            'bnd_id': 'str'
                            }
                        }) as dst:
            _cnt = 0
            for ibtype, bnds in self.boundaries.items():
                for id, bnd in bnds.items():
                    idxs = list(map(self.get_node_index, bnd['indexes']))
                    linear_ring = LineString(self.xy[idxs].tolist())
                    dst.write({
                            "geometry": mapping(linear_ring),
                            "properties": {
                                "id": _cnt,
                                "ibtype": ibtype,
                                "bnd_id": f"{ibtype}:{id}"
                                }
                            })
                    _cnt += 1

    def interpolate(
        self,
        raster,
        band=1,
        fix_invalid=False,
        method='spline',
        **kwargs
    ):
        assert method in ['griddata', 'spline']
        if raster.srs != self.srs:
            raster = Raster(raster.path)
            raster.warp(self.crs)
        getattr(self, f'_{method}_interp')(raster, band, raster.bbox, **kwargs)
        if fix_invalid:
            self.fix_invalid()

    def interpolate_collection(
        self,
        raster_collection,
        band=1,
        fix_invalid=False
    ):
        for raster in raster_collection:
            self.interpolate(raster, band, False)
        if fix_invalid:
            self.fix_invalid()

    def has_invalid(self):
        return np.any(np.isnan(self.values))

    def fix_invalid(self, method='nearest'):
        if self.has_invalid():
            if method == 'nearest':
                idx = np.where(~np.isnan(self.values))
                _idx = np.where(np.isnan(self.values))
                values = griddata(
                    (self.x[idx], self.y[idx]), self.values[idx],
                    (self.x[_idx], self.y[_idx]), method='nearest')
                new_values = self.values.copy()
                for i, idx in enumerate(_idx):
                    new_values[idx] = values[i]
                self._values = new_values
            else:
                msg = 'Only nearest method is available.'
                raise NotImplementedError(msg)

    @fig._figure
    def make_plot(
        self,
        axes=None,
        vmin=None,
        vmax=None,
        show=False,
        title=None,
        # figsize=rcParams["figure.figsize"],
        extent=None,
        cbar_label=None,
        **kwargs
    ):
        if vmin is None:
            vmin = np.min(self.values)
        if vmax is None:
            vmax = np.max(self.values)
        kwargs.update(**fig.get_topobathy_kwargs(self.values, vmin, vmax))
        kwargs.pop('col_val')
        levels = kwargs.pop('levels')
        if vmin != vmax:
            self.tricontourf(
                axes=axes,
                levels=levels,
                vmin=vmin,
                vmax=vmax,
                **kwargs
            )
        else:
            self.tripcolor(axes=axes, **kwargs)
        self.quadface(axes=axes, **kwargs)
        axes.axis('scaled')
        if extent is not None:
            axes.axis(extent)
        if title is not None:
            axes.set_title(title)
        mappable = ScalarMappable(cmap=kwargs['cmap'])
        mappable.set_array([])
        mappable.set_clim(vmin, vmax)
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("bottom", size="2%", pad=0.5)
        cbar = plt.colorbar(
            mappable,
            cax=cax,
            orientation='horizontal'
        )
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels([np.around(vmin, 2), np.around(vmax, 2)])
        if cbar_label is not None:
            cbar.set_label(cbar_label)
        return axes

    @fig._figure
    def plot_boundary(
        self,
        ibtype,
        id,
        tags=True,
        axes=None,
        show=False,
        figsize=None,
        **kwargs
    ):

        boundary = list(map(
            self.get_node_index, self.boundaries[ibtype][id]['indexes']))
        p = axes.plot(self.x[boundary], self.y[boundary], **kwargs)
        if tags:
            axes.text(
                self.x[boundary[len(boundary)//2]],
                self.y[boundary[len(boundary)//2]],
                f"ibtype={ibtype}\nid={id}",
                color=p[-1].get_color()
                )
        return axes

    @fig._figure
    def plot_boundaries(
        self,
        axes=None,
        show=False,
        figsize=None,
        **kwargs
    ):
        kwargs.update({'axes': axes})
        for ibtype, bnds in self.boundaries.items():
            for id in bnds:
                axes = self.plot_boundary(ibtype, id, **kwargs)
                kwargs.update({'axes': axes})
        return kwargs['axes']

    @property
    def logger(self):
        try:
            return self.__logger
        except AttributeError:
            self.__logger = logging.getLogger(
                __name__ + '.' + self.__class__.__name__)
            return self.__logger

    @property
    def boundaries(self):
        return self._boundaries

    @property
    @lru_cache(maxsize=None)
    def point_neighbors(self):
        point_neighbors = defaultdict(set)
        for simplex in self.triangulation.triangles:
            for i, j in permutations(simplex, 2):
                point_neighbors[i].add(j)
        return point_neighbors

    @property
    @lru_cache(maxsize=None)
    def point_distances(self):
        point_distances = {}
        for i, (x, y) in enumerate(self.xy):
            point_distances[i] = {}
            for neighbor in self.point_neighbors[i]:
                point_distances[i][neighbor] = np.sqrt(
                    (self.x[i] - self.x[neighbor])**2
                    +
                    (self.y[i] - self.y[neighbor])**2
                    )
        return point_distances

    @property
    def gr3(self):
        """
        Returns a gr3 string with boundary information included.
        """
        f = super().gr3
        # ocean boundaries
        f += f"{len(self.boundaries[None].keys()):d} "
        f += "! total number of ocean boundaries\n"
        # count total number of ocean boundaries
        _sum = np.sum(
            [len(boundary['indexes'])
             for boundary in self.boundaries[None].values()]
            )
        f += f"{int(_sum):d} ! total number of ocean boundary nodes\n"
        # write ocean boundary indexes
        for i, boundary in self.boundaries[None].items():
            f += f"{len(boundary['indexes']):d}"
            f += f" ! number of nodes for ocean_boundary_{i}\n"
            for idx in boundary['indexes']:
                f += f"{idx}\n"
        # count non-ocean boundaries
        _cnt = 0
        for ibtype, bnd in self.boundaries.items():
            if ibtype is not None:
                _cnt += len(self.boundaries[ibtype].keys())
        f += f"{_cnt:d}  ! total number of non-ocean boundaries\n"
        # count all remaining nodes of all non-ocean boundaries
        _cnt = 0
        for ibtype, bnd in self.boundaries.items():
            if ibtype is not None:
                _cnt = int(np.sum([len(x['indexes'])
                           for x in self.boundaries[ibtype].values()]))
        f += f"{_cnt:d} ! Total number of non-ocean boundary nodes\n"
        # write all non-ocean boundaries
        for ibtype, bnds in self.boundaries.items():
            if ibtype is not None:
                # write land boundaries
                for i, bnd in bnds.items():
                    f += f"{len(bnd['indexes']):d} "
                    f += f"{ibtype} "
                    f += "! number of nodes for boundary_"
                    f += f"{i}\n"
                    for idx in bnd['indexes']:
                        f += f"{idx}\n"
        return f

    def _spline_interp(self, raster, band, bbox, kx=3, ky=3, s=0):
        f = RectBivariateSpline(
            raster.x,
            np.flip(raster.y),
            np.flipud(raster.values).T,
            bbox=[bbox.xmin, bbox.xmax, bbox.ymin, bbox.ymax],
            kx=kx, ky=ky, s=s)
        idxs = np.where(np.logical_and(
                            np.logical_and(
                                bbox.xmin <= self.coords[:, 0],
                                bbox.xmax >= self.coords[:, 0]),
                            np.logical_and(
                                bbox.ymin <= self.coords[:, 1],
                                bbox.ymax >= self.coords[:, 1])))[0]
        values = f.ev(self.coords[idxs, 0], self.coords[idxs, 1])
        new_values = self.values.copy()
        new_values[idxs] = values
        self._values = new_values

    def _griddata_interp(
        self,
        raster,
        band,
        bbox,
        method='linear',
        fill_value=np.nan,
        rescale=False
    ):
        idxs = np.where(np.logical_and(
                            np.logical_and(
                                bbox.xmin <= self.coords[:, 0],
                                bbox.xmax >= self.coords[:, 0]),
                            np.logical_and(
                                bbox.ymin <= self.coords[:, 1],
                                bbox.ymax >= self.coords[:, 1])))[0]
        dstx = self.coords[idxs, 0]
        dsty = self.coords[idxs, 1]
        srcx, srcy = np.meshgrid(raster.x, raster.y)
        srcx = srcx.flatten()
        srcy = np.flipud(srcy.flatten())
        srcz = raster.values.flatten()
        values = griddata(
            (srcx, srcy), srcz, (dstx, dsty),
            method=method,
            fill_value=fill_value,
            rescale=rescale
            )
        new_values = self.values.copy()
        new_values[idxs] = values
        self._values = new_values

    @property
    @lru_cache(maxsize=None)
    def vert2(self):
        return np.array(
            [(coord, id) for id, coord in self._coords.items()],
            dtype=jigsaw_msh_t.VERT2_t
            )

    @property
    @lru_cache(maxsize=None)
    def tria3(self):
        return np.array(
            [(list(map(self.get_node_index, index)), id)
             for id, index in self._triangles.items()],
            dtype=jigsaw_msh_t.TRIA3_t)

    @property
    @lru_cache(maxsize=None)
    def quad4(self):
        return np.array(
            [(list(map(self.get_node_index, index)), id)
             for id, index in self._quads.items()],
            dtype=jigsaw_msh_t.QUAD4_t)

    @property
    def value(self):
        return np.array(
            self.values.reshape((self.values.size, 1)),
            dtype=jigsaw_msh_t.REALS_t)

    @property
    def mesh(self):
        mesh = jigsaw_msh_t()
        mesh.mshID = 'euclidean-mesh'
        mesh.ndims = 2
        mesh.vert2 = self.vert2
        mesh.tria3 = self.tria3
        mesh.quad4 = self.quad4
        # mesh.hexa8 = self.hexa8
        mesh.value = self.value
        return mesh

    @property
    @lru_cache(maxsize=None)
    def index_ring_collection(self):
        return utils.index_ring_collection(self.mesh)

    @property
    @lru_cache(maxsize=None)
    def vertices_around_vertex(self):
        return utils.vertices_around_vertex(self.mesh)

    @property
    @lru_cache(maxsize=None)
    def faces_around_vertex(self):
        return utils.faces_around_vertex(self.mesh)

    @property
    @lru_cache(maxsize=None)
    def outer_ring_collection(self):
        return utils.outer_ring_collection(self.mesh)

    @property
    @lru_cache(maxsize=None)
    def inner_ring_collection(self):
        return utils.inner_ring_collection(self.mesh)

    @property
    def _boundaries(self):
        return self.__boundaries

    @_boundaries.setter
    def _boundaries(self, boundaries):
        """
        boundaries = {ibtype: {id: {'indexes': [i0, ..., in], 'properties': object }}
        """
        self.clear_boundaries()  # clear
        if boundaries is not None:
            for ibtype, bnds in boundaries.items():
                self.add_boundary_type(ibtype)
                for id, bnd in bnds.items():
                    if 'properties' in bnd.keys():
                        properties = bnd['properties']
                    else:
                        properties = {}
                    self.set_boundary_data(
                        ibtype,
                        id,
                        bnd['indexes'],
                        **properties
                    )

    @property
    @lru_cache(maxsize=None)
    def _grd(self):
        """
        adds boundary data to _grd dict obtained from super()
        """
        _grd = super()._grd
        _grd.update({"boundaries": self.boundaries})
        return _grd

    @property
    @lru_cache(maxsize=None)
    def _sms2dm(self):
        """
        adds boundary data to _sms2dm dict obtained from super()
        """
        _sms2dm = super()._sms2dm
        # _sms2dm.update({"boundaries": self.boundaries})
        return _sms2dm

    @property
    @lru_cache(maxsize=None)
    def _nodes(self):
        """
        updates _nodes dict obtained from super() as positive-down
        """
        _nodes = super()._nodes
        _nodes.update(
            {id: ((x, y), -value) for id, ((x, y), value) in
             super()._nodes.items()})
        return _nodes
