import numpy as np
from pathlib import Path
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.interpolate import RectBivariateSpline
from geomesh.fix_point_normalize import FixPointNormalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
import uuid
from pyproj import Proj, transform


class TriangularMesh:

    def __init__(self, vertices, elements, crs, values=None):
        self._vertices = vertices
        self._elements = elements
        self._crs = crs
        self._values = values

    def transform_to(self, crs):
        x, y = transform(self.proj, Proj(init=crs), self.x, self.y)
        self._vertices = np.vstack([x, y]).T
        self._crs = crs

    def interpolate(self, raster, i=1, fix_invalid=False):
        if raster.srs != self.srs:
            raster.dst_crs = self.crs
        bbox = raster.bbox
        f = RectBivariateSpline(
            raster.x,
            np.flip(raster.y),
            np.flipud(raster.read(i)).T,
            bbox=[bbox.xmin, bbox.xmax, bbox.ymin, bbox.ymax])
        idxs = np.where(np.logical_and(
                            np.logical_and(
                                bbox.xmin <= self.vertices[:, 0],
                                bbox.xmax >= self.vertices[:, 0]),
                            np.logical_and(
                                bbox.ymin <= self.vertices[:, 1],
                                bbox.ymax >= self.vertices[:, 1])))[0]
        values = f.ev(self.vertices[idxs, 0], self.vertices[idxs, 1])
        new_values = self.values.copy()
        for i, idx in enumerate(idxs):
            new_values[idx] = np.nanmean([new_values[idx], values[i]])
        self._values = new_values
        if fix_invalid:
            self.fix_invalid()

    def interpolate_collection(
        self,
        raster_collection,
        i=1,
        fix_invalid=False
    ):
        for raster in raster_collection:
            self.interpolate(raster, i=i)
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
                return self.values
            else:
                raise NotImplementedError

    def save(self, path, driver='gr3', overwrite=False):
        if path is not None:
            path = Path(path)
            if path.is_file() and not overwrite:
                raise Exception(
                    'File exists, pass overwrite=True to allow overwrite.')
            else:
                with open(path, 'w') as f:
                    f.write(self.gr3)
        else:
            print(self.gr3)

    def make_plot(
        self,
        axes=None,
        vmin=None,
        vmax=None,
        cmap='topobathy',
        levels=None,
        show=False,
        title=None,
        figsize=None,
        colors=256,
        extent=None,
        cbar_label=None,
        norm=None,
        **kwargs
    ):
        if axes is None:
            axes = plt.figure(figsize=figsize).add_subplot(111)
        if vmin is None:
            vmin = np.min(self.values)
        if vmax is None:
            vmax = np.max(self.values)
        cmap, norm, levels, col_val = self.__get_cmap(
            vmin, vmax, cmap, levels, colors, norm)
        axes.tricontourf(
            self.triangulation,
            self.values,
            levels=levels,
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            **kwargs
            )
        axes.axis('scaled')
        if extent is not None:
            axes.axis(extent)
        if title is not None:
            axes.set_title(title)
        mappable = ScalarMappable(cmap=cmap)
        mappable.set_array([])
        mappable.set_clim(vmin, vmax)
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("bottom", size="2%", pad=0.5)
        cbar = plt.colorbar(mappable, cax=cax,  # extend=cmap_extend,
                            orientation='horizontal')
        if col_val != 0:
            cbar.set_ticks([vmin, vmin + col_val * (vmax-vmin), vmax])
            cbar.set_ticklabels([np.around(vmin, 2), 0.0, np.around(vmax, 2)])
        else:
            cbar.set_ticks([vmin, vmax])
            cbar.set_ticklabels([np.around(vmin, 2), np.around(vmax, 2)])
        if cbar_label is not None:
            cbar.set_label(cbar_label)
        if show is True:
            plt.show()
        return axes

    def __get_cmap(
        self,
        vmin,
        vmax,
        cmap=None,
        levels=None,
        colors=256,
        norm=None
    ):
        colors = int(colors)
        if cmap is None:
            cmap = plt.cm.get_cmap('jet')
            if levels is None:
                levels = np.linspace(vmin, vmax, colors)
            col_val = 0.
        elif cmap == 'topobathy':
            if vmax <= 0.:
                cmap = plt.cm.seismic
                col_val = 0.
                levels = np.linspace(vmin, vmax, colors)
            else:
                wet_count = int(np.floor(colors*(float((self.values < 0.).sum())
                                                 / float(self.values.size))))
                col_val = float(wet_count)/colors
                dry_count = colors - wet_count
                colors_undersea = plt.cm.bwr(np.linspace(1., 0., wet_count))
                colors_land = plt.cm.terrain(np.linspace(0.25, 1., dry_count))
                colors = np.vstack((colors_undersea, colors_land))
                cmap = LinearSegmentedColormap.from_list('cut_terrain', colors)
                wlevels = np.linspace(vmin, 0.0, wet_count, endpoint=False)
                dlevels = np.linspace(0.0, vmax, dry_count)
                levels = np.hstack((wlevels, dlevels))
        else:
            cmap = plt.cm.get_cmap(cmap)
            levels = np.linspace(vmin, vmax, colors)
            col_val = 0.
        if vmax > 0:
            if norm is None:
                norm = FixPointNormalize(sealevel=0.0, vmax=vmax, vmin=vmin,
                                         col_val=col_val)
        return cmap, norm, levels, col_val

    @property
    def vertices(self):
        return self._vertices

    @property
    def x(self):
        return self.triangulation.x

    @property
    def y(self):
        return self.triangulation.y

    @property
    def elements(self):
        return self.triangulation.triangles

    @property
    def triangulation(self):
        return self._triangulation

    @property
    def crs(self):
        return self._crs

    @property
    def proj(self):
        return Proj(init=self.crs)

    @property
    def srs(self):
        return self.proj.srs

    @property
    def values(self):
        return self._values

    @property
    def description(self):
        return self._description

    @property
    def node_id(self):
        return self._node_id

    @property
    def element_id(self):
        return self._element_id

    @property
    def gr3(self):
        f = "{}\n".format(self.description)
        f += "{} ".format(self.elements.shape[0])
        f += "{}\n".format(self.values.shape[0])
        for i in range(self.values.shape[0]):
            f += "{:d} ".format(self.node_id[i]+1)
            f += "{:<.16E} ".format(self.x[i])
            f += " {:<.16E} ".format(self.y[i])
            f += "{:<.16E}\n".format(-self.values[i])
        for i in range(self.elements.shape[0]):
            f += "{:d} ".format(self.element_id[i]+1)
            f += "{:d} ".format(3)
            f += "{:d} ".format(self.elements[i, 0]+1)
            f += "{:d} ".format(self.elements[i, 1]+1)
            f += "{:d}\n".format(self.elements[i, 2]+1)
        # f += "{:d} ".format(len(self.ocean_boundaries))
        # f += "! total number of ocean boundaries\n"
        # f += "{:d} ".format(len(self.ocean_boundary_nodes))
        # f += "! total number0 of ocean boundary nodes\n"
        # for i, indexes in enumerate(self.ocean_boundaries):
        #     f += "{:d}".format(len(indexes))
        #     f += " ! number of nodes for ocean_boundary_"
        #     f += "{}\n".format(i)
        #     for idx in indexes:
        #         f += "{:d}\n".format(idx+1)
        # f += "{:d}".format(
        #     len(self.land_boundaries.keys()) +
        #     len(self.inner_boundaries.keys()) +
        #     len(self.inflow_boundaries.keys()) +
        #     len(self.outflow_boundaries.keys()) +
        #     len(self.weir_boundaries.keys()) +
        #     len(self.culvert_boundaries.keys()))
        # f += " ! total number of non-ocean boundaries\n"
        # f += "{:d}".format(
        #     len(self.land_boundary) +
        #     len(self.inner_boundary) +
        #     len(self.inflow_boundary) +
        #     len(self.outflow_boundary) +
        #     len(self.weir_boundary) +
        #     len(self.culvert_boundary))
        # f += " ! total number of non-ocean boundary nodes\n"
        # for key, _ in self.land_boundaries.items():
        #     f += "{:d} ".format(len(_['indexes']))
        #     f += "{:d} ".format(_['ibtype'])
        #     f += "! number of nodes and ibtype for land_boundary_"
        #     f += "{}\n".format(key)
        #     for idx in _['indexes']:
        #         f += "{:d}\n".format(idx+1)
        # for key, _ in self.inner_boundaries.items():
        #     f += "{:d} ".format(len(_['indexes']))
        #     f += "{:d} ".format(_['ibtype'])
        #     f += "! number of nodes and ibtype for inner_boundary_"
        #     f += "{}\n".format(key)
        #     for idx in _['indexes']:
        #         f += "{:d}\n".format(idx+1)
        # for key, _ in self.inflow_boundaries.items():
        #     f += "{:d} ".format(len(_['indexes']))
        #     f += "{:d} ".format(_['ibtype'])
        #     f += "! number of nodes and ibtype for inflow_boundary_"
        #     f += "{}\n".format(key)
        #     for idx in _['indexes']:
        #         f += "{:d}\n".format(idx+1)
        # for key, _ in self.outflow_boundaries.items():
        #     f += "{:d} ".format(len(_['indexes']))
        #     f += "{:d} ".format(_['ibtype'])
        #     f += "! number of nodes and ibtype for outflow_boundary_"
        #     f += "{}\n".format(key)
        #     for i in range(len(_['indexes'])):
        #         f += "{:d} ".format(_['indexes'][i]+1)
        #         f += "{:<.16E} ".format(_["barrier_heights"][i])
        #         f += "{:<.16E} ".format(
        #                 _["subcritical_flow_coefficients"][i])
        #         f += "\n"
        # for key, _ in self.weir_boundaries.items():
        #     f += "{:d} ".format(len(_['front_face_indexes']))
        #     f += "{:d} ".format(_['ibtype'])
        #     f += "! number of nodes and ibtype for weir_boundary_"
        #     f += "{}\n".format(key)
        #     for i in range(len(_['front_face_indexes'])):
        #         f += "{:d} ".format(_['front_face_indexes'][i]+1)
        #         f += "{:d} ".format(_['back_face_indexes'][i]+1)
        #         f += "{:<.16E} ".format(_["barrier_heights"][i])
        #         f += "{:<.16E} ".format(
        #                 _["subcritical_flow_coefficients"][i])
        #         f += "{:<.16E} ".format(
        #                 _["supercritical_flow_coefficients"][i])
        #         f += "\n"
        # for key, _ in self.culvert_boundaries.items():
        #     f += "{:d} ".format(len(_['indexes']))
        #     f += "{:d} ".format(_['ibtype'])
        #     f += "! number of nodes and ibtype for culvert_boundary_"
        #     f += "{}\n".format(key)
        #     for i in range(len(_['front_face_indexes'])):
        #         f += "{:d} ".format(_['front_face_indexes'][i]+1)
        #         f += "{:d} ".format(_['back_face_indexes'][i]+1)
        #         f += "{:<.16E} ".format(_["barrier_heights"][i])
        #         f += "{:<.16E} ".format(_["subcritical_flow_coefficients"][i])
        #         f += "{:<.16E} ".format(
        #             _["supercritical_flow_coefficients"][i])
        #         f += "{:<.16E} ".format(_["cross_barrier_pipe_heights"][i])
        #         f += "{:<.16E} ".format(_["friction_factors"][i])
        #         f += "{:<.16E} ".format(_["pipe_diameters"][i])
        #         f += "\n"
        # f += "{}\n".format(self.SpatialReference.ExportToWkt())
        return f

    @property
    def _vertices(self):
        return self.__vertices

    @property
    def _elements(self):
        return self.__elements

    @property
    def _crs(self):
        return self.__crs

    @property
    def _values(self):
        return self.__values

    @property
    def _description(self):
        try:
            return self.__description
        except AttributeError:
            self.description = str(uuid.uuid4())[:8]
            return self.__description

    @property
    def _node_id(self):
        try:
            return self.__node_id
        except AttributeError:
            self.__node_id = np.arange(self.values.shape[0])
            return self.__node_id

    @property
    def _element_id(self):
        try:
            return self.__element_id
        except AttributeError:
            self.__element_id = np.arange(
                self.triangulation.triangles.shape[0])
            return self.__element_id

    @property
    def _triangulation(self):
        try:
            return self.__triangulation
        except AttributeError:
            self.__triangulation = Triangulation(
                self._vertices[:, 0], self._vertices[:, 1], self._elements)
            return self.__triangulation

    @description.setter
    def description(self, description):
        self._description = description

    @_description.setter
    def _description(self, description):
        assert isinstance(description, str)
        self.__description = description

    @_vertices.setter
    def _vertices(self, vertices):
        del(self._triangulation)
        vertices = np.asarray(vertices)
        assert vertices.shape[1] == 2
        self.__vertices = vertices

    @_elements.setter
    def _elements(self, elements):
        elements = np.asarray(elements)
        assert elements.shape[1] == 3
        self.__elements = elements

    @_values.setter
    def _values(self, values):
        if values is None:
            values = np.full((1, self.vertices.shape[0]), np.nan).flatten()
        else:
            values = np.asarray(values).flatten()
            assert values.shape[0] == self.vertices.shape[0]
        self.__values = values

    @_crs.setter
    def _crs(self, crs):
        self.__crs = crs

    @_triangulation.deleter
    def _triangulation(self):
        try:
            del(self.__triangulation)
        except AttributeError:
            pass


Mesh = TriangularMesh
