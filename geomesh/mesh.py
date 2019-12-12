import numpy as np
import pathlib
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.path import Path
from matplotlib.transforms import Bbox
from scipy.interpolate import RectBivariateSpline
from collections import defaultdict
from itertools import permutations
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
import uuid
from pyproj import Proj, CRS, Transformer
from tempfile import TemporaryDirectory
from shapely.geometry import shape, mapping, Polygon, LinearRing, MultiPolygon
from shapely.ops import transform
import fiona
from jigsawpy import loadmsh, jigsaw_msh_t
import geomesh
from geomesh.size_function import SizeFunction
from geomesh.pslg import PlanarStraightLineGraph
from geomesh import jigsaw_tools as jt
from geomesh.fix_point_normalize import FixPointNormalize


class Mesh:

    def __init__(
        self,
        vertices,
        elements,
        crs,
        values=None,
        node_id=None,
        element_id=None
    ):
        self._vertices = vertices
        self._elements = elements
        self._crs = crs
        self._values = values

    @classmethod
    def open(cls, mesh, crs, driver='grd'):
        # load from file
        if isinstance(mesh, (str, pathlib.Path)):
            # sanitize inputs
            if not pathlib.Path(str(mesh)).is_file():
                raise IOError('ERROR: File not found.')
            driver = cls._get_ascii_driver(driver)
            return cls(**driver(mesh), crs=crs)

        # load from jigsaw object
        elif isinstance(mesh, jigsaw_msh_t):
            return cls(
                mesh.vert2['coord'], mesh.tria3['index'], crs, mesh.values)

        # raise if incorrect type
        else:
            msg = f'Must be of type {str}, {pathlib.Path} or {jigsaw_msh_t}.'
            raise TypeError(msg)

    def get_x(self, crs=None):
        return self.get_xy(crs)[:, 0]

    def get_y(self, crs=None):
        return self.get_xy(crs)[:, 1]

    def get_xy(self, crs=None):
        if crs is not None:
            crs = CRS.from_user_input(crs)
            transformer = Transformer.from_crs(self.crs, crs, always_xy=True)
            x, y = transformer.transform(self.x, self.y)
            return np.vstack([x, y]).T
        else:
            return np.vstack([self.x, self.y]).T

    def tricontourf(self, **kwargs):
        return jt.tricontourf(self.mesh, **kwargs)

    def triplot(self, **kwargs):
        return jt.triplot(self.mesh, **kwargs)

    def transform_to(self, dst_crs):
        dst_crs = CRS.from_user_input(dst_crs)
        if self.srs != dst_crs.srs:
            transformer = Transformer.from_crs(
                self.crs, dst_crs, always_xy=True)
            x, y = transformer.transform(self.x, self.y)
            self._vertices = np.vstack([x, y]).T
            self._crs = dst_crs

    def interpolate(self, raster, band=1, fix_invalid=False):
        if raster.srs != self.srs:
            raster = geomesh.Raster(
                raster.path, crs=raster.crs, dst_crs=self.crs)
        bbox = raster.bbox
        f = RectBivariateSpline(
            raster.x,
            np.flip(raster.y),
            np.flipud(raster.values).T,
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
        # for i, idx in enumerate(idxs):
        #     new_values[idx] = np.nanmean([new_values[idx], values[i]])
        new_values[idxs] = values
        self._values = new_values
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

    def save(self, path, driver='gr3', overwrite=False):
        if path is not None:
            path = pathlib.Path(path)
            if path.is_file() and not overwrite:
                msg = 'File exists, use overwrite=True to allow overwrite.'
                raise Exception(msg)
            else:
                with open(path, 'w') as f:
                    f.write(self._get_output_string(driver))
        else:
            print(self._get_output_string(driver))

    def get_critical_timestep(self, cfl, maxvel, g=9.8):
        """
        http://swash.sourceforge.net/online_doc/swashuse/node47.html
        """
        points = self.get_xy(3395)
        distances = list()
        for k, v in self.node_neighbors.items():
            x0, y0 = points[k]
            _distances = list()
            for idx in v:
                x1, y1 = points[idx]
                _distances.append(np.sqrt((x0 - x1)**2 + (y0 - y1)**2))
            distances.append(np.mean(_distances))
        dt = cfl * np.asarray(distances) / np.sqrt(
            g*np.abs(self.values+np.finfo(float).eps)) + np.abs(maxvel)
        return np.min(dt)

    def get_multipolygon(self, dst_crs=None):
        multipolygon = self.multipolygon
        if dst_crs is not None:
            dst_crs = CRS.from_user_input(dst_crs)
            multipolygon = self._transform_multipolygon(
                multipolygon, self.crs, dst_crs)
        return multipolygon

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
            fig = plt.figure(figsize=figsize)
            axes = fig.add_subplot(111)
        vmin = np.min(self.values) if vmin is None else float(vmin)
        vmax = np.max(self.values) if vmax is None else float(vmax)
        cmap, norm, levels, col_val = self._get_cmap(
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

    def _get_cmap(
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
                wet_count = int(
                    np.floor(colors*(
                        float((self.values < 0.).sum())
                        / float(self.values.size))))
                col_val = float(wet_count)/colors
                dry_count = int(
                    np.floor(colors*(
                        float((self.values > 0.).sum())
                        / float(self.values.size))))
                colors_undersea = plt.cm.bwr(np.linspace(1., 0., wet_count))
                colors_land = plt.cm.terrain(np.linspace(0.25, 1., dry_count))
                colors = np.vstack((colors_undersea, colors_land))
                cmap = LinearSegmentedColormap.from_list('cut_terrain', colors)
                wlevels = np.linspace(vmin, 0.0, wet_count, endpoint=False)
                dlevels = np.linspace(0., vmax, dry_count)
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

    def _get_output_string(self, driver):
        if driver in ['grd', 'gr3', 'adcirc', 'schism']:
            return self.gr3
        if driver in ['sms', '2dm']:
            return self.sms

    @staticmethod
    def _parse_grd(path):
        grd = dict()
        grd['vertices'] = list()
        grd['values'] = list()
        grd['elements'] = list()
        with open(pathlib.Path(path), 'r') as f:
            f.readline()
            NE, NP = map(int, f.readline().split())
            for i in range(NP):
                _, x, y, z = f.readline().split()
                grd['vertices'].append([float(x), float(y)])
                grd['values'].append(-float(z))
            for i in range(NE):
                line = f.readline().split()
                grd['elements'].append([int(x)-1 for x in line[2:]])
        return grd

    @staticmethod
    def _load_mesh(path):
        mesh = jigsaw_msh_t()
        loadmsh(path, mesh)
        raise NotImplementedError

    @staticmethod
    def _parse_2dm(path):
        grd = dict()
        grd['node_id'] = list()
        grd['vertices'] = list()
        grd['values'] = list()
        grd['element_id'] = list()
        grd['elements'] = list()
        with open(pathlib.Path(path), 'r') as f:
            f.readline()
            while 1:
                line = f.readline().split()
                if len(line) == 0:
                    break
                if line[0] == 'E3T':
                    grd['element_id'].append(int(line[1]))
                    grd['elements'].append([int(x)-1 for x in line[2:]])
                if line[0] == 'ND':
                    grd['node_id'].append(line[1])
                    grd['vertices'].append([float(x) for x in line[2:-1]])
                    grd['values'].append(float(line[-1]))
        return grd

    @staticmethod
    def _transform_multipolygon(multipolygon, src_crs, dst_crs):
        if dst_crs.srs != src_crs.srs:
            transformer = Transformer.from_crs(
                src_crs, dst_crs, always_xy=True)
            polygon_collection = list()
            for polygon in multipolygon:
                polygon_collection.append(
                    transform(transformer.transform, polygon))
            outer = polygon_collection.pop(0)
            multipolygon = MultiPolygon([outer, *polygon_collection])
        return multipolygon

    @classmethod
    def _get_ascii_driver(cls, driver):
        # select ascii driver
        if driver is None:
            msg = 'Must specify a driver type for ASCII text files.'
            raise IOError(msg)

        # grd
        elif driver.lower() in ['grd', 'gr3', 'adcirc', 'schism']:
            return cls._parse_grd

        # jigsaw_msh_t
        elif driver.lower() in ['jigsaw', 'msh', 'geomesh']:
            return cls._load_mesh

        # 2dm
        elif driver.lower() in ['sms', '2dm']:
            return cls._parse_2dm

    @property
    def vertices(self):
        return self._vertices

    @property
    def x(self):
        return self.vertices[:, 0]

    @property
    def y(self):
        return self.vertices[:, 1]

    @property
    def xy(self):
        return self.vertices

    @property
    def elements(self):
        return self._elements

    @property
    def bbox(self):
        x0 = np.min(self.x)
        x1 = np.max(self.x)
        y0 = np.min(self.y)
        y1 = np.max(self.y)
        return Bbox([[x0, y0], [x1, y1]])

    @property
    def triangulation(self):
        return self._triangulation

    @property
    def triangles(self):
        return self.triangulation.triangles

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
    def index_ring_collection(self):
        tri = self.triangulation
        boundary_edges = list()
        idxs = np.vstack(list(np.where(tri.neighbors == -1))).T
        for i, j in idxs:
            boundary_edges.append(
                (tri.triangles[i, j], tri.triangles[i, (j+1) % 3]))
        boundary_edges = np.asarray(boundary_edges)
        index_ring_collection = list()
        ordered_edges = [boundary_edges[-1, :]]
        boundary_edges = np.delete(boundary_edges, -1, axis=0)
        while boundary_edges.shape[0] > 0:
            try:
                idx = np.where(
                    boundary_edges[:, 0]
                    == ordered_edges[-1][1])[0][0]
                ordered_edges.append(boundary_edges[idx, :])
                boundary_edges = np.delete(boundary_edges, idx, axis=0)
            except IndexError:
                index_ring_collection.append(
                    np.asarray(ordered_edges))
                ordered_edges = [boundary_edges[-1, :]]
                boundary_edges = np.delete(boundary_edges, -1, axis=0)
        return index_ring_collection

    @property
    def pslg(self):
        try:
            return self.__pslg
        except AttributeError:
            self.__pslg = PlanarStraightLineGraph(self)
            return self.__pslg

    @property
    def mesh(self):
        mesh = jigsaw_msh_t()
        mesh.mshID = 'euclidean-mesh'
        mesh.ndims = +2
        mesh.vert2 = self.vert2
        mesh.tria3 = self.tria3
        mesh.value = self.mesh_value
        return mesh

    @property
    def size_function(self):
        try:
            return self.__size_function
        except AttributeError:
            self.__size_function = SizeFunction(self.hfun, crs=self.crs)
            return self.__size_function

    @property
    def vert2(self):
        return np.asarray(
            [(coord, 0) for coord in self.vertices],
            dtype=jigsaw_msh_t.VERT2_t)

    @property
    def tria3(self):
        return np.asarray(
            [(tria3, 0) for tria3 in self.triangles],
            dtype=jigsaw_msh_t.TRIA3_t)

    @property
    def hfun(self):
        hfun = jigsaw_msh_t()
        hfun.mshID = 'euclidean-mesh'
        hfun.ndims = +2
        hfun.vert2 = self.vert2
        hfun.tria3 = self.tria3
        hfun.value = self.hfun_value
        return hfun

    @property
    def point_neighbors(self):
        point_neighbors = defaultdict(set)
        for simplex in self.elements:
            for i, j in permutations(simplex, 2):
                point_neighbors[i].add(j)
        return point_neighbors

    @property
    def hfun_values(self):
        try:
            return self.__hfun_values
        except AttributeError:
            assert self.crs is not None
            self.__hfun_values = np.empty(self.values.size)
            for idx, idxs in self.point_neighbors.items():
                x0, y0 = self.xy[idx]
                distances = list()
                for _idx in idxs:
                    x1, y1 = self.xy[_idx]
                    distances.append(np.sqrt((x0 - x1)**2 + (y0 - y1)**2))
                self.__hfun_values[idx] = np.mean(distances)
            return self.__hfun_values

    @property
    def hfun_value(self):
        return np.array(
            self.hfun_values.reshape((self.hfun_values.size, 1)),
            dtype=jigsaw_msh_t.REALS_t)

    @property
    def mesh_value(self):
        return np.array(
            self.values.reshape((self.values.size, 1)),
            dtype=jigsaw_msh_t.REALS_t)

    @property
    def multipolygon(self):
        multipolygon = shape(self._multipolygon_collection[0]['geometry'])
        if isinstance(multipolygon, Polygon):
            multipolygon = MultiPolygon([multipolygon])
        return multipolygon

    @property
    def gr3(self):
        # TODO: Make faster using np.array2string
        f = f"{self.description}\n"
        f += f"{self.elements.shape[0]} "
        f += f"{self.values.shape[0]}\n"
        for i in range(self.values.shape[0]):
            f += f"{i+1:d} "
            f += f"{self.x[i]:G} "
            f += f" {self.y[i]:G} "
            f += f"{-self.values[i]:G}\n"
        for i in range(self.elements.shape[0]):
            f += f"{i+1:d} "
            f += f"{len(self.elements[i, :]):d} "
            for e in self.elements[i, :]:
                f += f"{e+1:d} "
            f += "\n"
        return f

    @property
    def sms(self):
        # TODO: Make faster using np.array2string
        f = "MESH2D\n"
        for i in range(len(self.element_id)):
            # bug: this is not considering quads and other geometries
            f += f"E3T {i + 1} "
            f += f"{self.elements[i, 0]+1} "
            f += f"{self.elements[i, 1]+1} "
            f += f"{self.elements[i, 2]+1}\n"
        for i in range(len(self.node_id)):
            f += f"ND {i + 1} "
            f += f"{self.x[i]:G} "
            f += f"{self.y[i]:G} "
            f += f"{self.values[i]:G}\n"
        return f

    @property
    def geom(self):
        return self.pslg.geom

    @property
    def _vertices(self):
        return self.__vertices

    @property
    def _elements(self):
        return self.__elements

    @property
    def _values(self):
        return self.__values

    @property
    def _crs(self):
        return self.__crs

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
        return Triangulation(
                self.vertices[:, 0],
                self.vertices[:, 1],
                self.elements
            )

    @property
    def _multipolygon_collection(self):
        try:
            return self.__multipolygon_collection['shp']
        except AttributeError:
            linear_ring_collection = list()
            for i, index_ring in enumerate(self.index_ring_collection):
                xy = self.xy[index_ring[:, 0]]
                # if self.crs.is_geographic:
                #     xy = np.fliplr(xy)
                linear_ring_collection.append(LinearRing(xy))
            if len(linear_ring_collection) > 1:
                # reorder linear rings from above
                areas = [Polygon(linear_ring).area
                         for linear_ring in linear_ring_collection]
                idx = np.where(areas == np.max(areas))[0][0]
                polygon_collection = list()
                outer_ring = linear_ring_collection.pop(idx)
                path = Path(np.asarray(outer_ring.coords), closed=True)
                while len(linear_ring_collection) > 0:
                    inner_rings = list()
                    for i, linear_ring in reversed(
                            list(enumerate(linear_ring_collection))):
                        xy = np.asarray(linear_ring.coords)[0, :]
                        if path.contains_point(xy):
                            inner_rings.append(linear_ring_collection.pop(i))
                    polygon_collection.append(Polygon(outer_ring, inner_rings))
                    if len(linear_ring_collection) > 0:
                        areas = [Polygon(linear_ring).area
                                 for linear_ring in linear_ring_collection]
                        idx = np.where(areas == np.max(areas))[0][0]
                        outer_ring = linear_ring_collection.pop(idx)
                        path = Path(np.asarray(outer_ring.coords), closed=True)
                multipolygon = MultiPolygon(polygon_collection)
            else:
                multipolygon = MultiPolygon(
                    [Polygon(linear_ring_collection.pop())])
            tmpdir = TemporaryDirectory(
                prefix=geomesh.tmpdir, suffix='_mesh_multipolygon_collection')
            with fiona.open(
                tmpdir.name,
                'w',
                driver='ESRI Shapefile',
                crs=self.srs,
                schema={
                    'geometry': 'MultiPolygon',
                    'properties': {}}) as shp:
                shp.write({
                    "geometry": mapping(multipolygon),
                    "properties": {}})
            shp = fiona.open(tmpdir.name)
            self.__multipolygon_collection = {'tmpdir': tmpdir, 'shp': shp}
            return self.__multipolygon_collection['shp']

    @description.setter
    def description(self, description):
        self._description = description

    @_description.setter
    def _description(self, description):
        assert isinstance(description, str)
        self.__description = description

    @_vertices.setter
    def _vertices(self, vertices):
        # triangulation depends of vertices
        del(self._triangulation)
        del(self._multipolygon_collection)
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
            values = np.full((self.vertices.shape[0],), np.nan)
        else:
            values = np.asarray(values).flatten()
        assert values.shape[0] == self.vertices.shape[0]
        self.__values = values

    @_crs.setter
    def _crs(self, crs):
        self.__crs = CRS.from_user_input(crs)

    @_multipolygon_collection.deleter
    def _multipolygon_collection(self):
        try:
            del(self.__multipolygon_collection)
        except AttributeError:
            pass

    @_triangulation.deleter
    def _triangulation(self):
        try:
            del(self.__triangulation)
        except AttributeError:
            pass
