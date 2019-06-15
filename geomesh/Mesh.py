import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from osgeo import osr
from scipy.interpolate import RectBivariateSpline, griddata
import pymesh
from geomesh import gdal_tools
from geomesh import PlanarStraightLineGraph
from geomesh._SpatialReference import _SpatialReference


class Mesh(_SpatialReference):

    def __init__(self, vertices, elements, values=None, SpatialReference=None):
        super(Mesh, self).__init__(SpatialReference)
        self._vertices = vertices
        self._elements = elements
        self._values = values

    def get_x(self, SpatialReference=None):
        """ """
        return self.get_xy(SpatialReference)[:, 0]

    def get_y(self, SpatialReference=None):
        """ """
        return self.get_xy(SpatialReference)[:, 1]

    def get_xy(self, SpatialReference=None):
        return self.transform_array(self.xy, self.SpatialReference,
                                    SpatialReference)

    def get_extent(self, SpatialReference=None):
        xy = self.get_xy(SpatialReference)
        return (np.min(xy[:, 0]), np.max(xy[:, 0]),
                np.min(xy[:, 1]), np.max(xy[:, 1]))

    def get_attribute(self, attribute_name):
        """ """
        return self.__Mesh.get_attribute('attribute_name')

    def compute_planar_straight_line_graph(self):
        unique_edges = list()
        for i, elements in enumerate(self.mpl_tri.neighbors):
            for j, element_idx in enumerate(elements):
                if element_idx == -1:
                    unique_edges.append((self.mpl_tri.triangles[i, j],
                                         self.mpl_tri.triangles[i, (j+1) % 3]))
        # sort the unique edges into a collection of rings.
        ring_collection = list()
        ring = [unique_edges.pop(0)]
        while len(unique_edges) > 0:
            idx = np.where(ring[-1][1] == np.asarray(unique_edges)[:, 0])
            try:
                ring.append(unique_edges.pop(idx[0][0]))
            except IndexError:
                ring_collection.append(np.asarray(ring))
                ring = [unique_edges.pop(0)]
        # sort between outer and inner vertices
        lengths = []
        for ring in ring_collection:
            x = np.vstack([self.x[ring[:, 0]], self.x[ring[:, 0]]])
            dx = np.abs(np.diff(x, axis=0))
            y = np.vstack([self.y[ring[:, 0]], self.y[ring[:, 0]]])
            dy = np.abs(np.diff(y, axis=0))
            lengths.append(np.sum(np.sqrt(dx**2+dy**2)))
        outer_edges = ring_collection.pop(
                np.where(np.max(lengths) == lengths)[0][0])
        inner_edges = ring_collection
        outer_vertices = self.vertices[outer_edges[:, 0]]
        inner_vertices = [self.vertices[ring[:, 0]] for ring in inner_edges]
        print(self.x)
        plt.scatter(outer_vertices[:, 0], outer_vertices[:, 1])
        plt.show()
        BREAKEM
        return PlanarStraightLineGraph(
                self.SpatialReference, outer_vertices, *inner_vertices,
                outer_edges=outer_edges, inner_edges=inner_edges)

    def interpolate(self, Dataset, method='spline', **kwargs):
        if method == 'spline':
            return self.interpolate_bivariate_spline(Dataset, **kwargs)
        else:
            return self.interpolate_griddata(Dataset, method=method, **kwargs)

    def interpolate_bivariate_spline(self, Dataset, **kwargs):
        if not self.SpatialReference.IsSame(
                    gdal_tools.get_SpatialReference(Dataset)):
            Dataset = gdal_tools.Warp(Dataset, dstSRS=self.SpatialReference)
        x, y, z = gdal_tools.get_arrays(Dataset)
        bbox = gdal_tools.get_Bbox(Dataset)
        f = RectBivariateSpline(x, y, z.T, bbox=[bbox.xmin, bbox.xmax,
                                                 bbox.ymin, bbox.ymax])
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
            new_values[idx] = values[i]
        self._values = new_values
        return self.values

    def interpolate_griddata(self, Dataset, **kwargs):
        raise NotImplementedError
        xyz = gdal_tools.get_xyz(Dataset)
        values = griddata((xyz[:, 0], xyz[:, 1]), xyz[:, 2],
                          (self.vertices[:, 0], self.vertices[:, 1]),
                          **kwargs)
        self.__Mesh.set_attribute('values', values)
        return self.values

    def fix_invalid(self, method='nearest'):
        if method == 'nearest':
            idx = np.where(~np.isnan(self.values))
            _idx = np.where(np.isnan(self.values))
            values = griddata((self.x[idx], self.y[idx]), self.values[idx],
                              (self.x[_idx], self.y[_idx]), method='nearest')
            new_values = self.values.copy()
            for i, idx in enumerate(_idx):
                new_values[idx] = values[i]
            self._values = new_values
            return self.values
        else:
            raise NotImplementedError

    def transform_array(self, xy, input_SpatialReference,
                        output_SpatialReference=None):
        if isinstance(input_SpatialReference, int):
            EPSG = input_SpatialReference
            input_SpatialReference = osr.SpatialReference()
            input_SpatialReference.ImportFromEPSG(EPSG)
        assert isinstance(input_SpatialReference, osr.SpatialReference)
        if output_SpatialReference is None:
            output_SpatialReference = self.SpatialReference
        assert isinstance(output_SpatialReference, osr.SpatialReference)
        if not output_SpatialReference.IsSame(input_SpatialReference):
            CoordinateTransform = osr.CoordinateTransformation(
                                                    input_SpatialReference,
                                                    output_SpatialReference)
            xy = [(x, y) for x, y in xy]
            xy = CoordinateTransform.TransformPoints(xy)
            xy = np.asarray([(x, y) for x, y, _ in xy])
        return xy

    def make_plot(self, show=False, levels=256):
        z = np.ma.masked_invalid(self.values)
        vmin, vmax = z.min(), z.max()
        z = z.filled(fill_value=-99999.)
        if isinstance(levels, int):
            levels = np.linspace(vmin, vmax, levels)
        plt.tricontourf(self.mpl_tri, z, levels=levels)
        plt.gca().axis('scaled')
        if show:
            plt.show()
        return plt.gca()

    def __set_Mesh(self):
        self.__Mesh = pymesh.meshio.form_mesh(self.__vertices, self.__elements)

    def __raise_nan_values(self):
        if np.isnan(self.values).any():
            raise Exception(
                'auto_ibtypes() requires that no nan values are present in the'
                + ' domain. Use fix_invalid() to pad all nan values using '
                + 'nearest neighbors or make sure that all nodes have a '
                + 'corresponding finite value.')

    @property
    def vertices(self):
        return self.__Mesh.vertices

    @property
    def elements(self):
        return self.__Mesh.elements

    @property
    def values(self):
        if not self.__Mesh.has_attribute('values'):
            self.__Mesh.add_attribute('values')
            self.__Mesh.set_attribute('values', self.__values)
        return self.__Mesh.get_attribute('values')

    @property
    def xy(self):
        return self.vertices

    @property
    def x(self):
        return self.vertices[:, 0]

    @property
    def y(self):
        return self.vertices[:, 1]

    @property
    def SpatialReference(self):
        return self._SpatialReference

    @property
    def ndim(self):
        return 2

    @property
    def num_nodes(self):
        return self.__Mesh.num_nodes

    @property
    def num_elements(self):
        return self.__Mesh.num_elements

    @property
    def node_id(self):
        if not hasattr(self, "__node_id"):
            self.__node_id = np.arange(1, len(self.values)+1)
        return self.__node_id

    @property
    def element_id(self):
        if not hasattr(self, "__element_id"):
            self.__element_id = np.arange(1, len(self.elements)+1)
        return self.__element_id

    @property
    def outer_edges(self):
        return self.PlanarStraightLineGraph.outer_edges

    @property
    def inner_edges(self):
        return self.PlanarStraightLineGraph.inner_edges

    @property
    def outer_boundary(self):
        return self.outer_edges[:, 0]

    @property
    def inner_boundary(self):
        # return self.inner_edges[:, 0]
        if not self.__Mesh.has_attribute('inner_boundary'):
            inner_boundary = list()
            for inner_edges in self.inner_edges:
                inner_boundary.append(inner_edges[:, 0])
            self.__Mesh.add_attribute('inner_boundary')
            self.__Mesh.set_attribute('inner_boundary',
                                      np.hstack(inner_boundary))
        return np.where(self.__Mesh.get_attribute('inner_boundary'))[0]

    @property
    def planar_straight_line_graph(self):
        if not hasattr(self, "__planar_straight_line_graph"):
            self.__planar_straight_line_graph = \
                self.compute_planar_straight_line_graph()
        return self.__planar_straight_line_graph

    @property
    def mpl_tri(self):
        if not hasattr(self, "__mpl_tri"):
            self.__mpl_tri = Triangulation(self.x, self.y, self.elements)
        return self.__mpl_tri

    @property
    def ocean_boundary(self):
        if not self.__Mesh.has_attribute('ocean_boundary'):
            self.__Mesh.add_attribute('ocean_boundary')
            ocean_boundary_idxs = np.where(
                self.values[self.outer_boundary] < 0.)
            values = np.full(self.values.shape, False)
            values[self.outer_boundary][ocean_boundary_idxs] = True
            self.__Mesh.set_attribute('ocean_boundary', values)
        return np.where(self.__Mesh.get_attribute('ocean_boundary'))[0]

    @property
    def land_boundary(self):
        if not self.__Mesh.has_attribute('land_boundary'):
            land_boundary_idxs = np.where(
                self.values[self.outer_boundary] >= 0.)[0]
            values = np.full(self.values.shape, False)
            values[self.outer_boundary][land_boundary_idxs] = True
            self.__Mesh.add_attribute('land_boundary')
            self.__Mesh.set_attribute('land_boundary', values)
        return np.where(self.__Mesh.get_attribute('land_boundary'))[0]

    @property
    def _vertices(self):
        return self.__vertices

    @property
    def _elements(self):
        return self.__elements

    @property
    def _values(self):
        return self.__values

    @SpatialReference.setter
    def SpatialReference(self, SpatialReference):
        if isinstance(SpatialReference, int):
            EPSG = SpatialReference
            SpatialReference = osr.SpatialReference()
            SpatialReference.ImportFromEPSG(EPSG)
        assert isinstance(SpatialReference, osr.SpatialReference)
        if not self.SpatialReference.IsSame(SpatialReference):
            self._vertices = self.get_xy(SpatialReference)
            self.__set_Mesh()
            if hasattr(self, "__PlanarStraightLineGraph"):
                self.PlanarStraightLineGraph.SpatialReference \
                    = SpatialReference
        super(Mesh, self).__init__(SpatialReference)

    @_vertices.setter
    def _vertices(self, vertices):
        vertices = np.asarray(vertices)
        assert vertices.shape[1] == self.ndim
        self.__vertices = vertices

    @_elements.setter
    def _elements(self, elements):
        elements = np.asarray(elements)
        assert elements.shape[1] == 3
        self.__elements = elements

    @_values.setter
    def _values(self, values):
        if values is not None:
            values = np.asarray(values)
            assert values.shape[0] == self._vertices.shape[0]
        else:
            values = np.full((self._vertices.shape[0],), np.nan).flatten()
        if not hasattr(self, "__Mesh"):
            self.__set_Mesh()
        if self.__Mesh.has_attribute('values'):
            self.__Mesh.remove_attribute('values')
        self.__values = values
