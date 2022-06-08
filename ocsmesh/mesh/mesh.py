"""This module defines classes that handle mesh and mesh operations.

This module defines a factory class for mesh, similar to geometry and
size function factory class. It also defines concrete mesh types.
Currently two concrete mesh types are defined for generic Eucledian
mesh and specific 2D Eucledian mesh.
"""
from functools import lru_cache
import logging
from multiprocessing import Pool, cpu_count
import os
import pathlib
from collections import defaultdict
import warnings
from typing import Union, List, Tuple, Dict, Any, Optional
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import pandas as pd
import geopandas as gpd
from jigsawpy import jigsaw_msh_t, savemsh, loadmsh, savevtk
from matplotlib.path import Path
from matplotlib.transforms import Bbox
from matplotlib.tri import Triangulation
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pyproj import CRS, Transformer
from scipy.interpolate import (
        RectBivariateSpline, RegularGridInterpolator)
from shapely.geometry import (
        LineString, box, Polygon, MultiPolygon)
from shapely.ops import polygonize, linemerge


from ocsmesh import utils
from ocsmesh.raster import Raster
from ocsmesh.mesh.base import BaseMesh
from ocsmesh.mesh.parsers import grd, sms2dm


_logger = logging.getLogger(__name__)



class EuclideanMesh(BaseMesh):
    """Generic Euclidean mesh class

    This is the base class for 2D or 3D Euclidean mesh.

    Attributes
    ----------
    tria3 : npt.NDArray[jigsaw_msh_t.TRIA3_t]
        Reference to underlying jigsaw mesh's triangle element
        structure.
    triangles : npt.NDArray[np.float32]
        Array of node index for triangular elements.
    quad4 : npt.NDArray[jigsaw_msh_t.QUAD4_t]
        Reference to underlying jigsaw mesh's quadrangle element
        structure.
    quads : npt.NDArray[np.float32]
        Array of node index for quadrangular elements.
    crs : CRS
        Coodrinate reference system of the mesh object
    hull : Hull
        Handle to hull calculation helper object
    nodes : Nodes
        Handle to node handler helper object
    elements : Elements
        Handle to element handler helper object

    Methods
    -------
    write(path, overwrite=False, format='grd')
        Export mesh object to the disk in the specified format.
    """

    def __init__(self, mesh: jigsaw_msh_t) -> None:
        """Initialize Euclidean mesh object.

        Parameters
        ----------
        mesh : jigsaw_msh_t
            The underlying jigsaw_msh_t object to hold onto mesh data.

        Raises
        ------
        TypeError
            If input mesh is not of `jigsaw_msh_t` type.
        ValueError
            If input mesh's `mshID` is not equal to ``euclidean-mesh``.
            If input mesh has `crs` property which is not of `CRS` type.
        """

        if not isinstance(mesh, jigsaw_msh_t):
            raise TypeError(f'Argument mesh must be of type {jigsaw_msh_t}, '
                            f'not type {type(mesh)}.')
        if mesh.mshID != 'euclidean-mesh':
            raise ValueError(f'Argument mesh has property mshID={mesh.mshID}, '
                             "but expected 'euclidean-mesh'.")
        if not hasattr(mesh, 'crs'):
            warnings.warn('Input mesh has no CRS information.')
            mesh.crs = None
        else:
            if not isinstance(mesh.crs, CRS):
                raise ValueError(f'crs property must be of type {CRS}, not '
                                 f'type {type(mesh.crs)}.')

        self._hull = None
        self._nodes = None
        self._elements = None
        self._msh_t = mesh

    def write(
            self,
            path: Union[str, os.PathLike],
            overwrite: bool = False,
            format : Literal['grd', '2dm', 'msh', 'vtk'] = 'grd', # pylint: disable=W0622
            ) -> None:
        """Export the mesh object to the disk

        Parameters
        ----------
        path : path-like
            Path to which the mesh should be exported.
        overwrite : bool, default=False
            Whether to overwrite, if a file already exists in `path`
        format : { 'grd', '2dm', 'msh', 'vtk' }
            Format of the export, SMS-2DM or GRD.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If specified export format is **not** supported.
        """

        path = pathlib.Path(path)
        if path.exists() and overwrite is not True:
            raise IOError(
                f'File {str(path)} exists and overwrite is not True.')
        if format == 'grd':
            grd_dict = utils.msh_t_to_grd(self.msh_t)
            if self._boundaries and self._boundaries.data:
                grd_dict.update(boundaries=self._boundaries.data)
            grd.write(grd_dict, path, overwrite)

        elif format == '2dm':
            sms2dm.writer(utils.msh_t_to_2dm(self.msh_t), path, overwrite)

        elif format == 'msh':
            savemsh(str(path), self.msh_t)

        elif format == 'vtk':
            savevtk(str(path), self.msh_t)

        else:
            raise ValueError(f'Unhandled format {format}.')

    @property
    def tria3(self):
        """Reference to underlying mesh tirangle element structure"""

        return self.msh_t.tria3

    @property
    def triangles(self):
        """Reference to underlying mesh triangle element index array"""

        return self.msh_t.tria3['index']

    @property
    def quad4(self):
        """Reference to underlying mesh quadrangle element structure"""

        return self.msh_t.quad4

    @property
    def quads(self):
        """Reference to underlying mesh quadrangle element index array"""

        return self.msh_t.quad4['index']

    @property
    def crs(self):
        """Reference to underlying mesh crs"""

        return self.msh_t.crs

    @property
    def hull(self):
        """Reference to hull calculator helper object"""

        if self._hull is None:
            self._hull = Hull(self)
        return self._hull

    @property
    def nodes(self):
        """Reference to node handler helper object"""

        if self._nodes is None:
            self._nodes = Nodes(self)
        return self._nodes

    @property
    def elements(self):
        """Reference to element handler helper object"""

        if self._elements is None:
            self._elements = Elements(self)
        return self._elements


class EuclideanMesh2D(EuclideanMesh):
    """2D Euclidean mesh definition

    Attributes
    ----------
    boundaries
    vert2
    value
    bbox

    Methods
    -------
    get_bbox(crs=None, output_type=None)
        Gets the bounding box of the mesh elements.
    tricontourf(**kwargs)
        Create a contour plot from the value data on the nodes of
        the mesh
    interpolate(raster, method='spline', nprocs=None)
        Interpolate raster date on the nodes.
    get_contour(level)
        Get contour lines from node value data at specified levels.
    get_multipolygon(zmin=None, zmax=None)
        Get multipolygon of the mesh hull.
    """

    def __init__(self, mesh: jigsaw_msh_t) -> None:
        """Initialize Euclidean 2D mesh object.

        Parameters
        ----------
        mesh : jigsaw_msh_t
            The underlying jigsaw_msh_t object to hold onto mesh data.

        Raises
        ------
        ValueError
            If number of mesh dimensions is not equal to ``2``.
        """

        super().__init__(mesh)
        self._boundaries = None

        if mesh.ndims != +2:
            raise ValueError(f'Argument mesh has property ndims={mesh.ndims}, '
                             "but expected ndims=2.")

        if len(self.msh_t.value) == 0:
            self.msh_t.value = np.array(
                np.full((self.vert2['coord'].shape[0], 1), np.nan))

    def get_bbox(
            self,
            crs: Union[str, CRS, None] = None,
            output_type: Literal[None, 'polygon', 'bbox'] = None
            ) -> Union[Polygon, Bbox]:
        """Get the bounding box of mesh elements.

        Parameters
        ----------
        crs : str or CRS or None, default=None
            CRS to transform the calculated bounding box into before
            returning
        output_type : { None, 'polygon', 'bbox'}, default=None
            Output type

        Returns
        -------
        Polygon or Bbox
            Bounding box of the mesh elements.
        """

        output_type = 'polygon' if output_type is None else output_type
        xmin, xmax = np.min(self.coord[:, 0]), np.max(self.coord[:, 0])
        ymin, ymax = np.min(self.coord[:, 1]), np.max(self.coord[:, 1])
        crs = self.crs if crs is None else crs
        if crs is not None:
            if not self.crs.equals(crs):
                transformer = Transformer.from_crs(
                    self.crs, crs, always_xy=True)
                # pylint: disable=E0633
                (xmin, xmax), (ymin, ymax) = transformer.transform(
                    (xmin, xmax), (ymin, ymax))
        if output_type == 'polygon': # pylint: disable=R1705
            return box(xmin, ymin, xmax, ymax)
        elif output_type == 'bbox':
            return Bbox([[xmin, ymin], [xmax, ymax]])

        raise TypeError(
            'Argument output_type must a string literal \'polygon\' or '
            '\'bbox\'')

    @property
    def boundaries(self):
        """Handle to boundaries calculator helper object"""

        if self._boundaries is None:
            self._boundaries = Boundaries(self)
        return self._boundaries

    def tricontourf(self, **kwargs) -> Axes:
        """Generate contour for the data of triangular elements of the mesh

        Parameters
        ----------
        **kwargs : dict, optional
           Passed to underlying `matplotlib` API.

        Returns
        -------
        Axes
            Axes on which the filled contour is drawn.
        """

        return utils.tricontourf(self.msh_t, **kwargs)

    def interpolate(
            self,
            raster: Union[Raster, List[Raster]],
            method: Literal['spline', 'linear', 'nearest'] = 'spline',
            nprocs: Optional[int] = None,
            info_out_path: Union[pathlib.Path, str, None] = None,
            filter_by_shape: bool = False
            ) -> None:
        """Interplate values from raster inputs to the mesh nodes.

        Parameters
        ----------
        raster : Raster or list of Raster
            A single or a list of rasters from which values are
            interpolated onto the mesh
        method : {'spline', 'linear', 'nearest'}, default='spline'
            Method of interpolation.
        nprocs : int or None, default=None
            Number of workers to use when interpolating data.
        info_out_path : pathlike or str or None
            Path for the output node interpolation information file
        filter_by_shape : bool
            Flag for node filtering based on raster bbox or shape

        Returns
        -------
        None
        """

        if isinstance(raster, Raster):
            raster = [raster]

        nprocs = -1 if nprocs is None else nprocs
        nprocs = cpu_count() if nprocs == -1 else nprocs

        # Fix an issue on Jupyter notebook where having pool execute
        # interpolation even in case of nprocs == 1 would results in
        # application getting stuck
        if nprocs > 1:
            with Pool(processes=nprocs) as pool:
                res = pool.starmap(
                    _mesh_interpolate_worker,
                    [(self.vert2['coord'], self.crs,
                        _raster.tmpfile, _raster.chunk_size,
                        method, filter_by_shape)
                     for _raster in raster]
                    )
            pool.join()
        else:
            res = [_mesh_interpolate_worker(
                        self.vert2['coord'], self.crs,
                        _raster.tmpfile, _raster.chunk_size,
                        method, filter_by_shape)
                   for _raster in raster]

        values = self.msh_t.value.flatten()

        interp_info_map = {}
        for (mask, _values), rast in zip(res, raster):
            values[mask] = _values

            if info_out_path is not None:
                vert_cs = None
                rast_crs = rast.crs
                if rast_crs.is_vertical:
                    if rast_crs.sub_crs_list is not None:
                        for sub_crs in rast_crs.sub_crs_list:
                            if sub_crs.is_vertical:
                                # TODO: What if sub CRS is compound, etc.?
                                vert_cs = sub_crs
                    elif rast_crs.source_crs is not None:
                        if rast_crs.source_crs.is_vertical:
                            # TODO: What if source CRS is compound, etc.?
                            vert_cs = rast_crs.source_crs


                vert_cs_name = vert_cs.name
                idxs = np.argwhere(mask).ravel()
                interp_info_map.update({
                    idx: (rast.path, vert_cs_name)
                    for idx in idxs})

        if info_out_path is not None:
            coords = self.msh_t.vert2['coord'].copy()
            geo_coords = coords.copy()
            if not self.crs.is_geographic:
                transformer = Transformer.from_crs(
                    self.crs, CRS.from_epsg(4326), always_xy=True)
                # pylint: disable=E0633
                geo_coords[:, 0], geo_coords[:, 1] = transformer.transform(
                    coords[:, 0], coords[:, 1])
            vd_idxs=np.array(list(interp_info_map.keys()))
            df_interp_info = pd.DataFrame(
                index=vd_idxs,
                data={
                    'x': coords[vd_idxs, 0],
                    'y': coords[vd_idxs, 1],
                    'lat': geo_coords[vd_idxs, 0],
                    'lon': geo_coords[vd_idxs, 1],
                    'elev': values[vd_idxs],
                    'crs': [i[1] for i in interp_info_map.values()],
                    'source': [i[0] for i in interp_info_map.values()]
                }
            )
            df_interp_info.sort_index().to_csv(
                info_out_path, header=False, index=True)


        self.msh_t.value = np.array(values.reshape((values.shape[0], 1)),
                                    dtype=jigsaw_msh_t.REALS_t)


    def get_contour(self, level: float) -> LineString:
        """Extract contour lines at the specified `level` from mesh values

        Parameters
        ----------
        level : float
            The level at which contour lines must be extracted.

        Returns
        -------
        LineString
            Extracted and merged contour lines.

        Raises
        ------
        ValueError
            If mesh has nodes that have null value `np.nan`.
        """

        # ONLY SUPPORTS TRIANGLES
        for attr in ['quad4', 'hexa8']:
            if len(getattr(self.msh_t, attr)) > 0:
                warnings.warn(
                    'Mesh contour extraction only supports triangles')

        coords = self.msh_t.vert2['coord']
        values = self.msh_t.value
        trias = self.msh_t.tria3['index']
        if np.any(np.isnan(values)):
            raise ValueError(
                "Mesh contains invalid values. Raster values must"
                "be interpolated to the mesh before generating "
                "boundaries.")

        x, y = coords[:, 0], coords[:, 1]
        features = []
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            _logger.debug('Computing contours...')
            fig, ax = plt.subplots()
            ax.tricontour(
                x, y, trias, values.ravel(), levels=[level])
            plt.close(fig)
        for path_collection in ax.collections:
            for path in path_collection.get_paths():
                try:
                    features.append(LineString(path.vertices))
                except ValueError:
                    # LineStrings must have at least 2 coordinate tuples
                    pass
        return linemerge(features)


    def get_multipolygon(
            self,
            zmin: Optional[float] = None,
            zmax: Optional[float] = None
            ) -> MultiPolygon:
        """Calculate multipolygon covering mesh elements (hull)

        Parameters
        ----------
        zmin : float or None
            Minimum elevation to consider for multipolygon extraction
        zmax : float or None
            Maximum elevation to consider for multipolygon extraction

        Returns
        -------
        MultiPolygon
            Calculated multipolygon shape
        """

        values = self.msh_t.value
        mask = np.ones(values.shape)
        if zmin is not None:
            mask = np.logical_and(mask, values > zmin)
        if zmax is not None:
            mask = np.logical_and(mask, values < zmax)

        # Assuming value is of shape (N, 1)
        # ravel to make sure it's 1D
        verts_in = np.argwhere(mask).ravel()

        clipped_mesh = utils.clip_mesh_by_vertex(
            self.msh_t, verts_in,
            can_use_other_verts=True)

        boundary_edges = utils.get_boundary_edges(clipped_mesh)
        coords = clipped_mesh.vert2['coord']
        coo_to_idx = {
            tuple(coo): idx
            for idx, coo in enumerate(coords)}
        poly_gen = polygonize(coords[boundary_edges])
        polys = list(poly_gen)
        polys = sorted(polys, key=lambda p: p.area, reverse=True)

        rings = [p.exterior for p in polys]
        n_parents = np.zeros((len(rings),))
        represent = np.array([r.coords[0] for r in rings])
        for e, ring in enumerate(rings[:-1]):
            path = Path(ring.coords, closed=True)
            n_parents = n_parents + np.pad(
                np.array([
                    path.contains_point(pt) for pt in represent[e+1:]]),
                (e+1, 0), 'constant', constant_values=0)

        # Get actual polygons based on logic described above
        polys = [p for e, p in enumerate(polys) if not n_parents[e] % 2]

        return MultiPolygon(polys)

    @property
    def vert2(self):
        """Reference to underlying mesh 2D vertices structure"""
        return self.msh_t.vert2

    @property
    def value(self):
        """Reference to underlying mesh values"""
        return self.msh_t.value

    @property
    def bbox(self):
        """Calculates and returns bounding box of the mesh hull.

        See Also
        --------
        get_bbox
        """
        return self.get_bbox()

MeshType = Union[EuclideanMesh2D]

class Mesh(BaseMesh):
    """Mesh object factory

    Factory class that creates and returns concrete mesh object
    based on the input types.

    Methods
    -------
    open(path, crs=None)
        Read mesh data from a file on disk.
    """

    def __new__(cls, mesh: jigsaw_msh_t) -> MeshType:
        """Construct a concrete mesh object.

        Parameters
        ----------
        mesh : jigsaw_msh_t
            Input jigsaw mesh object

        Returns
        -------
        MeshType
            Mesh object created from the input

        Raises
        ------
        TypeError
            Input `mesh` is not a `jigsaw_msh_t` object.
        NotImplementedError
            Input `mesh` object cannot be used to create a EuclideanMesh2D
        """

        if not isinstance(mesh, jigsaw_msh_t):
            raise TypeError(f'Argument mesh must be of type {jigsaw_msh_t}, '
                            f'not type {type(mesh)}.')

        if mesh.mshID == 'euclidean-mesh':
            if mesh.ndims == 2:
                return EuclideanMesh2D(mesh)

            raise NotImplementedError(
                f'mshID={mesh.mshID} + mesh.ndims={mesh.ndims} not '
                'handled.')

        raise NotImplementedError(f'mshID={mesh.mshID} not handled.')

    @staticmethod
    def open(path: Union[str, Path], crs: Optional[CRS] = None) -> MeshType:
        """Read mesh from a file on disk

        Parameters
        ----------
        path : path-like
            Path to the file containig mesh.
        crs : CRS or None, default=None
            CRS of the mesh in the path. Overwrites any info read
            from file, no transformation is done.

        Returns
        -------
        MeshType
            Mesh object created by reading the file.

        Raises
        ------
        TypeError
           If cannot determine the input mesh type.

        Notes
        -----
        Currently only SMS-2DM and GRD formats are supported for
        reading.
        """

        try:
            msh_t = utils.grd_to_msh_t(grd.read(path, crs=crs))
            msh_t.value = np.negative(msh_t.value)
            return Mesh(msh_t)
        except Exception as e: #pylint: disable=W0703
            if 'not a valid grd file' in str(e):
                pass
            else:
                raise e

        try:
            return Mesh(utils.sms2dm_to_msh_t(sms2dm.read(path, crs=crs)))
        except ValueError:
            pass

        try:
            msh_t = jigsaw_msh_t()
            loadmsh(msh_t, path)
            msh_t.crs = crs
            return Mesh(msh_t)
        except Exception as e: #pylint: disable=W0703
            pass

        raise TypeError(
            f'Unable to automatically determine file type for {str(path)}.')


class Rings:
    """Helper class for handling mesh rings.

    This is a helper class to manage the calculation of internal
    and external rings of the mesh polygon or hull.

    Attributes
    ----------

    Methods
    -------
    __call__()
        Returns all rings of the mesh hull
    interior()
        Return the interior rings of the mesh hull
    exterior()
        Return the exterior rings of the mesh hull
    """

    def __init__(self, mesh: EuclideanMesh) -> None:
        """Initializes the ring calculator object for the input `mesh`

        Parameters
        ----------
        mesh : EuclideanMesh
            Input mesh for which this object calculates rings.
        """

        self.mesh = mesh

    @lru_cache(maxsize=1)
    def __call__(self) -> gpd.GeoDataFrame:
        """Calcluates all the polygons of the mesh and extracts its rings.

        Parameters
        ----------

        Returns
        -------
        gpd.GeoDataFrame
            Dataframe containing all rings of the mesh hull polygon.
            The rings are in the form of `shapely.geometry.LinearRing`.

        Notes
        -----
        The result of this method is cached, so that multiple calls
        to it won't result in multiple calculations. If the mesh
        is modified and the cache is not properly clear the calls
        to this method can result in invalid return values.
        """

        polys = utils.get_mesh_polygons(self.mesh.msh_t)

        data = []
        bnd_id = 0
        for poly in polys:
            data.append({
                    "geometry": poly.exterior,
                    "bnd_id": bnd_id,
                    "type": 'exterior'
                })
            for interior in poly.interiors:
                data.append({
                    "geometry": interior,
                    "bnd_id": bnd_id,
                    "type": 'interior'
                })
            bnd_id = bnd_id + 1
        return gpd.GeoDataFrame(data, crs=self.mesh.crs)

    def exterior(self) -> gpd.GeoDataFrame:
        """Extracts the exterior ring from the results of `__call__`.

        Parameters
        ----------

        Returns
        -------
        gpd.GeoDataFrame
            Dataframe containing exterior ring of the mesh hull polygon.
        """

        return self().loc[self()['type'] == 'exterior']

    def interior(self) -> gpd.GeoDataFrame:
        """Extracts the interior rings from the results of `__call__`.

        Parameters
        ----------

        Returns
        -------
        gpd.GeoDataFrame
            Dataframe containing interior rings of the mesh hull polygon.
        """

        return self().loc[self()['type'] == 'interior']


class Edges:
    """Helper class for handling mesh boundary edges.

    Attributes
    ----------

    Methods
    -------
    __call__()
        Return all boundary edges of the mesh hull
    interior()
        Return the interior boundary edges of the mesh hull
    exterior()
        Return the exterior boundary edges of the mesh hull
    """

    def __init__(self, mesh: EuclideanMesh) -> None:
        """Initializes the edge calculator object for the input `mesh`

        Parameters
        ----------
        mesh : EuclideanMesh
            Input mesh for which boundary edges are calculated.
        """

        self.mesh = mesh

    @lru_cache(maxsize=1)
    def __call__(self) -> gpd.GeoDataFrame:
        """Calculates all boundary edges for the mesh.

        Parameters
        ----------

        Returns
        -------
        gpd.GeoDataFrame
            Dataframe containing all boundary edges of the mesh in
            the form of `shapely.geometry.LineString` for each
            coordinate couple.

        Notes
        -----
        The result of this method is cached, so that multiple calls
        to it won't result in multiple calculations. If the mesh
        is modified and the cache is not properly clear the calls
        to this method can result in invalid return values.
        """

        data = []
        for ring in self.mesh.hull.rings().itertuples():
            coords = ring.geometry.coords
            for i in range(1, len(coords)):
                data.append({
                    "geometry": LineString([coords[i-1], coords[i]]),
                    "bnd_id": ring.bnd_id,
                    "type": ring.type})

        return gpd.GeoDataFrame(data, crs=self.mesh.crs)

    def exterior(self) -> gpd.GeoDataFrame:
        """Retruns exterior boundary edges from the results of `__call__`

        Parameters
        ----------

        Returns
        -------
        gpd.GeoDataFrame
            Dataframe containing exterior boundary edges of the mesh in
            the form of line string couples.
        """

        return self().loc[self()['type'] == 'exterior']

    def interior(self) -> gpd.GeoDataFrame:
        """Retruns interior boundary edges from the results of `__call__`

        Parameters
        ----------

        Returns
        -------
        gpd.GeoDataFrame
            Dataframe containing interior boundary edges of the mesh in
            the form of line string couples.
        """

        return self().loc[self()['type'] == 'interior']


class Hull:
    """Helper class for handling mesh hull calculations.

    This class wraps the functionality of ring and edge classes and
    adds additional methods to calculate or extract the polygon or
    triangulation of the mesh

    Attributes
    ----------

    Methods
    -------
    __call__()
        Calculates all the polys from all mesh rings
    exterior()
        Calculates the exterior rings of the mesh hull.
    interior()
        Calculates the interior rings of the mesh hull.
    implode()
        Calculates all the polygons (including isolated domain
        islands) in the mesh and returns a table of polygons.
    multipolygon()
        Calculates all the polygons (including isolated domain
        islands) in the mesh and returns a multipolygon.
    triangulation()
        Calcluates a triangulation from the triangles and quads of
        the mesh.
    """

    def __init__(self, mesh: EuclideanMesh) -> None:
        """Initialize helper class for handling mesh hull calculations

        Parameters
        ----------
        mesh : EuclideanMesh
            Input mesh for which hull calculations are done.

        Notes
        -----
        This object holds onto the ring and edge calculator objects
        as well as a reference to the input mesh.
        """

        self.mesh = mesh
        self.rings = Rings(mesh)
        self.edges = Edges(mesh)

    @lru_cache(maxsize=1)
    def __call__(self) -> gpd.GeoDataFrame:
        """Calculates all polygons of the mesh including domain islands

        Parameters
        ----------

        Returns
        -------
        gpd.GeoDataFrame
            Dataframe containing all polygons of the mesh.

        See Also
        --------
        implode()
            Dataframe with a single combined multipolygon.
        multipolygon()
            `shapely` multipolygon shape of combined mesh polygons.

        Notes
        -----
        The result of this method is cached, so that multiple calls
        to it won't result in multiple calculations. If the mesh
        is modified and the cache is not properly clear the calls
        to this method can result in invalid return values.
        """

        data = []
        for bnd_id in np.unique(self.rings()['bnd_id'].tolist()):
            exterior = self.rings().loc[
                (self.rings()['bnd_id'] == bnd_id) &
                (self.rings()['type'] == 'exterior')]
            interiors = self.rings().loc[
                (self.rings()['bnd_id'] == bnd_id) &
                (self.rings()['type'] == 'interior')]
            data.append({
                    "geometry": Polygon(
                        exterior.iloc[0].geometry.coords,
                        [row.geometry.coords for _, row
                            in interiors.iterrows()]),
                    "bnd_id": bnd_id
                })
        return gpd.GeoDataFrame(data, crs=self.mesh.crs)

    def exterior(self) -> gpd.GeoDataFrame:
        """Creates polygons from exterior rings of the mesh hull

        Parameters
        ----------

        Returns
        -------
        gpd.GeoDataFrame
            Polygons created from exterior rings of the mesh hull
        """
        data = []
        for exterior in self.rings().loc[
                self.rings()['type'] == 'exterior'].itertuples():
            data.append({"geometry": Polygon(exterior.geometry.coords)})
        return gpd.GeoDataFrame(data, crs=self.mesh.crs)

    def interior(self) -> gpd.GeoDataFrame:
        """Creates polygons from interior rings of the mesh hull

        Parameters
        ----------

        Returns
        -------
        gpd.GeoDataFrame
            Polygons created from interior rings of the mesh hull
        """
        data = []
        for interior in self.rings().loc[
                self.rings()['type'] == 'interior'].itertuples():
            data.append({"geometry": Polygon(interior.geometry.coords)})
        return gpd.GeoDataFrame(data, crs=self.mesh.crs)

    def implode(self) -> gpd.GeoDataFrame:
        """Creates a dataframe from mesh polygons.

        Parameters
        ----------

        Returns
        ------
        gpd.GeoDataFrame
            Dataframe containing polygons of the mesh.

        See Also
        --------
        __call__()
            Dataframe with multiple polygon and boundary ID entries
            of the mesh polygons.
        multipolygon()
            `shapely` multipolygon shape of combined mesh polygons.

        Notes
        -----
        The difference of the return value of this method and
        `__call__` is that the `implode` returns a dataframe with
        a single `MultiPolygon` where as `__call__` returns a
        dataframe with multiple `Polygon` entries with associated
        `bnd_id`.
        """

        return gpd.GeoDataFrame(
            {"geometry": MultiPolygon([polygon.geometry for polygon
                                       in self().itertuples()])},
            crs=self.mesh.crs)

    def multipolygon(self) -> MultiPolygon:
        """Returns mesh multi-polygons.

        Parameters
        ----------

        Returns
        ------
        MultiPolygon
            Combined shape of polygons of the mesh.

        See Also
        --------
        __call__()
            Dataframe with multiple polygon and boundary ID entries
            of the mesh polygons.
        implode()
            Dataframe with a single combined multipolygon of the mesh
            polygons.

        Notes
        -----
        The difference of the return value of this method and `implode`
        is that `multipolygon` returns a `MultiPolygon` object where
        as `implode` returns a dataframe warpping the multipolygon
        object.
        """

        mp = self.implode().iloc[0].geometry
        if isinstance(mp, Polygon):
            mp = MultiPolygon([mp])
        return mp

    def triangulation(self) -> Triangulation:
        """Create triangulation object from all the mesh elements.

        Parameters
        ----------

        Returns
        -------
        Triangulation
            The `matplotlib` triangulation object create from all
            the elements of the parent mesh.

        Notes
        -----
        Currently only tria3 and quad4 elements are considered.
        """

        triangles = self.mesh.msh_t.tria3['index'].tolist()
        for quad in self.mesh.msh_t.quad4['index']:
            triangles.extend([
                [quad[0], quad[1], quad[3]],
                [quad[1], quad[2], quad[3]]
            ])
        return Triangulation(self.mesh.coord[:, 0], self.mesh.coord[:, 1], triangles)



class Nodes:
    """Helper class for handling mesh nodes.

    Attributes
    ----------
    id_to_index : dict
        Mapping to convert node IDs to node indexes.
    index_to_id : dict
        Mapping to convert node indexes to node IDs.

    Methods
    -------
    __call__()
        Creates a mapping between node IDs (index + 1) and node
        coordinates
    id()
        Returns list of node IDs.
    index()
        Return array of node indices.
    coords()
        Return mesh coordinates.
    values()
        Return values stored for mesh nodes.
    get_index_by_id(node_id)
        Get the node index based on node ID.
    get_id_by_index(index)
        Get the node ID based on the node index.
    """

    def __init__(self, mesh: EuclideanMesh) -> None:
        """Initializes node handler helper object.

        Parameters
        ----------
        mesh : EuclideanMesh
            Input mesh for which this object handles nodes info.
        """

        self.mesh = mesh
        self._id_to_index = None
        self._index_to_id = None

    @lru_cache(maxsize=1)
    def __call__(self) -> Dict[int, int]:
        """Creates a mapping between node IDs and indexes.

        Parameters
        ----------

        Returns
        -------
        dict
            Mapping between node IDs and indexes.

        Notes
        -----
        The result of this method is cached, so that multiple calls
        to it won't result in multiple calculations. If the mesh
        is modified and the cache is not properly clear the calls
        to this method can result in invalid return values.
        """

        return {i+1: coord for i, coord in enumerate(self.coords())}

    def id(self) -> List[int]:
        """Retrives a list of element IDs.

        Parameters
        ----------

        Returns
        -------
        list of int
            List of node IDs as created by `__call__`
        """

        return list(self().keys())

    def index(self) -> npt.NDArray[int]:
        """Retrives an array of element indexes.

        Parameters
        ----------

        Returns
        -------
        array-like
            Array of node indexes.
        """

        return np.arange(len(self()))

    def coords(self) -> npt.NDArray[np.float32]:
        """Retrieve the coordinates of mesh nodes

        Parameters
        ----------

        Returns
        -------
        array-like
            Coordinates of the mesh nodes as returned by `BaseMesh.coord`
        """

        return self.mesh.coord

    def values(self):
        """Retrieve the values stored for mesh nodes

        Parameters
        ----------

        Returns
        -------
        array-like
            Values on the mesh nodes as returned by `BaseMesh.values`
        """

        return self.mesh.values

    def get_index_by_id(self, node_id):
        """Converts mesh ID to mesh index.

        Parameters
        ----------
        node_id : int
            ID of the node of interest

        Returns
        -------
        int
            Index of the node of interest
        """

        return self.id_to_index[node_id]

    def get_id_by_index(self, index: int):
        """Converts mesh index to mesh ID.

        Parameters
        ----------
        index : int
            Index of the node of interest.

        Returns
        -------
        int
            ID of the node of interest
        """

        return self.index_to_id[index]

    @property
    def id_to_index(self) -> Dict[int, int]:
        """Read-only property returning the mapping of ID to index

        Notes
        -----
        Although the property is read-only, the return value object
        is a cached mutable dictionary object. Modifying the mesh
        without clearing the cache properly or mutating the
        returned object could result in undefined behavior
        """

        if self._id_to_index is None:
            self._id_to_index = {node_id: index for index, node_id
                                 in enumerate(self().keys())}
        return self._id_to_index

    @property
    def index_to_id(self) -> Dict[int, int]:
        """Read-only property returning the mapping of index to ID

        Notes
        -----
        Although the property is read-only, the return value object
        is a cached mutable dictionary object. Modifying the mesh
        without clearing the cache properly or mutating the
        returned object could result in undefined behavior
        """

        if self._index_to_id is None:
            self._index_to_id = dict(enumerate(self().keys()))
        return self._index_to_id

    # def get_indexes_around_index(self, index):
    #     indexes_around_index = self.__dict__.get('indexes_around_index')
    #     if indexes_around_index is None:
    #         def append(geom):
    #             for simplex in geom:
    #                 for i, j in permutations(simplex, 2):
    #                     indexes_around_index[i].add(j)
    #         indexes_around_index = defaultdict(set)
    #         append(self.gr3.elements.triangles())
    #         append(self.gr3.elements.quads())
    #         self.__dict__['indexes_around_index'] = indexes_around_index
    #     return list(indexes_around_index[index])


class Elements:
    """Helper class for handling mesh elements.

    Attributes
    ----------

    Methods
    --------
    __call__()
        Creates a mapping between element IDs and associated node IDs.
    id()
        Returns a list of element IDs.
    index()
        Returns an array of element indexes.
    array()
        Creates and returns a masked array of element node indices.
    triangles()
        Creates and returns a 2D array of triangular element node indices.
    quads()
        Creates and returns a 2D array of quadrangular element node indices.
    triangulation()
        Calcluates a triangulation from the triangles and quads of
        the mesh.
    geodataframe()
        Creates and returns a dataframe of with polygon entires for
        each element.
    """

    def __init__(self, mesh: EuclideanMesh) -> None:
        """Initialize the element handler helper object.

        Parameters
        ----------
        mesh : EuclideanMesh
            Input mesh for which this object handles elements info.
        """

        self.mesh = mesh

    @lru_cache(maxsize=1)
    def __call__(self) -> Dict[int, npt.NDArray[int]]:
        """Creates a mapping between element IDs and associated node IDs.

        Parameters
        ----------

        Returns
        -------
        dict
            Mapping between element IDs and associated node Ids

        Notes
        -----
        The result of this method is cached, so that multiple calls
        to it won't result in multiple calculations. If the mesh
        is modified and the cache is not properly clear the calls
        to this method can result in invalid return values.
        """

        elements = {i+1: index+1 for i, index
                    in enumerate(self.mesh.msh_t.tria3['index'])}
        elements.update({i+len(elements)+1: index+1 for i, index
                         in enumerate(self.mesh.msh_t.quad4['index'])})
        return elements

    @lru_cache(maxsize=1)
    def id(self) -> List[int]:
        """Retrieves the list of element IDs as returned by `__call__`

        Parameters
        ----------

        Returns
        -------
        list of int
            List of element IDs.

        Notes
        -----
        The result of this method is cached, so that multiple calls
        to it won't result in multiple calculations. If the mesh
        is modified and the cache is not properly clear the calls
        to this method can result in invalid return values.
        """

        return list(self().keys())

    @lru_cache(maxsize=1)
    def index(self) -> npt.NDArray[int]:
        """Retrieves an array of element indices

        Parameters
        ----------

        Returns
        -------
        npt.NDArray
            1D array of element indices.

        Notes
        -----
        The result of this method is cached, so that multiple calls
        to it won't result in multiple calculations. If the mesh
        is modified and the cache is not properly clear the calls
        to this method can result in invalid return values.
        """

        return np.arange(len(self()))

    def array(self) -> npt.NDArray[int]:
        """Retrieves a masked array of element node IDs.

        The return value is ``n x m`` where ``n`` is the number of
        elements and ``m`` is the maximum number of element nodes, e.g.
        if there are only trias, then it's 3, for trias and quads it
        is 4.

        Parameters
        ----------

        Returns
        -------
        npt.NDArray
            Masked array where elements with fewer associated nodes
            have trailing masked node columns in the array.
        """

        rank = int(max(map(len, self().values())))
        array = np.full((len(self()), rank), -1)
        for i, elem_nd_ids in enumerate(self().values()):
            row = np.array(list(map(self.mesh.nodes.get_index_by_id, elem_nd_ids)))
            array[i, :len(row)] = row
        return np.ma.masked_equal(array, -1)

    @lru_cache(maxsize=1)
    def triangles(self) -> npt.NDArray[int]:
        """Retrieves an array of tria element node indices

        Parameters
        ----------

        Returns
        -------
        npt.NDArray
            2D array of element nodes for triangle nodes

        Notes
        -----
        The result of this method is cached, so that multiple calls
        to it won't result in multiple calculations. If the mesh
        is modified and the cache is not properly clear the calls
        to this method can result in invalid return values.
        """

        return np.array(
            [list(map(self.mesh.nodes.get_index_by_id, element))
             for element in self().values()
             if len(element) == 3])

    @lru_cache(maxsize=1)
    def quads(self):
        """Retrieves an array of quad element node indices

        Parameters
        ----------

        Returns
        -------
        npt.NDArray
            2D array of element nodes for quadrangle nodes

        Notes
        -----
        The result of this method is cached, so that multiple calls
        to it won't result in multiple calculations. If the mesh
        is modified and the cache is not properly clear the calls
        to this method can result in invalid return values.
        """

        return np.array(
            [list(map(self.mesh.nodes.get_index_by_id, element))
             for element in self().values()
             if len(element) == 4])

    def triangulation(self) -> Triangulation:
        """Create triangulation object from all the mesh elements.

        Parameters
        ----------

        Returns
        -------
        Triangulation
            The `matplotlib` triangulation object create from all
            the elements of the parent mesh.

        Notes
        -----
        Currently only tria3 and quad4 elements are considered.
        """

        triangles = self.triangles().tolist()
        for quad in self.quads():
            # TODO: Not tested.
            triangles.append([quad[0], quad[1], quad[3]])
            triangles.append([quad[1], quad[2], quad[3]])
        return Triangulation(
            self.mesh.coord[:, 0],
            self.mesh.coord[:, 1],
            triangles)

    def geodataframe(self) -> gpd.GeoDataFrame:
        """Create polygons for each element and return in dataframe

        Parameters
        ----------

        Returns
        -------
        gpd.GeoDataFrame
            Dataframe created from entries of `Polygon` type for
            each element.
        """

        data = []
        for elem_id, elem_nd_ids in self().items():
            data.append({
                'geometry': Polygon(
                    self.mesh.coord[list(
                        map(self.mesh.nodes.get_index_by_id, elem_nd_ids))]),
                'id': elem_id})
        return gpd.GeoDataFrame(data, crs=self.mesh.crs)


class Boundaries:
    """Helper class for mesh boundary condition calculation

    Attributes
    ----------
    data : dict
        Mapping for boundary information

    Methods
    -------
    __call__()
        Retrieves a dataframe for all boundary shapes and type info.
    __len__()
        Gets the number of calculated boundary segments.
    ocean()
        Retrieves a dataframe containing shapes and type info of ocean
        boundaries
    land()
        Retrieves a dataframe containing shapes and type info of land
        boundaries
    interior()
        Retrieves a dataframe containing shapes and type info of island
        boundaries
    auto_generate(threshold=0., land_ibtype=0, interior_ibtype=1)
        Automatically generate boundary information based on the
        input land indicator `threshold`
    """

    def __init__(self, mesh: EuclideanMesh) -> None:
        """Initialize boundary helper object

        Parameters
        ----------
        mesh : EuclideanMesh
            Input mesh for which this object calculates boundaries.
        """

        # TODO: Add a way to manually initialize
        self.mesh = mesh
        self._ocean = gpd.GeoDataFrame()
        self._land = gpd.GeoDataFrame()
        self._interior = gpd.GeoDataFrame()
        self._data = defaultdict(defaultdict)

    @lru_cache(maxsize=1)
    def _init_dataframes(self) -> None:
        """Internal: Creates boundary dataframes based on boundary data

        Parameters
        ----------

        Returns
        -------
        None

        Notes
        -----
        This method doesn't have any return value, but it is cached
        so that on re-execution it doesn't recalculate.
        """

        boundaries = self._data
        ocean_boundaries = []
        land_boundaries = []
        interior_boundaries = []
        if boundaries is not None:
            for ibtype, bnds in boundaries.items():
                if ibtype is None:
                    for bnd_id, data in bnds.items():
                        indexes = list(map(self.mesh.nodes.get_index_by_id,
                                       data['indexes']))
                        ocean_boundaries.append({
                            'id': bnd_id,
                            "index_id": data['indexes'],
                            "indexes": indexes,
                            'geometry': LineString(self.mesh.coord[indexes])
                            })

                elif str(ibtype).endswith('1'):
                    for bnd_id, data in bnds.items():
                        indexes = list(map(self.mesh.nodes.get_index_by_id,
                                       data['indexes']))
                        interior_boundaries.append({
                            'id': bnd_id,
                            'ibtype': ibtype,
                            "index_id": data['indexes'],
                            "indexes": indexes,
                            'geometry': LineString(self.mesh.coord[indexes])
                            })
                else:
                    for bnd_id, data in bnds.items():
                        _indexes = np.array(data['indexes'])
                        if _indexes.ndim > 1:
                            # ndim > 1 implies we're dealing with an ADCIRC
                            # mesh that includes boundary pairs, such as weir
                            new_indexes = []
                            for i, line in enumerate(_indexes.T):
                                if i % 2 != 0:
                                    new_indexes.extend(np.flip(line))
                                else:
                                    new_indexes.extend(line)
                            _indexes = np.array(new_indexes).flatten()
                        else:
                            _indexes = _indexes.flatten()
                        indexes = list(map(self.mesh.nodes.get_index_by_id,
                                       _indexes))

                        land_boundaries.append({
                            'id': bnd_id,
                            'ibtype': ibtype,
                            "index_id": data['indexes'],
                            "indexes": indexes,
                            'geometry': LineString(self.mesh.coord[indexes])
                            })

        self._ocean = gpd.GeoDataFrame(ocean_boundaries)
        self._land = gpd.GeoDataFrame(land_boundaries)
        self._interior = gpd.GeoDataFrame(interior_boundaries)

    def ocean(self) -> gpd.GeoDataFrame:
        """Retrieve the ocean boundary information dataframe

        Parameters
        ----------

        Returns
        -------
        gpd.GeoDataFrame
            Dataframe containing the geometry and information of
            ocean open boundary.
        """

        self._init_dataframes()
        return self._ocean

    def land(self):
        """Retrieve the land boundary information dataframe

        Parameters
        ----------

        Returns
        -------
        gpd.GeoDataFrame
            Dataframe containing the geometry and information of
            land boundary.
        """

        self._init_dataframes()
        return self._land

    def interior(self):
        """Retrieve the island boundary information dataframe

        Parameters
        ----------

        Returns
        -------
        gpd.GeoDataFrame
            Dataframe containing the geometry and information of
            island boundary.
        """

        self._init_dataframes()
        return self._interior

    @property
    def data(self) -> Dict[Optional[int], Any]:
        """Read-only property referencing the boundary data dictionary"""
        return self._data

    @lru_cache(maxsize=1)
    def __call__(self) -> gpd.GeoDataFrame:
        """Retrieve the dataframe for all boundaries information

        Parameters
        ----------

        Returns
        -------
        gpd.GeoDataFrame
            Dataframe containing information for all boundaries shape
            and type.

        Notes
        -----
        The result of this method is cached, so that multiple calls
        to it won't result in multiple calculations. If the mesh
        is modified and the cache is not properly clear the calls
        to this method can result in invalid return values.
        """

        self._init_dataframes()
        data = []
        for bnd in self.ocean().itertuples():
            data.append({
                'id': bnd.id,
                'ibtype': None,
                "index_id": bnd.index_id,
                "indexes": bnd.indexes,
                'geometry': bnd.geometry})

        for bnd in self.land().itertuples():
            data.append({
                'id': bnd.id,
                'ibtype': bnd.ibtype,
                "index_id": bnd.index_id,
                "indexes": bnd.indexes,
                'geometry': bnd.geometry})

        for bnd in self.interior().itertuples():
            data.append({
                'id': bnd.id,
                'ibtype': bnd.ibtype,
                "index_id": bnd.index_id,
                "indexes": bnd.indexes,
                'geometry': bnd.geometry})

        return gpd.GeoDataFrame(data, crs=self.mesh.crs)

    def __len__(self) -> int:
        """Returns the number of boundary segments"""

        return len(self())

    def auto_generate(
            self,
            threshold: float = 0.,
            land_ibtype: int = 0,
            interior_ibtype: int = 1,
            ):
        """Automatically detect boundaries based on elevation data.

        Parameters
        ----------
        threshold : float, default=0
            Threshold above which nodes are considered dry nodes
            for ocean vs land boundary detection
        land_ibtype : int, default=0
            Value to assign to land boundary type
        interior_ibtype : int, default=1
            Value to assign to island boundary type

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any of the values assigned to a mesh node is `np.nan`.

        Notes
        -----
        An edge is considered dry if any of the attached nodes are
        dry (its elevation is larger than or equal to the `threshold`).
        """

        values = self.mesh.value
        if np.any(np.isnan(values)):
            raise ValueError(
                "Mesh contains invalid values. Raster values must"
                "be interpolated to the mesh before generating "
                "boundaries.")


        coords = self.mesh.msh_t.vert2['coord']
        coo_to_idx = {
            tuple(coo): idx
            for idx, coo in enumerate(coords)}

        polys = utils.get_mesh_polygons(self.mesh.msh_t)

        # TODO: Split using shapely to get bdry segments

        boundaries = defaultdict(defaultdict)
        bdry_type = dict

        get_id = self.mesh.nodes.get_id_by_index
        # generate exterior boundaries
        for poly in polys:
            ext_ring_coo = poly.exterior.coords
            ext_ring = np.array([
                    (coo_to_idx[ext_ring_coo[e]],
                     coo_to_idx[ext_ring_coo[e + 1]])
                    for e, coo in enumerate(ext_ring_coo[:-1])])

            # find boundary edges
            edge_tag = np.full(ext_ring.shape, 0)
            edge_tag[
                np.where(values[ext_ring[:, 0]] < threshold)[0], 0] = -1
            edge_tag[
                np.where(values[ext_ring[:, 1]] < threshold)[0], 1] = -1
            edge_tag[
                np.where(values[ext_ring[:, 0]] >= threshold)[0], 0] = 1
            edge_tag[
                np.where(values[ext_ring[:, 1]] >= threshold)[0], 1] = 1
            # sort boundary edges
            ocean_boundary = []
            land_boundary = []
            for i, (e0, e1) in enumerate(edge_tag):
                if np.any(np.asarray((e0, e1)) == 1):
                    land_boundary.append(tuple(ext_ring[i, :]))
                elif np.any(np.asarray((e0, e1)) == -1):
                    ocean_boundary.append(tuple(ext_ring[i, :]))
#            ocean_boundaries = utils.sort_edges(ocean_boundary)
#            land_boundaries = utils.sort_edges(land_boundary)
            ocean_boundaries = []
            if len(ocean_boundary) != 0:
                #pylint: disable=not-an-iterable
                ocean_segs = linemerge(coords[np.array(ocean_boundary)].tolist())
                ocean_segs = [ocean_segs] if isinstance(ocean_segs, LineString) else ocean_segs
                ocean_boundaries = [
                        [(coo_to_idx[seg.coords[e]], coo_to_idx[seg.coords[e + 1]])
                         for e, coo in enumerate(seg.coords[:-1])]
                        for seg in ocean_segs]
            land_boundaries = []
            if len(land_boundary) != 0:
                #pylint: disable=not-an-iterable
                land_segs = linemerge(coords[np.array(land_boundary)].tolist())
                land_segs = [land_segs] if isinstance(land_segs, LineString) else land_segs
                land_boundaries = [
                        [(coo_to_idx[seg.coords[e]], coo_to_idx[seg.coords[e + 1]])
                         for e, coo in enumerate(seg.coords[:-1])]
                        for seg in land_segs]

            _bnd_id = len(boundaries[None])
            for bnd in ocean_boundaries:
                e0, e1 = [list(t) for t in zip(*bnd)]
                e0 = [get_id(vert) for vert in e0]
                data = e0 + [get_id(e1[-1])]
                boundaries[None][_bnd_id] = bdry_type(
                        indexes=data, properties={})
                _bnd_id += 1

            # add land boundaries
            _bnd_id = len(boundaries[land_ibtype])
            for bnd in land_boundaries:
                e0, e1 = [list(t) for t in zip(*bnd)]
                e0 = [get_id(vert) for vert in e0]
                data = e0 + [get_id(e1[-1])]
                boundaries[land_ibtype][_bnd_id] = bdry_type(
                        indexes=data, properties={})

                _bnd_id += 1

        # generate interior boundaries
        _bnd_id = 0
        interior_boundaries = defaultdict()
        for poly in polys:
            interiors = poly.interiors
            for interior in interiors:
                int_ring_coo = interior.coords
                int_ring = [
                        (coo_to_idx[int_ring_coo[e]],
                         coo_to_idx[int_ring_coo[e + 1]])
                        for e, coo in enumerate(int_ring_coo[:-1])]

                # TODO: Do we still need these?
                e0, e1 = [list(t) for t in zip(*int_ring)]
                if utils.signed_polygon_area(self.mesh.coord[e0, :]) < 0:
                    e0 = e0[::-1]
                    e1 = e1[::-1]
                e0 = [get_id(vert) for vert in e0]
                e0.append(e0[0])
                interior_boundaries[_bnd_id] = e0
                _bnd_id += 1

        for bnd_id, data in interior_boundaries.items():
            boundaries[interior_ibtype][bnd_id] = bdry_type(
                        indexes=data, properties={})

        self._data = boundaries
        self._init_dataframes.cache_clear()
        self.__call__.cache_clear()
        self._init_dataframes()

SortedRingType = Dict[int,
                      Dict[Literal['exterior', 'interiors'],
                           Union[npt.NDArray, List[npt.NDArray]]]
                 ]

def sort_rings(
        index_rings: List[List[Tuple[int, int]]],
        vertices: npt.NDArray[np.float32]) -> SortedRingType:
    """Sorts a list of index-rings.

    Takes a list of unsorted index rings and sorts them into
    "exterior" and "interior" components. Any doubly-nested rings
    are considered exterior rings.

    Parameters
    ----------
    index_rings : List[List[Tuple[int, int]]]
        Unosorted list of list of mesh edges as specified by end node
        indexs of each edge.
    vertices : npt.NDArray[np.float32]
        2D ``n x 2`` array of node coordinate couples.

    Returns
    -------
    SortedRingType
        Dictionary of information aboout polygon boundaries extracted
        based on the input

    Notes
    -----
    The return value is a mapping of ring index to dictionary
    containing exterior and interior linear ring information as
    numpy array
    This function is not currently used, instead a different faster
    approach is used for boundary and polygon calculation from
    elements.
    """

    # TODO: Refactor and optimize. Calls that use :class:matplotlib.path.Path can
    # probably be optimized using shapely.

    # sort index_rings into corresponding "polygons"
    areas = []
    for index_ring in index_rings:
        e0, e1 = [list(t) for t in zip(*index_ring)]
        areas.append(float(Polygon(vertices[e0, :]).area))

    # maximum area must be main mesh
    idx = areas.index(np.max(areas))
    exterior = index_rings.pop(idx)
    areas.pop(idx)
    _id = 0
    _index_rings = {}
    _index_rings[_id] = {
        'exterior': np.asarray(exterior),
        'interiors': []
    }
    e0, e1 = [list(t) for t in zip(*exterior)]
    path = Path(vertices[e0 + [e0[0]], :], closed=True)
    while len(index_rings) > 0:
        # find all internal rings
        potential_interiors = []
        for i, index_ring in enumerate(index_rings):
            e0, e1 = [list(t) for t in zip(*index_ring)]
            if path.contains_point(vertices[e0[0], :]):
                potential_interiors.append(i)
        # filter out nested rings
        real_interiors = []
        for i, p_interior in reversed(
                list(enumerate(potential_interiors))):
            _p_interior = index_rings[p_interior]
            check = [index_rings[k]
                     for j, k in
                     reversed(list(enumerate(potential_interiors)))
                     if i != j]
            has_parent = False
            for _path in check:
                e0, e1 = [list(t) for t in zip(*_path)]
                _path = Path(vertices[e0 + [e0[0]], :], closed=True)
                if _path.contains_point(vertices[_p_interior[0][0], :]):
                    has_parent = True
            if not has_parent:
                real_interiors.append(p_interior)
        # pop real rings from collection
        for i in reversed(sorted(real_interiors)):
            _index_rings[_id]['interiors'].append(
                np.asarray(index_rings.pop(i)))
            areas.pop(i)
        # if no internal rings found, initialize next polygon
        if len(index_rings) > 0:
            idx = areas.index(np.max(areas))
            exterior = index_rings.pop(idx)
            areas.pop(idx)
            _id += 1
            _index_rings[_id] = {
                'exterior': np.asarray(exterior),
                'interiors': []
            }
            e0, e1 = [list(t) for t in zip(*exterior)]
            path = Path(vertices[e0 + [e0[0]], :], closed=True)
    return _index_rings



def _mesh_interpolate_worker(
        coords: npt.NDArray[np.float32],
        coords_crs: CRS,
        raster_path: Union[str, Path],
        chunk_size: Optional[int],
        method: Literal['spline', 'linear', 'nearest'] = "spline",
        filter_by_shape: bool = False):
    """Interpolator worker function to be used in parallel calls

    Parameters
    ----------
    coords : npt.NDArray[np.float32]
        Mesh node coordinates.
    coords_crs : CRS
        Coordinate reference system of the input mesh coordinates.
    raster_path : str or Path
        Path to the raster temporary working file.
    chunk_size : int or None
        Chunk size for windowing over the raster.
    method : {'spline', 'linear', 'nearest'}, default='spline'
        Method of interpolation.
    filter_by_shape : bool
        Flag for node filtering based on raster bbox or shape

    Returns
    -------
    idxs : npt.NDArray[bool]
        Mask of the nodes whose values are updated by current
        interpolation
    values : npt.NDArray[np.float32]
        Interpolated values.

    Raises
    ------
    ValueError
        If specified interpolation `method` is not supported.
    """

    coords = np.array(coords)
    raster = Raster(raster_path)
    idxs = []
    values = []
    for window in raster.iter_windows(chunk_size=chunk_size, overlap=2):

        if not raster.crs.equals(coords_crs):
            transformer = Transformer.from_crs(
                    coords_crs, raster.crs, always_xy=True)
            # pylint: disable=E0633
            coords[:, 0], coords[:, 1] = transformer.transform(
                coords[:, 0], coords[:, 1])
        xi = raster.get_x(window)
        yi = raster.get_y(window)
        # Use masked array to ignore missing values from DEM
        zi = raster.get_values(window=window, masked=True)

        if not filter_by_shape:
            _idxs = np.logical_and(
                np.logical_and(
                    np.min(xi) <= coords[:, 0],
                    np.max(xi) >= coords[:, 0]),
                np.logical_and(
                    np.min(yi) <= coords[:, 1],
                    np.max(yi) >= coords[:, 1]))
        else:
            shape = raster.get_multipolygon()
            gs_pt = gpd.points_from_xy(coords[:, 0], coords[:, 1])
            _idxs = gs_pt.intersects(shape)


        interp_mask = None

        if method == 'spline':
            f = RectBivariateSpline(
                xi,
                np.flip(yi),
                np.flipud(zi).T,
                kx=3, ky=3, s=0,
                # bbox=[min(x), max(x), min(y), max(y)]  # ??
            )
            _values = f.ev(coords[_idxs, 0], coords[_idxs, 1])

        elif method in ['nearest', 'linear']:
            # Inspired by StackOverflow 35807321
            if np.any(zi.mask):
                m_interp = RegularGridInterpolator(
                    (xi, np.flip(yi)),
                    np.flipud(zi.mask).T.astype(bool),
                    method=method
                )
                # Pick nodes NOT "contaminated" by masked values
                interp_mask = m_interp(coords[_idxs]) > 0

            f = RegularGridInterpolator(
                (xi, np.flip(yi)),
                np.flipud(zi).T,
                method=method
            )
            _values = f(coords[_idxs])

        else:
            raise ValueError(
                    f"Invalid value method specified <{method}>!")

        if interp_mask is not None:
            # pylint: disable=invalid-unary-operand-type

            helper = np.ones_like(_values).astype(bool)
            helper[interp_mask] = False
            # _idxs is inverse mask
            _idxs[_idxs] = helper
            _values = _values[~interp_mask]
        idxs.append(_idxs)
        values.append(_values)

    return (np.hstack(idxs), np.hstack(values))
