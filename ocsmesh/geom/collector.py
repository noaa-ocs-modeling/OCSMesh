"""This module defines geometry collector.
`
Geometry collector objects accepts a list of valid basic `Geom`
inputs and creates an object that merges the results of all the
other types of geometries, e.g. mesh-based and raster-based.

Notes
-----
This enables the user to process multiple DEM without having to worry
about the details of merging the output polygons taken from each DEM.
"""

import os
import logging
import warnings
import tempfile
from numbers import Number
from pathlib import Path
from multiprocessing import cpu_count
from typing import Union, Tuple, Optional, Iterable, List, Any

import geopandas as gpd
from pyproj import CRS, Transformer
from shapely.geometry import MultiPolygon, Polygon
from shapely import ops
from shapely.validation import explain_validity

from ocsmesh.mesh import Mesh
from ocsmesh.mesh.base import BaseMesh
from ocsmesh.raster import Raster
from ocsmesh.geom.base import BaseGeom
from ocsmesh.geom.raster import RasterGeom
from ocsmesh.geom.mesh import MeshGeom
from ocsmesh.features.contour import FilledContour, Contour
from ocsmesh.features.patch import Patch
from ocsmesh.ops import combine_geometry

CanCreateSingleGeom = Union[Raster, BaseMesh, Polygon, MultiPolygon]
CanCreateMultipleGeom = Iterable[Union[CanCreateSingleGeom, str]]
CanCreateGeom = Union[CanCreateSingleGeom, CanCreateMultipleGeom]

_logger = logging.getLogger(__name__)

class ContourPatchInfoCollector:
    """Helper class for information related to contour patches"""

    def __init__(self) -> None:
        self._contour_patch_info = []

    def add(self,
            contour_defn: FilledContour,
            patch_defn: Optional[Patch]
            ) -> None:
        """Add contour definition information to the contour collection
        """
        self._contour_patch_info.append((contour_defn, patch_defn))

    def __iter__(self):
        for ctr_defn, ptch_defn in self._contour_patch_info:
            yield ctr_defn, ptch_defn


class GeomCollector(BaseGeom):
    """Geometry object based on multiple input types

    Geometry type that merges the information from different types of
    inputs such as raster, mesh and shapely geometries.

    Attributes
    ----------
    crs : CRSDescriptor
        EPSG:4326. All the inputs are transformed before merging
    multipolygon : MultiPolygon
        Lazily calculated `shapely` (multi)polygon of the geometry

    Methods
    -------
    msh_t(**kwargs)
        Returns the `jigsawpy` vertex-edge representation of the geometry
    get_multipolygon(**kwargs)
        Returns `shapely` object representation of the geometry
    add_patch(...)
        Define local (patch) contour extraction definition from all
        the input data (e.g. raster, mesh, etc.)
    """

    def __init__(
            self,
            in_list: CanCreateMultipleGeom,
            base_mesh: Optional[Mesh] = None,
            zmin: Optional[float] = None,
            zmax: Optional[float] = None,
            nprocs: Optional[int] = None,
            chunk_size: Optional[int] = None,
            overlap: Optional[int] = None,
            verbosity: int = 0,
            base_shape: Optional[Union[Polygon, MultiPolygon]] = None,
            base_shape_crs: Union[str, CRS] = 'EPSG:4326'
            ) -> None:
        """Initialize geometry collector object

        Parameters
        ----------
        in_list : list
            List of objects that from which single geometry object
            can be created. This includes path to a raster or mesh
            file as a string, as well as Raster, Mesh or Polygon
            objects. Note that objects are not copied and currently
            any clipping that happens during processing will affect
            the objects pass to the `GeomCollector` constructor.
        base_mesh : Mesh or None, default=None
            Base mesh to be used for extracting boundaries of the
            domain. If not `None` all the input rasters all clipped
            by `base_mesh` polygon before further processing. This
            is useful for cases where we'd like to locally refine
            features of the domain (domain region >> inputs region)
            or when input rasters are much larger that the domain
            and we'd like to extract contours only within domain
            to save on computation.
        zmin : float or None, default=None
            Minimum elevation for extracting domain.
        zmax : float or None, default=None
            Maximum elevation for extracting domain.
        nprocs: int or None, default=None
            Number of processors to use in parallel parts of the
            collector computation
        chunk_size: int or None, default=None
            Chunk size for windowed calculation on rasters
        overlap: int or None default=None
            Window overlap for windowed calculation on rasters
        verbosity: int, default=0,
            Verbosity of the output
        base_shape: Polygon or MultiPolygon or None, default=None
            Similar to `base_mesh`, but instead of calculating the
            polygon from mesh, directly receive it from the calling
            code.
        base_shape_crs: str or CRS, default='EPSG:4326'
            CRS of the input `base_shape`.
        """

        # TODO: Like hfun collector and ops, later move the geom
        # combine functionality here and just call it from ops instead
        # of the other way around

        # For shapely and potentially mesh geom there's no priority
        # definition, they are just unioned with whatever the rest
        # of the input results in

        # store all the info pass
        # pass dem tmpfile or store address of each series
        # (global, patchcontour) to ops and get results and store
        # in the end combine all results (unary_union)

        # NOTE: Input Hfuns and their Rasters can get modified

        # Check nprocs
        nprocs = -1 if nprocs is None else nprocs
        nprocs = cpu_count() if nprocs == -1 else nprocs


        self._elev_info = dict(zmin=zmin, zmax=zmax)
        self._nprocs = nprocs
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._geom_list = []

        self._base_shape = base_shape
        self._base_shape_crs = CRS.from_user_input(base_shape_crs)

        # NOTE: Base mesh has to have a crs otherwise MeshGeom throws
        # exception
        self._base_mesh = base_mesh
        self._contour_patch_info_coll = ContourPatchInfoCollector()

        self._type_chk(in_list)

        # TODO: CRS considerations -- geom combine doesn't necessarily
        # return EPSG:4326 (unlike hfun collector msh_t)
        self._crs = 'EPSG:4326'

        for in_item in in_list:
            # Add supports(ext) to each hfun type?

            if isinstance(in_item, BaseGeom):
                geom = in_item

            elif isinstance(in_item, Raster):
                if self._base_shape:
                    clip_shape = self._base_shape
                    if not self._base_shape_crs.equals(in_item.crs):
                        transformer = Transformer.from_crs(
                            self._base_shape_crs, in_item.crs, always_xy=True)
                        clip_shape = ops.transform(
                                transformer.transform, clip_shape)
                    try:
                        in_item.clip(clip_shape)
                    except ValueError as err:
                        # This raster does not intersect shape
                        _logger.debug(err)
                        continue

                elif self._base_mesh:
                    try:
                        in_item.clip(self._base_mesh.get_bbox(crs=in_item.crs))
                    except ValueError as err:
                        # This raster does not intersect shape
                        _logger.debug(err)
                        continue

                geom = RasterGeom(in_item, **self._elev_info)

            elif isinstance(in_item, BaseMesh):
                geom = MeshGeom(in_item)

            elif isinstance(in_item, str):
                if in_item.endswith('.tif'):
                    raster = Raster(in_item)
                    if self._base_shape:
                        clip_shape = self._base_shape
                        if not self._base_shape_crs.equals(raster.crs):
                            transformer = Transformer.from_crs(
                                self._base_shape_crs, raster.crs, always_xy=True)
                            clip_shape = ops.transform(
                                    transformer.transform, clip_shape)
                        try:
                            in_item.clip(clip_shape)
                        except ValueError as err:
                            # This raster does not intersect shape
                            _logger.debug(err)
                            continue

                    elif self._base_mesh:
                        try:
                            raster.clip(self._base_mesh.get_bbox(crs=raster.crs))
                        except ValueError as err:
                            # This raster does not intersect shape
                            _logger.debug(err)
                            continue

                    geom = RasterGeom(raster, **self._elev_info)

                elif in_item.endswith(
                        ('.14', '.grd', '.gr3', '.msh', '.2dm')):
                    geom = MeshGeom(Mesh.open(in_item))

                else:
                    raise TypeError("Input file extension not supported!")

            self._geom_list.append(geom)


    def get_multipolygon(self, **kwargs: Any) -> MultiPolygon:
        """Returns the `shapely` representation of the geometry

        Calculates and returns the `MultiPolygon` representation of
        the geometry.

        Parameters
        ----------
        **kwargs : dict, optional
            Currently unused for this class, needed for generic API
            support

        Returns
        -------
        MultiPolygon
            Calculated and merged polygons from all geometry inputs.

        Notes
        -----
        All calculations are done lazily and the results is **not**
        cached. During this process all the stored contour extraction
        specs are applied on all the inputs and then the resulting
        shapes are merged.

        Calculation for each DEM and feature is stored on disk as
        feather files.  In the last steps are all these feather files
        are combined using the out of core calculation by `GeoPandas`.
        """

        # For now we don't need to do any calculations here, the
        # ops will take care of extracting everything. Later the logic
        # in ops needs to move here (like hfun collector)

        # Since raster geoms are stateless, the polygons should be
        # calculated everytime

        epsg4326 = CRS.from_user_input("EPSG:4326")
        mp = None
        with tempfile.TemporaryDirectory() as temp_dir:
            feather_files = []

            temp_path = Path(temp_dir)

            base_multipoly = None
            if self._base_shape:
                base_multipoly = self._base_shape
                if not self._base_shape_crs.equals(epsg4326):
                    transformer = Transformer.from_crs(
                        self._base_shape_crs, epsg4326, always_xy=True)
                    base_multipoly = ops.transform(
                            transformer.transform, base_multipoly)

            elif self._base_mesh:
                # TODO: Make sure all calcs are in EPSG:4326
                base_multipoly = self._base_mesh.hull.multipolygon()

            feather_files.append(self._extract_global_boundary(
                temp_path, base_multipoly))
            feather_files.extend(self._extract_nonraster_boundary(
                temp_path, base_multipoly))
            feather_files.extend(self._extract_features(
                temp_path, base_multipoly))

            gdf = gpd.GeoDataFrame(columns=['geometry'], crs=epsg4326)
            for f in feather_files:
                gdf = gdf.append(gpd.read_feather(f))

            mp = gdf.unary_union
            if isinstance(mp, Polygon):
                mp = MultiPolygon([mp])

            elif not isinstance(mp, MultiPolygon):
                raise ValueError(
                    "Union of all shapes resulted in invalid geometry"
                    + " type")

        return mp

    def add_patch(
            self,
            shape: Optional[Union[MultiPolygon, Polygon]] = None,
            level: Optional[Union[Tuple[float, float], float]] = None,
            contour_defn: Optional[Union[FilledContour, Contour]] = None,
            patch_defn: Optional[Patch] = None,
            shapefile: Optional[Union[str, Path]] = None,
            ) -> None:

        """Specifies an area of geometry inputs to extract contours

        Specifies a localized area of geometry inputs to extract
        contours from.

        Parameters
        ----------
        shape : MultiPolygon or Polygon or None, default=None
            `shapely` (multi)polygon to specify the area from which
            contour specification for this patch should be extracted.
            CRS is assumed to be ``EPSG:4326``.
        level: float or tuple of float, default=None
            Contour level specification for this patch. If provided,
            it could either be a single floating point number for
            the maximum elevation or a tuple indicating minimum and
            maximum elevations for contour extraction.
        contour_defn : FilledContour or Contour or None, default=None
            Alternative way of specifying maximum or minimum or both
            elevation levels for contour extraction.
        patch_defn : Patch or None, default=None
            Alternative way to specify which region within the input
            the contours need to be extracted from.
        shapefile: str or Path or None, default=None
            Alternative way to specify which region within the input
            the contours need to be extracted from.

        Returns
        -------
        None

        Notes
        -----
        This method doesn't result in any calculations. It simply
        stores the information about the user specified contour.
        During the `get_multipolygon` call the contours are actually
        calculated on all the inputs.
        """

        # Always lazy

        if not contour_defn:
            level0 = None
            level1 = None
            if isinstance(level, tuple) and len(level) == 2:
                level0 = level[0]
                level1 = level[1]
            elif isinstance(level, Number):
                level0 = None
                level1 = level
            else:
                raise ValueError(
                    "Level must be specified either by min and max values"
                    " or by only max value ")

            contour_defn = FilledContour(level1=level)

        elif isinstance(contour_defn, Contour):
            contour_defn = FilledContour(max_contour_defn=contour_defn)

        elif not isinstance(contour_defn, FilledContour):
            raise TypeError(
                f"Filled contour definition must be of type"
                f" {FilledContour} not {type(contour_defn)}!")

        elif level is not None:
            msg = "Level is ignored since a contour definition is provided!"
            warnings.warn(msg)
            _logger.info(msg)

        if not patch_defn:
            if shape:
                patch_defn = Patch(shape=shape)

            elif shapefile:
                patch_defn = Patch(shapefile=shapefile)

        elif not isinstance(patch_defn, Patch):
            raise TypeError(
                f"Patch definition must be of type {Patch} not"
                f" {type(patch_defn)}!")


        # If patch defn is None it means the patch applies to
        # all the sources of the accompanying contour
        self._contour_patch_info_coll.add(
            contour_defn, patch_defn)


    def _type_chk(self, input_list) -> None:
        """Checks the if the input types are supported for geometry

        Checks if geometry collector supports handling geometry created
        from the input types.

        Parameters
        ----------
        input_list : List[Any]

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If the any of the inputs are not supported
        """

        valid_types = (str, Raster, BaseGeom, BaseMesh)
        if not all(isinstance(item, valid_types) for item in input_list):
            raise TypeError(
                f'Input list items must be of type {", ".join(valid_types)}'
                f', or a derived type.')

    def _get_raster_sources(self) -> List[Union[RasterGeom, Raster]]:
        """Get the list rasters from inputs to the object constructor

        Retruns
        -------
        list of RasterGeom or Raster
            Filtered input list to return only rasters
        """

        raster_types = (RasterGeom, Raster)
        rasters = [
            i for i in self._geom_list if isinstance(i, raster_types)]
        return rasters

    def _get_raster_source_files(self) -> List[Path]:
        """Get the list of raster temporary files

        Returns
        -------
        list of path-like
            List of path to geom input raster object temporary files.
        """

        rasters = self._get_raster_sources()
        return self._get_raster_files_from_source(rasters)

    def _get_raster_files_from_source(
            self,
            rasters: Iterable[Union[Raster, RasterGeom]]
            ) -> List[Path]:
        """Get the list of raster temporary files

        Get the list of raster temporary files for input argument
        `rasters`.

        Parameters
        ----------
        rasters : iterable
            List of raster objects or geometries for which the
            temporary file address is returned

        Returns
        -------
        list of path-like
            List of path to method input raster object temporary files.
        """

        raster_files = []
        for r in rasters:
            if isinstance(r, Raster):
                raster_files.append(r.tmpfile)
            elif isinstance(r, RasterGeom):
                raster_files.append(r.raster.tmpfile)

        return raster_files

    def _get_non_raster_sources(self) -> List[Any]:
        """Get the list non-rasters from inputs in the constructor

        Retruns
        -------
        list
            Filtered input list to return only non-rasters
        """
        raster_types = (RasterGeom, Raster)
        non_rasters = [
            i for i in self._geom_list if not isinstance(i, raster_types)]
        return non_rasters

    def _get_valid_multipolygon(
            self,
            polygon: Union[Polygon, MultiPolygon]
            ) -> MultiPolygon:
        """Get a valid multipolygon from the input `polygon`

        Validates and if applicable creates a multipolygon from the
        input argument `polygon`

        Parameters
        ----------
        polygon : Polygon or MultiPolygon
            The input polygon or multipolygon which might not be
            topologically valid.

        Returns
        -------
        MultiPolygon
            A validated `shapely` `MultiPolygon` entity
        """

        # TODO: Performance bottleneck for valid checks
        if not polygon.is_valid:
            polygon = ops.unary_union(polygon)

            if not polygon.is_valid:
                polygon = polygon.buffer(0)

            if not polygon.is_valid:
                raise ValueError(explain_validity(polygon))

        if isinstance(polygon, Polygon):
            polygon = MultiPolygon([polygon])

        return polygon

    def _extract_global_boundary(
            self,
            out_dir: Union[Path, str],
            base_multipoly: Optional[MultiPolygon]
            ) -> Path:
        """Calculates the final boundary from all the raster inputs

        Parameters
        ----------
        out_dir : str
            Output directory into which feather files should be written
        base_multipoly : MultiPolygon
            Base shape to use for clipping DEM data.

        Returns
        -------
        path-like
            Path to a feather file containing the final boundary
            based on raster inputs
        """

        out_path = Path(out_dir)

        geom_path = out_path / 'global_boundary.feather'

        raster_files = self._get_raster_source_files()
        zmin = self._elev_info['zmin']
        zmax = self._elev_info['zmax']
        _logger.info("Extracting global boundaries")
        combine_geometry(
            raster_files, geom_path, "feather",
            None, base_multipoly, False,
            zmin, zmax,
            self._chunk_size, self._overlap,
            self._nprocs)

        return geom_path

    def _extract_nonraster_boundary(
            self,
            out_dir: Union[Path, str],
            base_multipoly: Optional[MultiPolygon]
            ) -> Path:
        """Calculates the final boundary from all the non-raster inputs

        Parameters
        ----------
        out_dir : str
            Output directory into which feather files should be written
        base_multipoly : MultiPolygon
            Base shape to use for clipping elevation data mesh.

        Returns
        -------
        path-like
            Path to a feather file containing the final boundary
            based on non-raster inputs
        """

        out_path = Path(out_dir)

        non_rasters = self._get_non_raster_sources()
        feather_files = []
        for e, geom in enumerate(non_rasters):

            geom_path = out_path / f'nonraster_{os.getpid()}_{e}.feather'

            crs = geom.crs
            multipoly = self._get_valid_multipolygon(
                    geom.get_multipolygon())
            gdf_non_raster = gpd.GeoDataFrame(
                    {'geometry': multipoly}, crs=crs)
            if crs != CRS.from_user_input("EPSG:4326"):
                gdf_non_raster = gdf_non_raster.to_crs("EPSG:4326")

            # TODO: Clip using base_multipoly?

            gdf_non_raster.to_feather(geom_path)

            feather_files.append(geom_path)

        return feather_files

    def _extract_features(
            self,
            out_dir: Union[Path, str],
            base_multipoly: Optional[MultiPolygon]
            ) -> List[Path]:
        """Calculates the local feature polygons

        Parameters
        ----------
        out_dir : str
            Output directory into which feather files should be written
        base_multipoly : MultiPolygon
            Base shape to use for clipping elevation data.

        Returns
        -------
        list of path-like
            Paths to feather files containing all the extracted features

        Notes
        -----
        Currently the only feature available is patch feature extraction
        """


        feather_files = []
        feather_files.extend(self._apply_patch(out_dir, base_multipoly))

        return feather_files

    def _apply_patch(
            self,
            out_dir: Union[Path, str],
            base_multipoly: Optional[MultiPolygon]
            ) -> List[Path]:
        """Extracts contours based on patch specifications

        Extracts the domain filled contours based on the specifications
        user provided earlier by `add_patch`

        Parameters
        ----------
        out_dir : str
            Output directory into which feather files should be written
        base_multipoly : MultiPolygon
            Base shape to use for clipping elevation data.

        Returns
        -------
        list of path-like
            List of extracted contours stored on files on the disk
        """

        out_path = Path(out_dir)


        raster_files = self._get_raster_source_files()
        zmin = self._elev_info['zmin']
        zmax = self._elev_info['zmax']

        feather_files = []
        for e, (ctr_defn, ptch_defn) in enumerate(self._contour_patch_info_coll):

            patch_zmin, patch_zmax = ctr_defn.level
            if not patch_zmin:
                patch_zmin = zmin
            if not patch_zmax:
                patch_zmax = zmax

            patch_raster_files = raster_files
            if ctr_defn.has_source:
                patch_rasters = ctr_defn.sources
                patch_raster_files = self._get_raster_files_from_source(
                        patch_rasters)

            # Pass patch shape instead of base mesh
            # See explanation in add_patch
            _logger.info("Extracting patch contours")
            combine_poly = base_multipoly
            if ptch_defn:
                patch_mp, crs = ptch_defn.get_multipolygon()
                gdf_patch = gpd.GeoDataFrame(
                        {'geometry': patch_mp}, crs=crs)
                if crs != CRS.from_user_input("EPSG:4326"):
                    gdf_patch = gdf_patch.to_crs("EPSG:4326")
                combine_poly = MultiPolygon(list(gdf_patch.geometry))
            geom_path = out_path / f'patch_{os.getpid()}_{e}.feather'
            combine_geometry(
                patch_raster_files, geom_path, "feather",
                None, combine_poly, True,
                patch_zmin, patch_zmax,
                self._chunk_size, self._overlap,
                self._nprocs)

            if geom_path.is_file():
                feather_files.append(geom_path)

        return feather_files
