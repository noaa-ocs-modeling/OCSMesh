from __future__ import annotations
from typing import Optional
import logging
from copy import deepcopy
import uuid

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

try:
    import gmsh
    _HAS_GMSH = True
except ImportError:
    _HAS_GMSH = False

from ocsmesh.internal import MeshData
from ocsmesh.engines.base import BaseMeshEngine, BaseMeshOptions

_logger = logging.getLogger(__name__)


class GmshOptions(BaseMeshOptions):
    """
    Wraps options for the Gmsh library.
    """

    def __init__(self, **kwargs):
        if not _HAS_GMSH:
            raise ImportError("Gmsh library not installed.")

        # Extract known args that are not part of gmsh native options
        self._bnd_representation = kwargs.pop('bnd_representation', 'fixed')
        self._optimize_mesh = kwargs.pop('optimize_mesh', True)

        self._options = {
            # Algorithm 5 (Delaunay) often respects variable sizing fields better
            "Mesh.Algorithm": 6,
            "General.Verbosity": 2,
            # Critical options for sizing fields in 4.x
            "Mesh.MeshSizeExtendFromBoundary": 0,
            "Mesh.MeshSizeFromPoints": 0,
            "Mesh.MeshSizeFromCurvature": 0,
        }
        self._options.update(kwargs)

    @property
    def bnd_representation(self):
        return self._bnd_representation

    @property
    def optimize_mesh(self):
        return self._optimize_mesh

    def get_config(self) -> dict:
        return deepcopy(self._options)


class GmshEngine(BaseMeshEngine):
    """
    Concrete Gmsh engine supporting direct point-based sizing via Background Fields
    """

    def __init__(self, options: BaseMeshOptions):
        super().__init__(options)
        if not _HAS_GMSH:
            raise ImportError("Gmsh library not installed.")

    def generate(
        self,
        shape: gpd.GeoSeries,
        sizing: Optional[MeshData | int | float] = None,
        seed: Optional[MeshData] = None,
    ) -> MeshData:

        combined_shape = shape.union_all()
        if combined_shape.is_empty:
            raise ValueError("Input shape is empty.")

        # If Gmsh is already running (user session), we don't want to finalize it.
        # If we start it, we finalize it.
        we_initialized = False
        if not gmsh.isInitialized():
            gmsh.initialize()
            we_initialized = True

        # Use a unique model name to avoid clashing with existing user models
        model_name = f"ocsmesh_model_{uuid.uuid4().hex}"

        # Save current model to restore later if needed
        prev_model = None
        try:
            prev_model = gmsh.model.getCurrent()
        except:
            pass

        gmsh.model.add(model_name)
        gmsh.model.setCurrent(model_name)

        try:
            # 1. Build geometry (Respects bnd_representation)
            self._add_shapely_geometry_to_gmsh(combined_shape)

            # 2. Apply options
            opts = self._options.get_config()
            for key, val in opts.items():
                if isinstance(val, str):
                    gmsh.option.setString(key, val)
                else:
                    gmsh.option.setNumber(key, float(val))

            # 3. Apply sizing (Background Field method)
            self._apply_sizing(sizing)

            # 4. Generate mesh
            gmsh.model.mesh.generate(2)

            # 5. Optimize (Optional)
            if self._options.optimize_mesh:
                gmsh.model.mesh.optimize("Netgen")
                gmsh.model.mesh.optimize("Laplace2D")

            return self._extract_meshdata()

        except Exception as e:
            _logger.error(f"Gmsh generation failed: {e}")
            raise e

        finally:
            # Clean up OUR model only
            gmsh.model.remove()

            # Restore previous state
            if prev_model:
                try:
                    gmsh.model.setCurrent(prev_model)
                except:
                    pass

            # Only finalize if we started the session
            if we_initialized:
                gmsh.finalize()

    # --------------------------
    # Geometry
    # --------------------------
    def _add_shapely_geometry_to_gmsh(self, shape):
        """
        Convert Shapely Polygon/MultiPolygon to Gmsh OCC geometry.
        Handles 'exact' boundary locking.
        """
        if isinstance(shape, Polygon):
            polys = [shape]
        elif isinstance(shape, MultiPolygon):
            polys = list(shape.geoms)
        else:
            raise TypeError(f"Unsupported geometry type: {type(shape)}")

        self._point_tags = {}

        # Check boundary preference
        bnd_rep = self._options.bnd_representation
        is_exact = (bnd_rep == 'exact')

        def add_loop(coords):
            pts = []
            # 1. Add Points
            for x, y in coords[:-1]:
                key = (float(x), float(y))
                if key not in self._point_tags:
                    self._point_tags[key] = gmsh.model.occ.addPoint(*key, 0.0)
                pts.append(self._point_tags[key])

            # 2. Add Lines
            line_tags = []
            for i in range(len(pts)):
                p1 = pts[i]
                p2 = pts[(i + 1) % len(pts)]
                l_tag = gmsh.model.occ.addLine(p1, p2)
                line_tags.append(l_tag)

            return gmsh.model.occ.addCurveLoop(line_tags), line_tags

        all_line_tags = []
        for poly in polys:
            if poly.is_empty:
                continue
            ext_loop, ext_lines = add_loop(poly.exterior.coords)
            all_line_tags.extend(ext_lines)

            holes = []
            for r in poly.interiors:
                hole_loop, hole_lines = add_loop(r.coords)
                holes.append(hole_loop)
                all_line_tags.extend(hole_lines)

            gmsh.model.occ.addPlaneSurface([ext_loop] + holes)

        gmsh.model.occ.synchronize()

        # HANDLING BOUNDARY TYPES
        if is_exact:
            _logger.info("Boundary representation is 'exact': Locking edges.")
            for l_tag in all_line_tags:
                gmsh.model.mesh.setTransfiniteCurve(l_tag, 2)

        # For 'fixed' and 'adapt', we leave the curves standard.
        # Since we used occ.addLine between vertices, the vertices are already hard points ('fixed').
        # If 'adapt' was chosen, the Driver has already resampled these vertices for us.

    # --------------------------
    # Sizing
    # --------------------------
    def _apply_sizing(self, sizing: Optional[MeshData | int | float]):
        if sizing is None:
            return

        # Case 1: Constant Sizing
        if isinstance(sizing, (int, float)):
            gmsh.option.setNumber("Mesh.MeshSizeMin", float(sizing))
            gmsh.option.setNumber("Mesh.MeshSizeMax", float(sizing))
            return

        if not isinstance(sizing, MeshData):
            raise TypeError("Sizing must be MeshData, int, or float")

        coords = sizing.coords
        values = sizing.values
        if coords is None or values is None:
            raise ValueError("MeshData sizing lacks coords/values")

        _logger.info(f"Applying sizing field with {len(coords)} points...")

        # Case 2: Hfun MeshData (Background Field)
        # 1. Create a "Post-Processing View"
        view_tag = gmsh.view.add("hfun_sizing")

        # 2. Prepare data [x, y, z, val]
        n_pts = len(coords)
        try:
            z_col = np.zeros((n_pts, 1))
            val_col = values.reshape(-1, 1)
            # Flatten to list
            data_block = np.hstack((coords, z_col, val_col))
            flat_data = data_block.ravel().tolist()
        except Exception:
            flat_data = []
            for (x, y), h in zip(coords, values):
                flat_data.extend([x, y, 0.0, float(h)])

        # 3. Upload data
        gmsh.view.addListData(view_tag, "SP", n_pts, flat_data)

        # 4. Create Field
        field_tag = gmsh.model.mesh.field.add("PostView")
        gmsh.model.mesh.field.setNumber(field_tag, "ViewTag", view_tag)

        # 5. Set As Background
        gmsh.model.mesh.field.setAsBackgroundMesh(field_tag)

        gmsh.model.occ.synchronize()

    # --------------------------
    # Extract mesh
    # --------------------------
    def _extract_meshdata(self) -> MeshData:
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        if len(node_tags) == 0:
            raise RuntimeError("Gmsh generated an empty mesh.")

        coords = np.array(node_coords).reshape(-1, 3)[:, :2]
        tag_map = {tag: i for i, tag in enumerate(node_tags)}

        elem_types, _, elem_node_tags = gmsh.model.mesh.getElements()
        triangles = []

        for etype, tags in zip(elem_types, elem_node_tags):
            if etype == 2:  # triangle
                t = np.array(tags).reshape(-1, 3)
                triangles.append(np.vectorize(tag_map.get)(t))

        tri_arr = np.vstack(triangles) if triangles else None
        return MeshData(coords=coords, tria=tri_arr, quad=None)

    def remesh(self, mesh, remesh_region=None, sizing=None, seed=None):
        """
        Dummy implementation to satisfy abstract base class requirements.
        """
        raise NotImplementedError("Remeshing is not currently implemented for Gmsh engine.")
