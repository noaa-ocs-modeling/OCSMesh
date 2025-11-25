from typing import Any, Optional
import logging

try:
    import jigsawpy
    _HAS_JIGSAW = True
except ImportError:
    _HAS_JIGSAW = False

from ocsmesh.internal import MeshData
from ocsmesh.engines.base import BaseMeshEngine, BaseMeshOptions


_logger = logging.getLogger(__name__)


class JigsawOptions(BaseMeshOptions):
    """
    Wraps jigsaw_opts_t options.
    """

    def __init__(self, **kwargs):
        if not _HAS_JIGSAW:
            raise ImportError("Jigsawpy not installed.")
        ############# OLD CODE FROM DRIVER
        opts = obj.__dict__.get('opts')
        if opts is None:
            opts = jigsaw_jig_t()
            opts.mesh_dims = +2
            opts.optm_tria = True
            opts.hfun_scal = 'absolute'
            obj.__dict__['opts'] = opts
        ################

        ############ FROM HFUN
            opts = jigsaw_jig_t()
            # additional configuration options
            opts.mesh_dims = +2
            opts.hfun_scal = 'absolute'
            # no need to optimize for size function generation
            opts.optm_tria = False

            opts.hfun_hmin = np.min(hfun.value) if self.hmin is None else \
                self.hmin
            opts.hfun_hmax = np.max(hfun.value) if self.hmax is None else \
                self.hmax
            opts.verbosity = self.verbosity if verbosity is None else \
                verbosity
        #####################

        # internal storage for options
        self._opts = jigsawpy.jigsaw_opts_t()

        # Set defaults
        self._opts.hfun_scal = "absolute"
        self._opts.hfun_hmax = float("inf")
        self._opts.hfun_hmin = 0.0
        self._opts.mesh_dims = +2
        self._opts.mesh_top1 = True
        self._opts.geom_feat = True

        # Apply user overrides
        for key, value in kwargs.items():
            if hasattr(self._opts, key):
                setattr(self._opts, key, value)
            else:
                _logger.warning(
                    f"Unknown Jigsaw option: {key}"
                )

    def get_config(self) -> Any:
        return self._opts


class JigsawEngine(BaseMeshEngine):
    """
    Concrete implementation using JIGSAW.
    """

    def generate(
        self,
        shape: Any,
        sizing: Optional[Any] = None
    ) -> MeshData:

        ################## FROM HFUN
        if marche is True:
            libsaw.marche(opts, hfun)
        ###########################

        if not _HAS_JIGSAW:
            raise ImportError("Jigsawpy not installed.")

        # 1. Prepare Geometry
        geom = shape_to_msh_t(shape)

        # 2. Prepare Sizing
        hfun = jigsawpy.jigsaw_msh_t()
        if sizing is not None:
            # Assuming sizing is convertible to msh_t or
            # is already a MeshData/msh_t.
            # Logic depends on input type of sizing.
            if isinstance(sizing, MeshData):
                hfun = meshdata_to_jigsaw(sizing)
            # If sizing is function/raster, extra logic needed here
            # For now, we assume MeshData-like input for Hfun
            pass

        # 3. Prepare Output Container
        mesh = jigsawpy.jigsaw_msh_t()

        # 4. Run Engine
        opts = self._options.get_config()
        jigsawpy.lib.jigsaw(
            opts,
            geom,
            mesh,
            init=hfun,
            hfun=hfun
        )

        # 5. Convert back to MeshData
        return jigsaw_to_meshdata(mesh)
        ############################### OLD CODE 
        sieve=None, quality_metric=1.05, remesh_tiny_elements=False):

        hfun_msh_t = self.hfun.msh_t()

        output_mesh = jigsaw_msh_t()
        output_mesh.mshID = 'euclidean-mesh'
        output_mesh.ndims = 2

        self.opts.hfun_hmin = np.min(hfun_msh_t.value)
        self.opts.hfun_hmax = np.max(hfun_msh_t.value)
        self.opts.mesh_rad2 = float(quality_metric)

        geom_msh_t = self.geom.msh_t()

        # When the center of geom and hfun are NOT the same, utm
        # zones would be different for resulting msh_t.
        if geom_msh_t.crs != hfun_msh_t.crs:
            utils.reproject(hfun_msh_t, geom_msh_t.crs)
        output_mesh.crs = hfun_msh_t.crs

        _logger.info('Calling libsaw.jigsaw() ...')
        libsaw.jigsaw(
            self.opts,
            geom_msh_t,
            output_mesh,
            init=hfun_msh_t if self._init is True else None,
            hfun=hfun_msh_t
        )

        # post process
        if output_mesh.tria3['index'].shape[0] == 0:
            _err = 'ERROR: Jigsaw returned empty mesh.'
            _logger.error(_err)
            raise RuntimeError(_err)

        if self._crs is not None:
            utils.reproject(output_mesh, self._crs)

        _logger.info('Finalizing mesh...')
        if self.opts.hfun_hmin > 0 and remesh_tiny_elements:
            output_mesh = remesh_small_elements(
                self.opts, geom_msh_t, output_mesh, hfun_msh_t)

        _logger.info('done!')
        return Mesh(output_mesh)
        ##############################

    def remesh(
        self,
        mesh: MeshData,
        shape: Optional[Any] = None,
        sizing: Optional[Any] = None
    ) -> MeshData:

        if not _HAS_JIGSAW:
            raise ImportError("Jigsawpy not installed.")

        # 1. Convert MeshData to Jigsaw format (Initial Mesh)
        init_mesh = meshdata_to_jigsaw(mesh)

        # 2. Prepare Geometry (if a region/shape is provided)
        geom = None
        if shape is not None:
            geom = shape_to_msh_t(shape)

        # 3. Setup Sizing (Hfun)
        hfun = jigsawpy.jigsaw_msh_t()
        if sizing is not None:
             if isinstance(sizing, MeshData):
                hfun = meshdata_to_jigsaw(sizing)

        # 4. Output Container
        out_mesh = jigsawpy.jigsaw_msh_t()

        # 5. Run Engine
        # Jigsaw uses 'geom' to constrain the remeshing if provided
        opts = self._options.get_config()

        # If no geom provided for remesh, Jigsaw might behave
        # differently depending on opts.
        # Usually it needs a geom definition or relies on init mesh
        # boundary if geom is missing.
        jigsawpy.lib.jigsaw(
            opts,
            geom,
            out_mesh,
            init=init_mesh,
            hfun=hfun
        )

        # 6. Return Result
        return jigsaw_to_meshdata(out_mesh)


def multipolygon_to_jigsaw_msh_t(
        multipolygon: MultiPolygon,
        crs: CRS
    ) -> jigsawpy.jigsaw_msh_t:
    """Calculate vertex-edge representation of multipolygon

    Calculate `jigsawpy` vertex-edge representation of the input
    `shapely` multipolygon. The resulting object is in a projected or
    local UTM CRS

    Parameters
    ----------
    multipolygon : MultiPolygon
        Input polygon for which the vertex-edge representation is to
        be calculated
    crs : CRS
        CRS of the input polygon

    Returns
    -------
    jigsawpy.jigsaw_msh_t
        Vertex-edge representation of the input multipolygon

    Raises
    ------
    NotImplementedError
    """

    utm_crs = utils.estimate_bounds_utm(
            multipolygon.bounds, crs)
    if utm_crs is not None:
        transformer = Transformer.from_crs(crs, utm_crs, always_xy=True)
        multipolygon = ops.transform(transformer.transform, multipolygon)

    msht = utils.shape_to_msh_t(multipolygon)
    msht.crs = crs
    if utm_crs is not None:
        msht.crs = utm_crs
    return msht


def meshdata_to_jigsaw(mesh: MeshData) -> 'jigsawpy.jigsaw_msh_t':
    if not _HAS_JIGSAW:
        raise ImportError("Jigsawpy not installed.")

    msh = jigsawpy.jigsaw_msh_t()
    msh.ndims = +2
    msh.mshID = 'euclidean-mesh'

    # Vertices
    if mesh.coords is not None:
        msh.vert2 = np.array(
            [(c, 0) for c in mesh.coords],
            dtype=jigsawpy.jigsaw_msh_t.VERT2_t
        )

    # Triangles
    if mesh.tria is not None:
        msh.tria3 = np.array(
            [(t, 0) for t in mesh.tria],
            dtype=jigsawpy.jigsaw_msh_t.TRIA3_t
        )

    # Quads
    if mesh.quad is not None:
        msh.quad4 = np.array(
            [(q, 0) for q in mesh.quad],
            dtype=jigsawpy.jigsaw_msh_t.QUAD4_t
        )

    # Values (Scalars)
    if mesh.values is not None:
        vals = mesh.values
        if vals.ndim == 1:
            vals = vals.reshape(-1, 1)
        msh.value = np.array(
            vals, dtype=jigsawpy.jigsaw_msh_t.REALS_t
        )

    # CRS
    if mesh.crs is not None:
        msh.crs = mesh.crs

    return msh


def jigsaw_to_meshdata(msh: 'jigsawpy.jigsaw_msh_t') -> MeshData:
    coords = msh.vert2['coord']

    tria = None
    if msh.tria3.size > 0:
        tria = msh.tria3['index']

    quad = None
    if msh.quad4.size > 0:
        quad = msh.quad4['index']

    values = None
    if msh.value.size > 0:
        values = msh.value

    # Extract CRS if available in the Jigsaw object
    # (Note: jigsawpy doesn't strictly enforce .crs field standard,
    # but we assume it might exist if we put it there)
    crs = getattr(msh, 'crs', None)

    return MeshData(
        coords=coords,
        tria=tria,
        quad=quad,
        values=values,
        crs=crs
    )


def shape_to_msh_t(
    shape: Union[Polygon, MultiPolygon]
) -> 'jigsawpy.jigsaw_msh_t':

    if not _HAS_JIGSAW:
        raise ImportError("Jigsawpy not installed.")

    if isinstance(shape, Polygon):
        shape = MultiPolygon([shape])

    vert2_list = []
    edge2_list = []

    # Helper to process a linear ring
    def process_ring(ring, start_idx):
        coords = ring.coords[:-1] # Drop duplicate end point
        n_pts = len(coords)
        if n_pts < 3:
            return 0

        for xy in coords:
            vert2_list.append((xy, 0))

        # Create edges: i -> i+1, and close last -> first
        for i in range(n_pts):
            u = start_idx + i
            v = start_idx + ((i + 1) % n_pts)
            edge2_list.append(((u, v), 0))

        return n_pts

    current_idx = 0
    for poly in shape.geoms:
        # Exterior
        n = process_ring(poly.exterior, current_idx)
        current_idx += n

        # Interiors
        for interior in poly.interiors:
            n = process_ring(interior, current_idx)
            current_idx += n

    msh = jigsawpy.jigsaw_msh_t()
    msh.ndims = +2
    msh.mshID = 'euclidean-mesh'

    if vert2_list:
        msh.vert2 = np.array(
            vert2_list,
            dtype=jigsawpy.jigsaw_msh_t.VERT2_t
        )
    if edge2_list:
        msh.edge2 = np.array(
            edge2_list,
            dtype=jigsawpy.jigsaw_msh_t.EDGE2_t
        )

    return msh
