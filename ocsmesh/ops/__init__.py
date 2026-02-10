from ocsmesh.ops.combine_geom import GeomCombine
from ocsmesh.ops.combine_hfun import HfunCombine

# Import the modules (scripts) themselves
from ocsmesh.ops import combine_mesh
from ocsmesh.ops import river_mesh

def combine_geometry(*args, **kwargs):
    return GeomCombine(*args, **kwargs).run()

def combine_hfun(*args, **kwargs):
    return HfunCombine(*args, **kwargs).run()

__all__ = [
    "combine_geometry",
    "combine_hfun",
    "combine_mesh",  # Expose the module
    "river_mesh"     # Expose the module
]
