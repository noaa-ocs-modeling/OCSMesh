from ocsmesh.ops.combine_geom import GeomCombine
from ocsmesh.ops.combine_hfun import HfunCombine

def combine_geometry(*args, **kwargs):
    return GeomCombine(*args, **kwargs).run()

def combine_hfun(*args, **kwargs):
    return HfunCombine(*args, **kwargs).run()

__all__ = [
        "combine_geometry",
        "combine_hfun"
]
