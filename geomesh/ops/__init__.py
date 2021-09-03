from geomesh.ops.combine_geom import GeomCombine
from geomesh.ops.combine_hfun import HfunCombine

combine_geometry = lambda *args, **kwargs: GeomCombine(
        *args, **kwargs).run()

combine_hfun = lambda *args, **kwargs: HfunCombine(
        *args, **kwargs).run()

__all__ = [
        "combine_geometry",
        "combine_hfun"
]
