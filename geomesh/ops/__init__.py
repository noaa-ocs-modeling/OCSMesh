from geomesh.ops.combine_geom import GeomCombine

combine_geometry = lambda *args, **kwargs: GeomCombine(
        *args, **kwargs).run()

__all__ = [
        "combine_geometry"
]
