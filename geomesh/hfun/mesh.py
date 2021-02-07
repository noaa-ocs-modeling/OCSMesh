from jigsawpy import jigsaw_msh_t
import numpy as np
from pyproj import CRS, Transformer
import utm

from geomesh.hfun.base import BaseHfun
from geomesh.crs import CRS as CRSDescriptor


class HfunMesh(BaseHfun):

    _crs = CRSDescriptor()

    def __init__(self, mesh):
        self._mesh = mesh
        self._crs = mesh.crs

    def msh_t(self) -> jigsaw_msh_t:
        if self.crs.is_geographic:
            x0, y0, x1, y1 = self.mesh.get_bbox().bounds
            _, _, number, letter = utm.from_latlon(
                    (y0 + y1)/2, (x0 + x1)/2)
            utm_crs = CRS(
                    proj='utm',
                    zone=f'{number}{letter}',
                    ellps={
                        'GRS 1980': 'GRS80',
                        'WGS 84': 'WGS84'
                        }[self.crs.ellipsoid.name]
                )
            transformer = Transformer.from_crs(
                self.crs, utm_crs, always_xy=True)
            msh_t = self.mesh.msh_t
            msh_t.vert2['coord'] = np.vstack(
                transformer.transform(
                    self.mesh.msh_t.vert2['coord'][:, 0],
                    self.mesh.msh_t.vert2['coord'][:, 1]
                    )).T
            msh_t.crs = utm_crs
            return msh_t
        else:
            return self.mesh.msh_t

    @property
    def mesh(self):
        return self._mesh

    @property
    def crs(self):
        return self._crs
