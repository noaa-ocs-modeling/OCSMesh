#! python
from copy import deepcopy
import numpy as np
import ocsmesh

rast = ocsmesh.raster.Raster('test_dem.tif')

hfun_orig = ocsmesh.hfun.hfun.Hfun(rast, hmin=100, hmax=1500)
hfun_orig.add_contour(level=0, expansion_rate=0.001, target_size=100)
hfun_orig_jig = hfun_orig.msh_t()

hfun_calc_jig = deepcopy(hfun_orig_jig)
mesh_calc = ocsmesh.mesh.mesh.Mesh(hfun_calc_jig)
hfun_calc = ocsmesh.hfun.hfun.Hfun(mesh_calc)
hfun_calc.size_from_mesh()
hfun_calc_jig = hfun_calc.msh_t()

hfun_orig_val = hfun_orig_jig.value
hfun_calc_val = hfun_calc_jig.value
hfun_val_diff = hfun_orig_val - hfun_calc_val

# TODO: Come up with a more robust criteria
threshold = 0.2
err_value = np.max(np.abs(hfun_val_diff))/np.max(hfun_orig_val)
if err_value > threshold:
    raise ValueError(
            "Error for calculated size in this example exceeds the"
            f" threshold: {err_value} > {threshold}")
