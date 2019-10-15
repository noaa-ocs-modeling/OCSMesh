"""
Notes:
    After much reading and very many multiple attempts at parallel processing,
    the techinques discussed below were the only things that worked.
    https://medium.com/@rvprasad/data-and-chunk-sizes-matter-when-using-multiprocessing-pool-map-in-python-5023c96875ef
"""


def parallel_worker(raster, coord):
    try:
        sample = list(raster.sample(coord, raster.count))
        if sample[0][0] == 0.:
            return True
        else:
            return False
    except (TypeError, IndexError):
        return False


def pool_initializer(_raster):
    global raster
    raster = _raster


def pool_worker(coord):
    return parallel_worker(raster, coord)
