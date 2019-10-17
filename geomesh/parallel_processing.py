"""
Notes:
    After much reading and very many multiple attempts at parallel processing,
    the techinques discussed in the link below were the only things that worked
    for me:
    https://medium.com/@rvprasad/data-and-chunk-sizes-matter-when-using-multiprocessing-pool-map-in-python-5023c96875ef

    Another good source for discussion on the use of Python's Pool:
    https://thelaziestprogrammer.com/python/multiprocessing-pool-expect-initret-proposal

"""


def hfun_parallel_worker(raster, coord):
    try:
        sample = list(raster.sample(coord, raster.count))
        if sample[0][0] == raster.nodataval(raster.count):
            return True
        else:
            return False
    except (TypeError, IndexError):
        return False


def hfun_pool_initializer(_raster):
    global raster
    raster = _raster


def hfun_pool_worker(coord):
    return hfun_parallel_worker(raster, coord)


def pslg_parallel_worker(path, coord):
    return path.contains_point(coord)


def pslg_pool_initializer(mpl_path):
    global path
    path = mpl_path


def pslg_pool_worker(coord):
    return pslg_parallel_worker(path, coord)
