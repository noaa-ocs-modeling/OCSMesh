"""
Notes:
    After much reading and very many multiple attempts at parallel processing,
    the techinques discussed in the link below were the only things that worked
    for me:
    https://medium.com/@rvprasad/data-and-chunk-sizes-matter-when-using-multiprocessing-pool-map-in-python-5023c96875ef

    Another good source for discussion on the use of Python's Pool:
    https://thelaziestprogrammer.com/python/multiprocessing-pool-expect-initret-proposal

"""


def inpoly_parallel_worker(path, coord):
    return path.contains_point(coord)


def inpoly_pool_initializer(mpl_path):
    global path
    path = mpl_path


def inpoly_pool_worker(coord):
    return inpoly_parallel_worker(path, coord)
