import pyproj


class CRS:

    def __set__(self, obj, crs):
        obj.__dict__['crs'] = pyproj.CRS.from_user_input(crs)

    def __get__(self, obj, val):
        return obj.__dict__['crs']
