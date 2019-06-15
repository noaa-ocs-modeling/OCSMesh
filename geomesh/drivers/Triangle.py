from pymesh import triangle
import geomesh


class Triangle(object):

    def __init__(self, PlanarStraightLineGraph):
        self._Triangle = triangle()
        self._PlanarStraightLineGraph = PlanarStraightLineGraph

    def run(self):
        self.Triangle.points = self.PlanarStraightLineGraph.vert2
        self.Triangle.segments = self.PlanarStraightLineGraph.edge2
        self.Triangle.auto_hole_detection = True
        self.Triangle.verbosity = 0
        self.Triangle.max_area = 4.33e6
        self.Triangle.run()

    @property
    def PlanarStraightLineGraph(self):
        return self._PlanarStraightLineGraph

    @property
    def Triangle(self):
        return self._Triangle

    @property
    def mesh(self):
        return self.Triangle.mesh

    @property
    def _PlanarStraightLineGraph(self):
        return self.__PlanarStraightLineGraph

    @property
    def _Triangle(self):
        return self.__Triangle

    @_PlanarStraightLineGraph.setter
    def _PlanarStraightLineGraph(self, PlanarStraightLineGraph):
        assert isinstance(PlanarStraightLineGraph,
                          geomesh.PlanarStraightLineGraph)
        self.__PlanarStraightLineGraph = PlanarStraightLineGraph

    @_Triangle.setter
    def _Triangle(self, Triangle):
        assert isinstance(Triangle, triangle)
        self.__Triangle = Triangle

    # @property
    # def points(self):
    #     return self._points
