#! python
import unittest
from pathlib import Path

import numpy as np
from jigsawpy import jigsaw_msh_t
from shapely import geometry

from ocsmesh.mesh.mesh import Mesh


def edge_at (x, y):
    return geometry.Point(x, y).buffer(0.05)

class BoundaryExtraction(unittest.TestCase):

    def setUp(self):
        """
        Note:
            x = x-index
            y = y-index

            node-index(node-id)

              25(26)             29(30)
          5     *---*---*---*---*
                | \ | \ | \ | \ |
          4     *---*---*---*---*
                | \ | \ | \ | \ |
          3     *---*---*---*---*
                | \ |   | \ | \ |
          2     *---*---*---*---*
                | \ | \ | \ | \ |
          1     *---*---*---*---*
                | \ | \ | \ | \ |
          0     *---*---*---*---*
              0(1)               4(5)

                0   1   2   3   4
        """

        # Create a basic grid with a hole
        nx, ny = 5, 6
        hole_square = 10
        X, Y = np.meshgrid(range(nx), range(ny))
        verts = np.array(list(zip(X.ravel(), Y.ravel())))
        cells = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                if (i + 1) + ((nx-1) * j) == hole_square:
                    continue
                cells.append([j * nx + i, j * nx + (i + 1), (j + 1) * nx + i])
                cells.append([j * nx + (i + 1), (j + 1) * nx + (i + 1), (j + 1) * nx + i])
        # NOTE: Everywhere is above 0 (auto: land) unless modified later
        vals = np.ones((len(verts), 1)) * 10

        # TODO: Replace with "make_mesh" util function
        mesh_msht = jigsaw_msh_t()
        mesh_msht.ndims = +2
        mesh_msht.mshID = 'euclidean-mesh'
        mesh_msht.tria3 = np.array(
            [(c, 0) for c in cells], dtype=jigsaw_msh_t.TRIA3_t
        )
        mesh_msht.vert2 = np.array(
            [(v, 0) for v in verts], dtype=jigsaw_msh_t.VERT2_t
        )
        mesh_msht.value = np.array(
            vals, dtype=jigsaw_msh_t.REALS_t
        )

        self.mesh = Mesh(mesh_msht)


    def test_auto_boundary_fails_if_na_elev(self):
        # Set one node to nan value
        self.mesh.msh_t.value[-1] = np.nan
        with self.assertRaises(ValueError):
            self.mesh.boundaries.auto_generate()


    def test_auto_boundary_1open_correctness(self):
        # Set right boundary to be open
        self.mesh.msh_t.value[self.mesh.msh_t.vert2['coord'][:, 0] > 3] = -10

        self.mesh.boundaries.auto_generate()
        # bdry is referring to mesh object and can be mutated
        bdry = self.mesh.boundaries

        # Mesh has one segment of each boundary type
        self.assertEqual(len(bdry.open()), 1)
        self.assertEqual(len(bdry.land()), 1)
        self.assertEqual(len(bdry.interior()), 1)

        # Boundaries node ID list
        self.assertEqual(bdry.open().iloc[0]['index_id'], [30, 25, 20, 15, 10, 5])
        self.assertEqual(
            bdry.land().iloc[0]['index_id'],
            [5, 4, 3, 2, 1, 6, 11, 16, 21, 26, 27, 28, 29, 30]
        )
        self.assertEqual(bdry.interior().iloc[0]['index_id'], [12, 13, 18, 17, 12])


    def test_auto_boundary_2open_correctness(self):
        # Set left and right boundary to be open
        self.mesh.msh_t.value[self.mesh.msh_t.vert2['coord'][:, 0] > 3] = -10
        self.mesh.msh_t.value[self.mesh.msh_t.vert2['coord'][:, 0] < 1] = -10

        self.mesh.boundaries.auto_generate()
        # bdry is referring to mesh object and can be mutated
        bdry = self.mesh.boundaries

        self.assertEqual(len(bdry.open()), 2)
        self.assertEqual(len(bdry.land()), 2)
        self.assertEqual(len(bdry.interior()), 1)

        # Boundaries node ID list
        self.assertEqual(bdry.open().iloc[0]['index_id'], [1, 6, 11, 16, 21, 26])
        self.assertEqual(bdry.open().iloc[1]['index_id'], [30, 25, 20, 15, 10, 5])
        self.assertEqual(bdry.land().iloc[0]['index_id'], [5, 4, 3, 2, 1])
        self.assertEqual(bdry.land().iloc[1]['index_id'], [26, 27, 28, 29, 30])
        self.assertEqual(bdry.interior().iloc[0]['index_id'], [12, 13, 18, 17, 12])


    def test_manual_boundary_specification_correctness(self):
        # Shape for wrapping bottom boundary
        shape1 = geometry.box(0.5, -0.5, 3.5, 0.5)
        shape2 = geometry.box(1.5, -0.5, 2.5, 0.5)

        self.mesh.boundaries.auto_generate()
        # bdry is referring to mesh object and can be mutated
        bdry = self.mesh.boundaries

        self.mesh.boundaries.set_open(region=shape1)
        self.mesh.boundaries.set_land(region=shape2)

        self.assertEqual(len(bdry.open()), 2)
        self.assertEqual(len(bdry.land()), 2)
        self.assertEqual(len(bdry.interior()), 1)

        # Boundaries node ID list
        self.assertEqual(bdry.open().iloc[0]['index_id'], [1, 2])
        self.assertEqual(bdry.open().iloc[1]['index_id'], [4, 5])
        self.assertEqual(
            bdry.land().iloc[0]['index_id'],
            [1, 6, 11, 16, 21, 26, 27, 28, 29, 30, 25, 20, 15, 10, 5]
        )
        self.assertEqual(bdry.land().iloc[1]['index_id'], [2, 3, 4])
        self.assertEqual(bdry.interior().iloc[0]['index_id'], [12, 13, 18, 17, 12])


    def test_manual_boundary_notaffect_interior(self):
        self.mesh.boundaries.auto_generate()
        # bdry is referring to mesh object and can be mutated
        bdry = self.mesh.boundaries

        self.mesh.boundaries.set_open(region=geometry.box(0.5, 1.5, 2.5, 3.5))

        self.assertEqual(len(bdry.open()), 0)
        self.assertEqual(len(bdry.land()), 1)
        self.assertEqual(len(bdry.interior()), 1)


    def test_manual_boundary_convex_region(self):
        self.mesh.boundaries.auto_generate()
        # bdry is referring to mesh object and can be mutated
        bdry = self.mesh.boundaries

        self.mesh.boundaries.set_open(region=geometry.Polygon([
            (-1, 4.5),
            (0.5, 4.5),
            (0.5, 3.5),
            (-0.5, 3.5),
            (-0.5, 1.5),
            (0.5, 1.5),
            (0.5, 0.5),
            (-1, 0.5),
        ]))

        self.assertEqual(len(bdry.open()), 2)
        self.assertEqual(len(bdry.land()), 2)
        self.assertEqual(len(bdry.interior()), 1)

        self.assertEqual(bdry.open().iloc[0]['index_id'], [1, 6, 11])
        self.assertEqual(bdry.open().iloc[1]['index_id'], [16, 21, 26])
        self.assertEqual(bdry.land().iloc[0]['index_id'], [11, 16])
        self.assertEqual(
            bdry.land().iloc[1]['index_id'],
            [26, 27, 28, 29, 30, 25, 20, 15, 10, 5, 4, 3, 2, 1]
        )


    def test_specified_boundary_order_nomerge(self):
        self.mesh.boundaries.auto_generate()
        # bdry is referring to mesh object and can be mutated
        bdry = self.mesh.boundaries

        self.mesh.boundaries.set_open(region=edge_at(1, 0))
        self.mesh.boundaries.set_open(region=edge_at(4, 1))
        self.mesh.boundaries.set_open(region=edge_at(0, 5))
        self.mesh.boundaries.set_open(region=edge_at(4, 5))

        self.assertEqual(len(bdry.open()), 4)
        self.assertEqual(len(bdry.land()), 4)
        self.assertEqual(len(bdry.interior()), 1)

        self.assertEqual(bdry.open().iloc[0]['index_id'], [1, 2, 3])
        self.assertEqual(bdry.open().iloc[1]['index_id'], [5, 10, 15])
        self.assertEqual(bdry.open().iloc[2]['index_id'], [21, 26, 27])
        self.assertEqual(bdry.open().iloc[3]['index_id'], [29, 30, 25])


    def test_manual_boundary_brokenring_stillconnected(self):
        self.mesh.boundaries.auto_generate()
        # bdry is referring to mesh object and can be mutated
        bdry = self.mesh.boundaries

        self.mesh.boundaries.set_open(region=edge_at(4, 5))

        # Mesh has one segment of each boundary type
        self.assertEqual(len(bdry.open()), 1)
        self.assertEqual(len(bdry.land()), 1)
        self.assertEqual(len(bdry.interior()), 1)


    def test_manual_boundary_merge_sametype(self):
        self.mesh.boundaries.auto_generate()
        # bdry is referring to mesh object and can be mutated
        bdry = self.mesh.boundaries

        self.mesh.boundaries.set_land(region=edge_at(1, 0), merge=True)
        self.mesh.boundaries.set_open(region=edge_at(4, 3), merge=True)
        self.mesh.boundaries.set_open(region=edge_at(4, 4), merge=True)


        # Mesh has one segment of each boundary type
        self.assertEqual(len(bdry.open()), 1)
        self.assertEqual(len(bdry.land()), 1)
        self.assertEqual(len(bdry.interior()), 1)


    def test_manual_boundary_connect_two_separate_segments(self):
        self.mesh.boundaries.auto_generate()
        # bdry is referring to mesh object and can be mutated
        bdry = self.mesh.boundaries

        self.mesh.boundaries.set_open(region=edge_at(0, 0), merge=True)
        self.mesh.boundaries.set_open(region=edge_at(4, 0), merge=True)

        self.assertEqual(len(bdry.open()), 2)
        self.assertEqual(len(bdry.land()), 2)
        self.assertEqual(len(bdry.interior()), 1)

        self.mesh.boundaries.set_open(region=edge_at(2, 0), merge=True)

        self.assertEqual(len(bdry.open()), 1)
        self.assertEqual(len(bdry.land()), 1)
        self.assertEqual(len(bdry.interior()), 1)


    def test_specified_boundary_order_withmerge(self):
        self.mesh.boundaries.auto_generate()
        # bdry is referring to mesh object and can be mutated
        bdry = self.mesh.boundaries

        self.mesh.boundaries.set_open(region=edge_at(1, 0))
        self.mesh.boundaries.set_open(region=edge_at(4, 1))
        self.mesh.boundaries.set_open(region=edge_at(0, 5))
        self.mesh.boundaries.set_open(region=edge_at(4, 5))
        self.mesh.boundaries.set_open(region=edge_at(4, 4))
        self.mesh.boundaries.set_open(region=edge_at(0, 0), merge=True)
        self.mesh.boundaries.set_open(region=edge_at(0, 4), merge=True)

        bdry = self.mesh.boundaries

        # Mesh has one segment of each boundary type
        self.assertEqual(len(bdry.open()), 4)
        self.assertEqual(len(bdry.land()), 4)
        self.assertEqual(len(bdry.interior()), 1)

        self.assertEqual(bdry.open().iloc[0]['index_id'], [6, 1, 2, 3])
        self.assertEqual(bdry.open().iloc[1]['index_id'], [5, 10, 15])
        self.assertEqual(bdry.open().iloc[2]['index_id'], [16, 21, 26, 27])
        self.assertEqual(bdry.open().iloc[3]['index_id'], [20, 25, 30, 29])

