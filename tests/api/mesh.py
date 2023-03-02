#! python
import unittest
from pathlib import Path

import numpy as np
from jigsawpy import jigsaw_msh_t

from ocsmesh.mesh.mesh import Mesh

class BoundaryExtraction(unittest.TestCase):

    def setUp(self):
        """
        index(id):

          0(1)               4(5)
            *---*---*---*---*
            | / | / | / | / |
            *---*---*---*---*
            | / | / | / | / |
            *---*---*---*---*
            | / |   | / | / |
            *---*---*---*---*
            | / | / | / | / |
            *---*---*---*---*
            | / | / | / | / |
            *---*---*---*---*
          20(21)             29(30)
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
        vals = np.ones((len(verts), 1)) * 10

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
        

    def test_auto_boundary_1ocean_correctness(self):
        # Set right boundary to be ocean
        self.mesh.msh_t.value[self.mesh.msh_t.vert2['coord'][:, 0] > 3] = -10

        self.mesh.boundaries.auto_generate()

        bdry = self.mesh.boundaries.data

        # Mesh has one segment of each boundary type
        self.assertEqual(len(bdry[None]), 1)
        self.assertEqual(len(bdry[0]), 1)
        self.assertEqual(len(bdry[1]), 1)

        # Counter-clockwise boundaries node ID list
        self.assertEqual(bdry[None][0]['indexes'], [30, 25, 20, 15, 10, 5])
        self.assertEqual(
            bdry[0][0]['indexes'],
            [5, 4, 3, 2, 1, 6, 11, 16, 21, 26, 27, 28, 29, 30]
        )
        self.assertEqual(bdry[1][0]['indexes'], [12, 13, 18, 17, 12])


    def test_auto_boundary_2oceans_correctness(self):
        # Set left and right boundary to be ocean
        self.mesh.msh_t.value[self.mesh.msh_t.vert2['coord'][:, 0] > 3] = -10
        self.mesh.msh_t.value[self.mesh.msh_t.vert2['coord'][:, 0] < 1] = -10

        self.mesh.boundaries.auto_generate()

        bdry = self.mesh.boundaries.data

        # Mesh has one segment of each boundary type
        self.assertEqual(len(bdry[None]), 2)
        self.assertEqual(len(bdry[0]), 2)
        self.assertEqual(len(bdry[1]), 1)

        # Counter-clockwise boundaries node ID list
        self.assertEqual(bdry[None][0]['indexes'], [1, 6, 11, 16, 21, 26])
        self.assertEqual(bdry[None][1]['indexes'], [30, 25, 20, 15, 10, 5])
        self.assertEqual(bdry[0][0]['indexes'], [5, 4, 3, 2, 1])
        self.assertEqual(bdry[0][1]['indexes'], [26, 27, 28, 29, 30])
        self.assertEqual(bdry[1][0]['indexes'], [12, 13, 18, 17, 12])
