"""What ``import ocsmesh`` changes under a launcher.

Any launcher variable used to count as an MPI job, and SLURM_NTASKS is
set in every SLURM batch job: a serial script in a one-task job got its
BLAS threads pinned to 1 and the 'spawn' start method at import.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from ocsmesh import mpi
from tests.api import multitile

REPO = Path(__file__).resolve().parents[2]
PROBE = ("import os, multiprocessing as mp, ocsmesh; "
         "print(os.environ.get('OMP_NUM_THREADS'), mp.get_start_method())")
LAUNCHER_VARS = ('OMPI_', 'PMI', 'MPI_', 'SLURM', 'HYDRA')


def _clean_env(**extra):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(LAUNCHER_VARS)
           and k not in mpi._MPI_THREAD_PIN_VARS}  # pylint: disable=protected-access
    env.update(PYTHONPATH=str(REPO), **extra)
    return env


class TestDetection(unittest.TestCase):

    def test_counts(self):
        cases = {
            (): False,
            (('SLURM_NTASKS', '1'),): False,
            (('SLURM_NTASKS', '40'),): True,
            (('PMI_SIZE', '2'),): True,
            (('OMPI_COMM_WORLD_SIZE', '1'),): False,
            (('SLURM_NTASKS', 'x'),): False,
        }
        for env, expected in cases.items():
            with self.subTest(env=env), mock.patch.dict(
                    os.environ, _clean_env(**dict(env)), clear=True):
                self.assertEqual(mpi._is_mpi_env_detected(), expected)  # pylint: disable=protected-access


class TestImportSideEffects(unittest.TestCase):

    def setUp(self):
        if not multitile.is_manager():
            self.skipTest('subprocesses run from rank 0')

    def _probe(self, **env):
        proc = subprocess.run(
            [sys.executable, '-c', PROBE], cwd=REPO, env=_clean_env(**env),
            capture_output=True, text=True, timeout=120, check=True)
        return proc.stdout.split()

    def test_serial_slurm_job_untouched(self):
        threads, method = self._probe(SLURM_NTASKS='1')
        self.assertEqual(threads, 'None')
        if sys.platform == 'linux':   # elsewhere 'spawn' is the default
            self.assertNotEqual(method, 'spawn')

    def test_multi_task_job_pinned(self):
        threads, method = self._probe(SLURM_NTASKS='4')
        self.assertEqual((threads, method), ('1', 'spawn'))

    def test_user_thread_setting_kept(self):
        threads, _ = self._probe(SLURM_NTASKS='4', OMP_NUM_THREADS='8')
        self.assertEqual(threads, '8')


if __name__ == '__main__':
    unittest.main()
