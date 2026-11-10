"""HfunCollector cleanup at interpreter exit.

``HfunCollector.__del__`` imported ``ocsmesh.mpi`` to decide whether
this rank owns the work directory. When the collector lived until the
interpreter exited, that import failed ("import of ocsmesh.mpi halted")
and Python printed "Exception ignored in ... __del__" after every run.
"""

import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

from tests.api import multitile

REPO = Path(__file__).resolve().parents[2]

SCRIPT = textwrap.dedent("""
    from tests.api import multitile
    stack = multitile.build_stack(scale=4.0, include_empty=False)
    HFUN = multitile.make_hfun(stack.paths[:2], 'serial')
    HFUN.meshdata()
    print(HFUN._work_dir)
""")


class TestInterpreterExit(unittest.TestCase):

    def test_no_error_and_work_dir_removed(self):
        if not multitile.is_manager():
            self.skipTest('runs one subprocess from rank 0')
        env = dict(os.environ, PYTHONPATH=str(REPO))
        # a child of an MPI rank must not pick up the parent's MPI job
        for key in [k for k in env if k.startswith(('PMI', 'OMPI', 'MPI'))]:
            del env[key]
        proc = subprocess.run(
            [sys.executable, '-c', SCRIPT], cwd=REPO, env=env,
            capture_output=True, text=True, timeout=300, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn('Exception ignored', proc.stderr)
        work_dir = proc.stdout.strip().splitlines()[-1]
        self.assertFalse(os.path.exists(work_dir), work_dir)


if __name__ == '__main__':
    unittest.main()
