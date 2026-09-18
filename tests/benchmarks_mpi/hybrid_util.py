"""Shared helpers for the hybrid MPI + Pool benchmarks (configs A-D).

Two jobs:

1. **Rank-aware core budgeting.** Under plain ``mpiexec -n 16`` every rank
   sees the *whole* machine in its affinity mask, so naively calling
   ``os.sched_getaffinity(0)`` makes all 15 workers spawn a full-size Pool
   and oversubscribe the node ~15x. :func:`plan_cores` divides the node's
   cores by the number of ranks actually sharing that node, so a single
   ``mpiexec -n N`` command is enough — no ``--cpus-per-task`` needed.

2. **Timing + CPU utilization accounting.** :class:`CpuMeter` measures wall
   time and CPU seconds (this process + reaped children, i.e. Pool workers)
   summed across every MPI rank, and reports how many cores were actually
   busy versus how many were allocated.
"""

import os
import tempfile
import time
from pathlib import Path


def get_comm():
    """Return ``MPI.COMM_WORLD``, or None when mpi4py is unavailable."""
    try:
        from mpi4py import MPI
    except ImportError:
        return None
    return MPI.COMM_WORLD


def comm_rank(comm):
    return comm.Get_rank() if comm is not None else 0


def comm_size(comm):
    return comm.Get_size() if comm is not None else 1


def affinity_cores():
    """Cores this process is allowed to run on (respects SLURM/taskset)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def node_local_size(comm):
    """Number of ranks sharing this physical node."""
    if comm is None or comm.Get_size() == 1:
        return 1
    from mpi4py import MPI
    node = comm.Split_type(MPI.COMM_TYPE_SHARED)
    try:
        return node.Get_size()
    finally:
        node.Free()


def node_cores(comm):
    """Cores usable on this node by this job, and ranks sharing the node.

    The union of every local rank's affinity mask. When the launcher pins
    ranks to disjoint core subsets, each rank's own mask is only its
    slice, so the union is what the job actually owns on the node.
    """
    try:
        mask = set(os.sched_getaffinity(0))
    except AttributeError:
        mask = set(range(os.cpu_count() or 1))
    if comm is None or comm.Get_size() == 1:
        return len(mask), 1
    from mpi4py import MPI
    node = comm.Split_type(MPI.COMM_TYPE_SHARED)
    try:
        masks = node.allgather(mask)
        local = node.Get_size()
    finally:
        node.Free()
    return len(set().union(*masks)), local


def plan_cores(comm, override=None):
    """Decide how many Pool workers this rank may start.

    Resolution order:

    ``override``            explicit ``--cores-per-rank``
    ``SLURM_CPUS_PER_TASK`` the launcher already sized the rank
    bound mask             this rank owns fewer cores than the node has,
                           so the launcher already pinned it
    divided                node cores / worker ranks on this node

    Rank 0 is a coordinator and runs no tiles, so it is excluded from the
    divisor when it shares the node with workers.

    Returns a dict with the plan and enough context to print it.
    """
    mine = affinity_cores()
    on_node, local = node_cores(comm)
    size = comm_size(comm)
    workers_here = max(1, local - 1) if local > 1 else 1

    if override:
        cores, source = max(1, int(override)), 'cli --cores-per-rank'
    elif os.environ.get('SLURM_CPUS_PER_TASK'):
        cores = max(1, int(os.environ['SLURM_CPUS_PER_TASK']))
        source = 'SLURM_CPUS_PER_TASK'
    elif size == 1:
        cores, source = mine, 'affinity (no MPI)'
    elif mine < on_node:
        # Launcher already pinned each rank to a disjoint core subset.
        cores, source = mine, 'affinity (launcher-bound)'
    else:
        cores = max(1, on_node // workers_here)
        source = f'auto ({on_node} cores / {workers_here} worker ranks on node)'

    return {
        'cores_per_rank': cores,
        'affinity_cores': mine,
        'node_cores': on_node,
        'ranks_on_node': local,
        'worker_ranks_on_node': workers_here,
        'world_size': size,
        'source': source,
        'total_cores_used': cores * max(1, size - 1) if size > 1 else cores,
    }


def format_plan(plan):
    lines = [
        f"  cores/rank: {plan['cores_per_rank']}  [{plan['source']}]",
        f"  ranks: {plan['world_size']} "
        f"({plan['world_size'] - 1 if plan['world_size'] > 1 else 1} workers), "
        f"node cores: {plan['node_cores']}, "
        f"cores in flight: {plan['total_cores_used']}",
    ]
    if plan['world_size'] > 1 and plan['cores_per_rank'] == 1:
        best = max(2, plan['node_cores'] // 4)
        lines.append(
            f"  WARNING: 1 core/rank -> mpi_hybrid degenerates into "
            f"mpi_no_pool. You asked for {plan['world_size'] - 1} worker "
            f"ranks on {plan['node_cores']} cores. For a real hybrid "
            f"test use fewer, fatter ranks, e.g. `-n {best + 1}` "
            f"({best} tiles x {plan['node_cores'] // best} cores).")
    if plan['total_cores_used'] > plan['node_cores']:
        lines.append(
            f"  WARNING: {plan['total_cores_used']} cores requested on a "
            f"{plan['node_cores']}-core node - oversubscribed, timings "
            f"will be pessimistic.")
    if plan['affinity_cores'] < plan['node_cores']:
        lines.append(
            f"  NOTE: rank 0 is pinned to {plan['affinity_cores']} of "
            f"{plan['node_cores']} cores, so the serial_mp baseline cannot "
            f"use the whole node. Run `--mode serial_mp` without a launcher "
            f"for a fair baseline.")
    return '\n'.join(lines)


def resolve_tiles(comm, requested):
    """``--tiles 0`` (default) means one tile per worker rank."""
    if requested and requested > 0:
        return requested
    size = comm_size(comm)
    return max(1, size - 1) if size > 1 else 4


def _cpu_seconds():
    """CPU seconds burned by this process plus its reaped children.

    Pool workers are joined when the Pool closes, so their time lands in
    the ``children_*`` fields.
    """
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


class CpuMeter:
    """Context manager measuring wall time and cluster-wide CPU utilization.

    ``cores`` is the number of cores allocated to the whole run, so
    ``utilization`` answers "did we actually use what we asked for?".
    """

    def __init__(self, comm=None, cores=1, collective=False):
        self.comm = comm if collective and comm_size(comm) > 1 else None
        self.cores = max(1, cores)
        self.wall = 0.0
        self.cpu = 0.0
        self.busy_cores = 0.0
        self.utilization = 0.0

    def __enter__(self):
        if self.comm is not None:
            self.comm.Barrier()
        self._t0 = time.perf_counter()
        self._c0 = _cpu_seconds()
        return self

    def __exit__(self, *exc):
        cpu = _cpu_seconds() - self._c0
        if self.comm is not None:
            self.comm.Barrier()
            cpu = self.comm.allreduce(cpu)
        self.wall = time.perf_counter() - self._t0
        self.cpu = cpu
        self.busy_cores = cpu / self.wall if self.wall > 0 else 0.0
        self.utilization = self.busy_cores / self.cores
        return False

    def as_dict(self):
        return {
            'wall_s': self.wall,
            'cpu_s': self.cpu,
            'cores_allocated': self.cores,
            'busy_cores': self.busy_cores,
            'utilization': self.utilization,
        }

    def format(self):
        # Idle MPI ranks busy-wait inside recv/Barrier, and that spin is
        # charged as CPU time. Set OMPI_MCA_mpi_yield_when_idle=1 to damp it.
        return (f'wall {self.wall:.2f}s | cpu {self.cpu:.1f}s | '
                f'busy {self.busy_cores:.1f}/{self.cores} cores | '
                f'utilization {100 * self.utilization:.0f}% '
                f'(incl. MPI idle-spin)')


def shared_tmpdir(comm, prefix):
    """Rank 0 makes the dir, every rank gets the same path."""
    tdir = tempfile.mkdtemp(prefix=prefix) if comm_rank(comm) == 0 else None
    if comm_size(comm) > 1:
        tdir = comm.bcast(tdir, root=0)
    return Path(tdir)


def tile_paths(out_dir, n_tiles):
    return [Path(out_dir) / f'dem_{i}.tif' for i in range(n_tiles)]


def add_common_args(parser):
    parser.add_argument('--mode',
                        choices=['serial_mp', 'mpi_no_pool', 'mpi_hybrid',
                                 'all'],
                        default='all')
    parser.add_argument('--tiles', type=int, default=0,
                        help='0 = auto (one tile per worker rank)')
    parser.add_argument('--size', type=int, default=120)
    parser.add_argument('--nprocs', type=int, default=0,
                        help='serial_mp Pool size; 0 = all available cores')
    parser.add_argument('--cores-per-rank', type=int, default=0,
                        help='override auto core budget for mpi_hybrid')
    parser.add_argument('--json', type=Path, default=None)
    return parser
