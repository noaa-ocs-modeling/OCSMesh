"""Hybrid MPI + Pool e2e benchmark — Config E (fully MPI-dispatched pipeline).

Same structure as benchmark_e2e_D.py. See that file for full docs.

Config E is specifically designed to benchmark the fully MPI-parallel pipeline
with NO coordinator-only bottleneck (i.e. no add_patch):

  Stage                         MPI dispatch?
  ─────────────────────────────────────────────
  add_contour                   YES — all ranks
  add_subtidal_flow_limiter     YES — all ranks
  add_constant_value            YES — all ranks
  add_channel                   YES — all ranks
  add_topo_bound_constraint     YES — all ranks  (TopoConstConstraint)
  add_topo_func_constraint      YES — all ranks  (TopoFuncConstraint, named fn)
  add_patch                     NOT INCLUDED     (coordinator-only bottleneck)

With no patch step, all worker cores are busy for the entire pipeline,
allowing mpi_hybrid to show its true advantage over mpi_no_pool.
"""

import argparse
import gc
import json
import sys
import time
import shutil
from pathlib import Path

import numpy as np

import ocsmesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hybrid_util as hu  # noqa: E402


HMIN = 200
HMAX = 5000
CONFIG = 'E'
CONFIG_DESC = (
    'fully MPI pipeline — contours + channels + flow limiter + const val '
    '+ topo-bound constraint + topo-func constraint (no patch)'
)


def make_tiles(out_dir, n_tiles, size):
    """Write n_tiles DEM tiles side by side."""
    paths = []
    span = 1.0
    overlap = 0.1
    for i in range(n_tiles):
        x0 = i * (span - overlap)
        x1 = x0 + span
        gx, gy = np.mgrid[x0:x1:complex(0, size), 0:1:complex(0, size)]
        z = (gy * 40.0) - 20.0 + 3.0 * np.sin(gx * 6.0)
        path = Path(out_dir) / f'dem_{i}.tif'
        ocsmesh.utils.raster_from_numpy(path, z, (gx, gy), 4326)
        paths.append(path)
    return paths


def _half_depth(depth):
    """Named function for TopoFuncConstraint — returns depth / 2.

    Must be defined at module level (not a lambda) so it can be
    pickled for MPI and multiprocessing dispatch.
    """
    return depth / 2.0


def build_and_run(tile_paths, nprocs, execution_mode):
    """Build collector with config E refinements and run full msh_t().

    All steps in this config are MPI-dispatched. No add_patch is used,
    so there is no coordinator-only serialization bottleneck.
    """
    hfun = ocsmesh.Hfun(
        [str(p) for p in tile_paths],
        hmin=HMIN, hmax=HMAX, nprocs=nprocs, method='exact')
    hfun.execution_mode = execution_mode

    # ── All steps below are fully MPI-dispatched ──────────────────────

    # 1. Contours — geometry-based, uses intra-rank thread pool
    hfun.add_contour(level=0, expansion_rate=0.005, target_size=500)

    # 2. Flow limiter — pure NumPy, inter-tile MPI only
    hfun.add_subtidal_flow_limiter(hmin=HMIN, hmax=HMAX)

    # 3. Constant value — pure NumPy, inter-tile MPI only
    hfun.add_constant_value(value=2000, lower_bound=-20, upper_bound=-10)

    # 4. Channels — geometry-based, uses intra-rank thread pool
    hfun.add_channel(level=0, width=2000, target_size=500, expansion_rate=0.005)

    # 5. TopoConstConstraint — pure NumPy, inter-tile MPI
    hfun.add_topo_bound_constraint(
        value=800, upper_bound=0, lower_bound=-20, value_type='min')

    # 6. TopoFuncConstraint — named function (pickleable), inter-tile MPI
    hfun.add_topo_func_constraint(
        func=_half_depth,
        upper_bound=0, lower_bound=-20,
        value_type='min', rate=0.01)

    start = time.perf_counter()
    msh = hfun.msh_t()
    elapsed = time.perf_counter() - start

    stages = {'msh_t': elapsed, 'total': elapsed}

    values = [np.array(msh.values, copy=True)] if msh is not None else []
    del hfun
    gc.collect()
    return stages, values


def run_serial_mp(tile_paths, nprocs):
    return build_and_run(tile_paths, nprocs, 'serial')


def run_mpi(tile_paths, nprocs):
    return build_and_run(tile_paths, nprocs, 'mpi')


def compare_values(values_a, values_b, label_a, label_b):
    if len(values_a) != len(values_b):
        return [f'{label_b}: {len(values_b)} rasters vs {len(values_a)}']
    problems = []
    for i, (a, b) in enumerate(zip(values_a, values_b)):
        if not np.array_equal(a, b, equal_nan=True):
            diff = np.abs(np.nan_to_num(a) - np.nan_to_num(b))
            problems.append(
                f'raster {i}: {int((diff > 0).sum())} px differ, '
                f'max diff {float(diff.max()):.6g}')
    return problems


def _fmt_stages(s):
    return '  '.join(f'{k}: {v:.2f}s' for k, v in s.items())


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    hu.add_common_args(parser)
    args = parser.parse_args()

    comm = hu.get_comm()
    rank = hu.comm_rank(comm)
    size = hu.comm_size(comm)
    n_tiles = hu.resolve_tiles(comm, args.tiles)
    plan = hu.plan_cores(comm, args.cores_per_rank or None)
    serial_nprocs = args.nprocs or hu.affinity_cores()

    tdir = hu.shared_tmpdir(comm, 'ocsmesh_e2eE_')
    tile_paths = hu.tile_paths(tdir, n_tiles)
    try:
        if rank == 0:
            print(f'=== Config {CONFIG}: {CONFIG_DESC} ===')
            print(f'Tiles: {n_tiles}, size: {args.size}x{args.size}')
            print(f'Start method: {__import__("multiprocessing").get_start_method()}')
            print(hu.format_plan(plan))
            print(f'Creating tiles in {tdir} ...')
            make_tiles(tdir, n_tiles, args.size)
        if size > 1:
            comm.Barrier()

        results = {}

        if args.mode in ('serial_mp', 'all') and rank == 0:
            print(f'\n--- serial_mp (Pool nprocs={serial_nprocs}) ---')
            with hu.CpuMeter(cores=serial_nprocs) as meter:
                s, v = run_serial_mp(tile_paths, serial_nprocs)
            results['serial_mp'] = {'stages': s, 'values': v, 'meter': meter}
            print(f'  {_fmt_stages(s)}')
            print(f'  {meter.format()}')

        gc.collect()
        if size > 1:
            comm.Barrier()

        if args.mode in ('mpi_no_pool', 'all') and size > 1:
            if rank == 0:
                print('\n--- mpi_no_pool (1 core/rank) ---')
            with hu.CpuMeter(comm, cores=size - 1, collective=True) as meter:
                s, v = run_mpi(tile_paths, 1)
            if rank == 0:
                results['mpi_no_pool'] = {'stages': s, 'values': v, 'meter': meter}
                print(f'  {_fmt_stages(s)}')
                print(f'  {meter.format()}')

        gc.collect()
        if size > 1:
            comm.Barrier()

        if args.mode in ('mpi_hybrid', 'all') and size > 1:
            if rank == 0:
                print(f'\n--- mpi_hybrid ({plan["cores_per_rank"]} cores/rank) ---')
            with hu.CpuMeter(comm, cores=plan['total_cores_used'],
                             collective=True) as meter:
                s, v = run_mpi(tile_paths, plan['cores_per_rank'])
            if rank == 0:
                results['mpi_hybrid'] = {'stages': s, 'values': v, 'meter': meter}
                print(f'  {_fmt_stages(s)}')
                print(f'  {meter.format()}')

        if rank == 0 and size == 1 and args.mode != 'serial_mp':
            print('\nMPI modes skipped: launch with `mpiexec -n <tiles+1> ...`')

        if rank == 0 and len(results) > 1:
            print('\n=== Correctness ===')
            baseline_key = 'serial_mp' if 'serial_mp' in results else list(results.keys())[0]
            baseline_v = results[baseline_key]['values']
            for key, data in results.items():
                if key == baseline_key:
                    continue
                problems = compare_values(baseline_v, data['values'], baseline_key, key)
                if problems:
                    print(f'FAIL {baseline_key} vs {key}:')
                    for p in problems:
                        print(f'  {p}')
                else:
                    print(f'OK   {baseline_key} vs {key}: values match exactly')
                speedup = results[baseline_key]['meter'].wall / data['meter'].wall
                print(f'     Speedup: {speedup:.2f}x')

        if rank == 0 and args.json:
            out = {k: dict(v['meter'].as_dict(), stages=v['stages'])
                   for k, v in results.items()}
            args.json.write_text(json.dumps({
                'config': CONFIG,
                'tiles': n_tiles,
                'size': args.size,
                'plan': plan,
                'results': out,
            }, indent=2))
            print(f'\nWrote {args.json}')

    finally:
        gc.collect()
        if size > 1:
            comm.Barrier()
        if rank == 0:
            shutil.rmtree(tdir, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main() or 0)
