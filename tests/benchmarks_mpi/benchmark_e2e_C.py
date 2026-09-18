"""Hybrid MPI + Pool benchmark — Config C (contours + channels + flow limiter + const val).

Same structure as benchmark_hybrid_A.py. See that file for full docs.

# DIFF from Config B: adds flow limiter and constant value.
# Flow limiters and const_val are already parallelized via the
# 3-phase file-path pattern. This config tests that they work
# correctly alongside the hybrid MPI + Pool contour path.

One command is enough — tiles and cores/rank are auto-detected:
    mpiexec -n 16 python tests/benchmarks/benchmark_hybrid_C.py
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path
import shutil

import numpy as np

import ocsmesh
from ocsmesh.hfun.raster import HfunRaster

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hybrid_util as hu  # noqa: E402


HMIN = 200
HMAX = 5000
CONFIG = 'C'
CONFIG_DESC = 'contours + channels + flow limiter + const val'


def make_tiles(out_dir, n_tiles, size):
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


def build_and_run(tile_paths, nprocs, execution_mode):
    """Build collector with config C refinements and run all stages."""

    hfun = ocsmesh.Hfun(
        [str(p) for p in tile_paths],
        hmin=HMIN, hmax=HMAX, nprocs=nprocs, method='exact')
    hfun.execution_mode = execution_mode

    # Config C: contours + channels + flow limiter + const val
    hfun.add_contour(level=0, expansion_rate=0.005, target_size=500)
    hfun.add_channel(
        level=0, width=2000, target_size=500, expansion_rate=0.005)
    # DIFF from B: flow limiter and constant value added
    hfun.add_subtidal_flow_limiter(hmin=HMIN, hmax=HMAX)
    hfun.add_constant_value(value=2000, lower_bound=-20, upper_bound=-10)

    stages = {}
    start = time.perf_counter()
    msh = hfun.msh_t()
    stages['msh_t'] = time.perf_counter() - start
    stages['total'] = stages['msh_t']

    if msh is not None:
        values = [np.array(msh.values, copy=True)]
    else:
        values = []
    del hfun
    gc.collect()
    return stages, values


def run_serial_mp(tile_paths, nprocs):
    return build_and_run(tile_paths, nprocs, 'serial')


def run_mpi(tile_paths, nprocs):
    # Every rank must enter: MPIExecutor.run() is a collective call.
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
    parts = [f'{k}: {v:.2f}s' for k, v in s.items()]
    return '  '.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    hu.add_common_args(parser)
    args = parser.parse_args()

    comm = hu.get_comm()
    rank = hu.comm_rank(comm)
    size = hu.comm_size(comm)
    n_tiles = hu.resolve_tiles(comm, args.tiles)
    plan = hu.plan_cores(comm, args.cores_per_rank or None)
    serial_nprocs = args.nprocs or hu.affinity_cores()

    tdir = hu.shared_tmpdir(comm, 'ocsmesh_hybridC_')
    tile_paths = hu.tile_paths(tdir, n_tiles)
    try:
        if rank == 0:
            print(f'=== Config {CONFIG}: {CONFIG_DESC} ===')
            print(f'Tiles: {n_tiles}, size: {args.size}x{args.size}')
            print(f'Start method: {__import__("multiprocessing").get_start_method()}')
            print(hu.format_plan(plan))
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

        # Force cleanup of HfunCollector temp dirs before next mode
        gc.collect()
        if size > 1:
            comm.Barrier()

        if args.mode in ('mpi_no_pool', 'all') and size > 1:
            if rank == 0:
                print('\n--- mpi_no_pool (1 core/rank) ---')
            with hu.CpuMeter(comm, cores=size - 1, collective=True) as meter:
                s, v = run_mpi(tile_paths, 1)
            if rank == 0:
                results['mpi_no_pool'] = {'stages': s, 'values': v,
                                          'meter': meter}
                print(f'  {_fmt_stages(s)}')
                print(f'  {meter.format()}')

        # Force cleanup of HfunCollector temp dirs before next mode
        gc.collect()
        if size > 1:
            comm.Barrier()

        if args.mode in ('mpi_hybrid', 'all') and size > 1:
            if rank == 0:
                print(f'\n--- mpi_hybrid '
                      f'({plan["cores_per_rank"]} cores/rank) ---')
            with hu.CpuMeter(comm, cores=plan['total_cores_used'],
                             collective=True) as meter:
                s, v = run_mpi(tile_paths, plan['cores_per_rank'])
            if rank == 0:
                results['mpi_hybrid'] = {'stages': s, 'values': v,
                                         'meter': meter}
                print(f'  {_fmt_stages(s)}')
                print(f'  {meter.format()}')

        if rank == 0 and size == 1 and args.mode != 'serial_mp':
            print('\nMPI modes skipped: launch with '
                  '`mpiexec -n <tiles+1> ...` to run them.')

        if rank == 0 and len(results) > 1:
            print('\n=== Comparison ===')
            baseline_key = 'serial_mp' if 'serial_mp' in results else list(results.keys())[0]
            for key, data in results.items():
                if key == baseline_key:
                    continue
                problems = compare_values(
                    results[baseline_key]['values'], data['values'],
                    baseline_key, key)
                status = 'OK' if not problems else 'FAIL'
                speedup = (results[baseline_key]['meter'].wall
                           / data['meter'].wall)
                print(f'{status}  {baseline_key} vs {key}: '
                      f'speedup {speedup:.2f}x')
                for p in problems:
                    print(f'  {p}')

        if rank == 0 and args.json:
            out = {k: dict(v['meter'].as_dict(), stages=v['stages'])
                   for k, v in results.items()}
            args.json.write_text(json.dumps({
                'config': CONFIG, 'tiles': n_tiles, 'plan': plan,
                'results': out}, indent=2))

    finally:
        gc.collect()
        if size > 1:
            comm.Barrier()
        if rank == 0:
            shutil.rmtree(tdir, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main() or 0)
