#!/bin/bash
# =============================================================================
#  Config E: fully MPI-dispatched pipeline (no patch)
#
#  Every stage in this config is MPI-distributed across worker ranks:
#    - add_contour              (geometry, uses intra-rank thread pool)
#    - add_subtidal_flow_limiter (pure NumPy, inter-tile MPI only)
#    - add_constant_value       (pure NumPy, inter-tile MPI only)
#    - add_channel              (geometry, uses intra-rank thread pool)
#    - add_topo_bound_constraint (TopoConstConstraint, pure NumPy, MPI)
#    - add_topo_func_constraint  (TopoFuncConstraint, named fn, MPI)
#
#  add_patch is intentionally excluded — it is coordinator-only (serial on
#  Rank 0) and would stall all workers, masking hybrid vs no_pool differences.
#
#  This config lets mpi_hybrid show its true advantage: with a 4-core thread
#  pool per rank, the geometry steps (contours + channels) run faster per tile.
#
#  ISOLATION STRATEGY:
#    Phase 1 — Quick correctness check: all modes in one mpiexec call
#              with small tiles to confirm raster-exact matching between
#              serial_mp and MPI modes (OK / FAIL).
#    Phase 2 — Fair timing: each mode in a SEPARATE process invocation
#              with sync+sleep(120) between runs to flush OS page cache,
#              release memory, and clear Python allocator state.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${SCRIPT_DIR}/benchmark_e2e_E.py"

# Generate timestamp for logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${SCRIPT_DIR}/log_e2e_E_${TIMESTAMP}.txt"

# Tee all output (stdout & stderr) to both console and the timestamped log file
exec > >(tee -a "${LOG_FILE}") 2>&1

# Intel MPI environment
export I_MPI_PIN=0            # Unpin ranks so Pool workers can use all cores
export I_MPI_FABRICS=shm      # Shared memory fabric for single-node
export I_MPI_WAIT_MODE=1      # Yield/sleep when idle instead of busy-spinning

echo "========================================="
echo " Config E Benchmark — ${TIMESTAMP}"
echo " Logging to: ${LOG_FILE}"
echo " Node: $(hostname)  Cores: $(nproc)"
echo " I_MPI_PIN=$I_MPI_PIN  I_MPI_FABRICS=$I_MPI_FABRICS  I_MPI_WAIT_MODE=$I_MPI_WAIT_MODE"
echo "========================================="


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 1: CORRECTNESS VALIDATION
#  All modes run together in one process with small tiles.
#  Fast (~30s).  Confirms raster-exact matching (OK / FAIL).
#  serial_mp runs on rank 0 only; MPI modes run collectively — both correct.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " PHASE 1: Correctness validation (4 tiles, 300x300, 5 ranks)"
echo "================================================================"
PYTHONWARNINGS=ignore mpiexec -n 5 python "${BENCH}" \
    --mode all --tiles 4 --size 300 \
    --json "${SCRIPT_DIR}/results_E_correctness_${TIMESTAMP}.json"

sync; sleep 5


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE 2: FAIR TIMING BENCHMARKS
#  Each mode in a SEPARATE process for a clean memory & allocator state.
#  120s sleep between runs ensures OS flushes dirty pages and cgroup
#  memory pressure from the previous run fully subsides.
# ─────────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------
# VARIANT 1: 15 Tiles (1500x1500), 16 ranks (1 core/rank)
#   At 1 core/rank, mpi_hybrid degenerates to mpi_no_pool —
#   no room for a per-rank Pool. Still validates MPI scaling
#   and exercises the full coordinator + 15 worker path.
# ---------------------------------------------------------
SIZE_15=1500
TILES_15=15
RANKS_15=16

echo ""
echo "================================================================"
echo " PHASE 2 — VARIANT 1: 15 tiles (${SIZE_15}x${SIZE_15}), ${RANKS_15} ranks"
echo "================================================================"

echo ""
echo "--- [V1] serial_mp ---"
PYTHONWARNINGS=ignore python "${BENCH}" \
    --mode serial_mp --tiles $TILES_15 --size $SIZE_15 \
    --json "${SCRIPT_DIR}/results_E_15t_serial_${TIMESTAMP}.json"
echo "  [sleep 120s cooldown]"
sync; sleep 120

echo ""
echo "--- [V1] mpi_no_pool (${RANKS_15} ranks, 1 core/rank) ---"
PYTHONWARNINGS=ignore mpiexec -n $RANKS_15 python "${BENCH}" \
    --mode mpi_no_pool --tiles $TILES_15 --size $SIZE_15 \
    --json "${SCRIPT_DIR}/results_E_15t_npool_${TIMESTAMP}.json"
echo "  [sleep 120s cooldown]"
sync; sleep 120

echo ""
echo "--- [V1] mpi_hybrid (${RANKS_15} ranks, 1 core/rank — degenerates to no_pool) ---"
PYTHONWARNINGS=ignore mpiexec -n $RANKS_15 python "${BENCH}" \
    --mode mpi_hybrid --tiles $TILES_15 --size $SIZE_15 \
    --json "${SCRIPT_DIR}/results_E_15t_hybrid_${TIMESTAMP}.json"
echo "  [sleep 120s cooldown]"
sync; sleep 120


# ---------------------------------------------------------
# VARIANT 2: 4 Heavy Tiles (3000x3000), 5 ranks (4 cores/rank)
#   True hybrid config: each worker rank gets ~4 cores for its
#   Pool, showing the real intra-tile parallelism speedup.
# ---------------------------------------------------------
SIZE_HEAVY=3000
TILES_HEAVY=4
RANKS_HEAVY=5

echo ""
echo "================================================================"
echo " PHASE 2 — VARIANT 2: 4 heavy tiles (${SIZE_HEAVY}x${SIZE_HEAVY}), ${RANKS_HEAVY} ranks"
echo "================================================================"

echo ""
echo "--- [V2] serial_mp ---"
PYTHONWARNINGS=ignore python "${BENCH}" \
    --mode serial_mp --tiles $TILES_HEAVY --size $SIZE_HEAVY \
    --json "${SCRIPT_DIR}/results_E_4t_serial_${TIMESTAMP}.json"
echo "  [sleep 120s cooldown]"
sync; sleep 120

echo ""
echo "--- [V2] mpi_no_pool (${RANKS_HEAVY} ranks, 1 core/rank) ---"
PYTHONWARNINGS=ignore mpiexec -n $RANKS_HEAVY python "${BENCH}" \
    --mode mpi_no_pool --tiles $TILES_HEAVY --size $SIZE_HEAVY \
    --json "${SCRIPT_DIR}/results_E_4t_npool_${TIMESTAMP}.json"
echo "  [sleep 120s cooldown]"
sync; sleep 120

echo ""
echo "--- [V2] mpi_hybrid (${RANKS_HEAVY} ranks, auto cores/rank) ---"
PYTHONWARNINGS=ignore mpiexec -n $RANKS_HEAVY python "${BENCH}" \
    --mode mpi_hybrid --tiles $TILES_HEAVY --size $SIZE_HEAVY \
    --json "${SCRIPT_DIR}/results_E_4t_hybrid_${TIMESTAMP}.json"
echo "  [sleep 120s cooldown]"
sync; sleep 120


# ─────────────────────────────────────────────────────────────────────────────
#  TIMING SUMMARY — reads all JSON files and prints a comparison table
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " TIMING SUMMARY"
echo "================================================================"

python3 - "${SCRIPT_DIR}" "${TIMESTAMP}" "E" <<'PYEOF'
import json, sys, os

script_dir = sys.argv[1]
ts         = sys.argv[2]
cfg        = sys.argv[3]

variants = [
    ("15t", "15 tiles (1500x1500), 16 ranks (1 core/rank)"),
    ("4t",  "4 heavy tiles (3000x3000), 5 ranks (~4 cores/rank)"),
]
modes = [
    ("serial", "serial_mp"),
    ("npool",  "mpi_no_pool"),
    ("hybrid", "mpi_hybrid"),
]

def read_wall(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    for v in d.get("results", {}).values():
        return v.get("wall_s")
    return None

for tag, desc in variants:
    print(f"\n  === {desc} ===")
    walls = {}
    for file_tag, mode_name in modes:
        path = os.path.join(script_dir, f"results_{cfg}_{tag}_{file_tag}_{ts}.json")
        w = read_wall(path)
        walls[mode_name] = w
        if w is not None:
            line = f"    {mode_name:15s}  {w:8.1f}s"
            if walls.get("serial_mp") and mode_name != "serial_mp":
                line += f"   speedup {walls['serial_mp'] / w:.2f}x"
            print(line)
        else:
            print(f"    {mode_name:15s}  (no data)")

    np_w = walls.get("mpi_no_pool")
    hy_w = walls.get("mpi_hybrid")
    if np_w and hy_w and tag == "15t":
        ratio = hy_w / np_w
        if 0.85 <= ratio <= 1.15:
            print(f"    ✓ mpi_hybrid/mpi_no_pool = {ratio:.2f}x (within 15% — expected at 1 core/rank)")
        else:
            print(f"    ⚠ mpi_hybrid/mpi_no_pool = {ratio:.2f}x (>15% gap — investigate)")

print()
PYEOF

echo ""
echo "Done. Log: ${LOG_FILE}"
echo "JSON files: ${SCRIPT_DIR}/results_E_*_${TIMESTAMP}.json"
