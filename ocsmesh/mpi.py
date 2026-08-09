"""MPI infrastructure for OCSMesh.

Provides :class:`MPIExecutor` — a Singleton manager/worker coordinator
that any collector (HfunCollector, GeomCollector, etc.) can use for
distributed task execution on HPC clusters.

Typical collector usage::

    from ocsmesh.mpi import MPIExecutor

    # Register domain-specific worker functions at import time
    MPIExecutor.register_op('meshdata', _meshdata_task_worker)

    # Inside the collector method (all ranks call collectively):
    results = MPIExecutor.run(tasks, work_dir=self._work_dir)

``run()`` is the only public entry point for dispatching tasks.
``_execute()`` and ``_submit()`` are private — calling them directly
bypasses the worker recv loop setup and causes deadlocks.

Users never interact with this module directly. They simply set
``hfun.execution_mode = 'mpi'`` and call ``hfun.meshdata()``.
"""

import logging
import os
import sys
import traceback

_logger = logging.getLogger(__name__)

# pylint: disable=c-extension-no-member,import-outside-toplevel

# ── MPI lazy import ────────────────────────────────────────────────
# mpi4py is an optional dependency. These helpers let the rest of
# the module work without it (graceful fallback to serial).

_MPI = None
_MPI_IMPORT_ATTEMPTED = False
_MPI_ENV_HINTS = (
    'OMPI_COMM_WORLD_SIZE',     # Open MPI
    'PMI_SIZE',                  # MPICH / Cray
    'MPI_LOCALNRANKS',           # Intel MPI
    'SLURM_NTASKS',              # SLURM (srun)
)
_MPI_THREAD_PIN_VARS = (
    'OMP_NUM_THREADS',           # OpenMP (generic)
    'MKL_NUM_THREADS',           # Intel MKL (conda numpy)
    'OPENBLAS_NUM_THREADS',      # OpenBLAS (pip numpy)
)


def _get_mpi():
    """Lazy-import mpi4py.MPI. Returns the module or None."""
    global _MPI, _MPI_IMPORT_ATTEMPTED  # pylint: disable=global-statement
    if not _MPI_IMPORT_ATTEMPTED:
        _MPI_IMPORT_ATTEMPTED = True
        try:
            from mpi4py import MPI
            _MPI = MPI
        except ImportError:
            pass
    return _MPI


def _is_mpi_env_detected():
    """Check if environment variables indicate an MPI launcher (mpiexec, srun, etc.)."""
    return any(var in os.environ for var in _MPI_ENV_HINTS)


def _get_mpi_comm():
    """Return MPI.COMM_WORLD if running under an MPI launcher and mpi4py is available, else None."""
    if not _is_mpi_env_detected():
        return None
    MPI = _get_mpi()
    if MPI is None:
        return None
    return MPI.COMM_WORLD


def _is_mpi_active():
    """Check if we're running under an MPI launcher with >1 rank."""
    comm = _get_mpi_comm()
    if comm is None:
        return False
    try:
        return comm.Get_size() > 1
    except Exception:  # pylint: disable=broad-exception-caught
        return False


def _configure_mpi_environment():
    """Safety net for MPI environment, thread pinning, and multiprocessing.

    Pins numerical library threads (OpenMP/MKL/OpenBLAS) to 1 to prevent
    thread oversubscription when multiple MPI ranks run on the same node,
    and sets the multiprocessing start method to 'spawn' to avoid fork
    deadlocks with open MPI communicators.
    """
    if _is_mpi_env_detected():
        for var in _MPI_THREAD_PIN_VARS:
            os.environ.setdefault(var, '1')

        import multiprocessing as mp
        try:
            mp.set_start_method('spawn', force=False)
        except RuntimeError:
            if mp.get_start_method() != 'spawn':
                import warnings
                warnings.warn(
                    f"multiprocessing start method is '{mp.get_start_method()}', "
                    f"but MPI requires 'spawn' to avoid deadlocks. Call "
                    f"multiprocessing.set_start_method('spawn') before "
                    f"importing ocsmesh.",
                    UserWarning
                )


# ── Generic worker functions ──────────────────────────────────────

def _check_shared_fs_task_worker(task: dict):
    """Worker task that validates cross-node shared filesystem access.

    Verifies that this worker rank can both read a file created by Rank 0
    in ``work_dir`` (proving cross-node read visibility) and create a new
    file in ``work_dir`` (proving write permissions).
    """
    original_index = task.get('original_index', -1)
    work_dir = task['work_dir']
    test_read_file = task['test_read_file']
    worker_rank = task.get('worker_rank', -1)

    try:
        # 1. Check read visibility of Rank 0's test file
        if not (os.path.exists(test_read_file)
                and os.access(test_read_file, os.R_OK)):
            raise FileNotFoundError(
                f"Rank {worker_rank} cannot see/read Rank 0 test file: "
                f"{test_read_file}"
            )

        # 2. Check write permission by creating and removing a test file
        worker_test_file = os.path.join(
            work_dir,
            f".mpi_fs_write_test_rank_{worker_rank}_{os.getpid()}.tmp"
        )
        with open(worker_test_file, 'w') as f:
            f.write(f"write_ok_from_rank_{worker_rank}")
        if os.path.exists(worker_test_file):
            os.remove(worker_test_file)
        else:
            raise FileNotFoundError(
                f"Rank {worker_rank} wrote file {worker_test_file} "
                f"but cannot see it."
            )

        return {
            'status': 'success',
            'original_index': original_index,
            'worker_rank': worker_rank,
        }
    except Exception as e:  # pylint: disable=broad-exception-caught
        return {
            'status': 'error',
            'original_index': original_index,
            'op': task.get('op'),
            'worker_rank': worker_rank,
            'error': repr(e),
            'traceback': traceback.format_exc(),
        }


# ── MPIExecutor ───────────────────────────────────────────────────

class MPIExecutor:
    """Singleton MPI manager/worker coordinator.

    Encapsulates all MPI dispatch logic. Any OCSMesh collector can
    use this executor for distributed task execution.

    Usage inside a collector::

        results = MPIExecutor.run(
            tasks, work_dir=self._work_dir
        )

    ``run()`` is the only public entry point for dispatching tasks.
    ``_execute()`` and ``_submit()`` are private — calling them
    directly bypasses the worker recv loop setup, which caused
    deadlocks (see PR review Issue #1 and #2).

    Features
    --------
    - Singleton: one instance per process, shared across all collectors
    - Extensible: collectors register ops via :meth:`register_op`
    - Dynamic on-demand scheduling (fast ranks pull more work)
    - Rank 0 = dedicated coordinator
    - Soft-fail: worker exceptions -> structured error dicts,
      worker stays alive for more tasks
    - Global excepthook prevents zombie cloud/HPC processes
    - Individual TAG_STOP per worker (no collective shutdown)
    - Sequential fallback when size == 1 or no MPI
    - Fail-fast dispatch: stop sending new tasks on first error

    Adding a new MPI operation requires:
      1. Write a ``_*_task_worker(task)`` function
      2. Register it: ``MPIExecutor.register_op('my_op', _my_op_worker)``
    """

    _instance = None
    _registered_ops = {
        'check_shared_fs': _check_shared_fs_task_worker,
    }

    # Message tags for the manager/worker protocol.
    # Using a dict keeps them grouped and discoverable.
    _TAGS = {
        'TASK':   1,   # rank 0 -> worker : here is a task dict
        'RESULT': 2,   # worker -> rank 0 : task succeeded (result dict)
        'ERROR':  3,   # worker -> rank 0 : task raised / structured failure
        'STOP':   4,   # rank 0 -> worker : no more work, leave the loop
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize MPI state and install global safety net."""
        if getattr(self, '_initialized', False):
            return
        self._initialized = True

        comm = _get_mpi_comm()
        if comm is not None:
            self.comm = comm
            self.rank = comm.Get_rank()
            self.size = comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1

        # Install global safety net for multi-rank jobs
        if self.comm is not None and self.size > 1:
            self.install_mpi_excepthook()

    # ── Rank Queries ──────────────────────────────────────────────

    @classmethod
    def is_manager(cls):
        """Return True if this process is rank 0 (or not running under MPI).

        Convenience check so callers don't need to instantiate the
        singleton just to inspect rank. Safe to call from any context:
        returns True when MPI is not active (serial fallback).
        """
        instance = cls()
        return instance.rank == 0

    # ── Registration ──────────────────────────────────────────────

    @classmethod
    def register_op(cls, name, fn):
        """Register a worker function for the given operation name.

        Collectors call this at module load time to make their
        domain-specific worker functions available to the executor.

        Parameters
        ----------
        name : str
            Operation name (matches the ``'op'`` key in task dicts).
        fn : callable
            Worker function: ``fn(task_dict) -> result_dict``.
        """
        cls._registered_ops[name] = fn

    def _worker_registry(self):
        """Return the current operation -> function mapping."""
        return dict(self._registered_ops)

    # ── Global Safety Net ─────────────────────────────────────────

    @staticmethod
    def install_mpi_excepthook():
        """Install an MPI-aware ``sys.excepthook`` that aborts all ranks.

        When an uncaught exception propagates to the top of the stack on any
        rank, this hook writes a traceback to stderr, chains to the
        **previous** excepthook (so logging frameworks / debuggers still
        fire), and then calls ``comm.Abort(1)`` to kill ALL processes
        across ALL nodes immediately.  This prevents zombie cloud/HPC
        instances that keep running while blocked in a ``recv()``.

        The hook is installed **at most once** (idempotent).  Calling
        this method again after it has already been installed is a no-op.

        Only triggers on truly uncaught exceptions; normal task failures
        are caught by the worker loop and returned as structured error
        dicts.
        """
        # Idempotency guard.
        if getattr(sys.excepthook, '_is_mpi_hook', False):
            return

        MPI = _get_mpi()
        if MPI is None:
            return

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        _previous_hook = sys.excepthook

        def _mpi_excepthook(exctype, value, tb):
            sys.stderr.write(
                f"\n[CRITICAL] Uncaught exception on Rank {rank}:\n"
            )
            traceback.print_exception(exctype, value, tb, file=sys.stderr)
            sys.stderr.flush()
            # Chain to previous hook
            try:
                _previous_hook(exctype, value, tb)
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            comm.Abort(1)

        _mpi_excepthook._is_mpi_hook = True  # pylint: disable=protected-access,attribute-defined-outside-init
        sys.excepthook = _mpi_excepthook

    # ── Public API ────────────────────────────────────────────────

    @classmethod
    def run(cls, tasks, work_dir=None, fail_fast=True):
        """All ranks call collectively. Dispatch tasks, return results.

        Single-call entry point — the only public way to dispatch MPI
        tasks. Internally calls ``_execute()`` + ``_submit()``, which
        are private to prevent misuse (calling them directly bypasses
        worker recv loop setup and causes deadlocks).

        Design note: all ranks execute the calling code *before* this
        method is reached. This is intentional — it keeps every rank
        available for future MPI work in pre-dispatch stages (e.g.
        distributed task building, parallel feature application).

        Parameters
        ----------
        tasks : list of dict
            Task dicts, each with ``'op'`` and ``'original_index'`` keys.
        work_dir : path-like or None, default=None
            If provided, verify shared filesystem before dispatching.
        fail_fast : bool, default=True
            Stop sending new tasks on first error.

        Returns
        -------
        dict or None
            ``{original_index: result_dict}`` on Rank 0; ``None`` on
            worker ranks.
        """
        instance = cls()
        return instance._execute(
            lambda: instance._submit(
                tasks, work_dir=work_dir, fail_fast=fail_fast
            )
        )

    def _execute(self, pipeline_fn):
        """Execute pipeline_fn on Rank 0, worker loop on all other ranks.

        Private — must only be called via :meth:`run`. Calling directly
        requires the caller to guarantee workers enter this method
        collectively, which is error-prone and caused deadlocks

        All ranks must call this collectively.  Rank 0 runs
        ``pipeline_fn()`` inside a try/finally that guarantees
        ``TAG_STOP`` is sent to every worker, regardless of success
        or exception.  Workers enter :meth:`_run_worker` and block
        until they receive ``TAG_STOP``.

        Parameters
        ----------
        pipeline_fn : callable
            A zero-argument callable that contains the Rank-0-only
            pipeline logic. Must call :meth:`_submit` for distributed
            work. The return value is passed through on Rank 0.

        Returns
        -------
        object or None
            Return value of ``pipeline_fn()`` on Rank 0; ``None`` on
            all worker ranks.
        """
        # Non-MPI fallback: just run the callable directly.
        if self.comm is None or self.size == 1:
            if callable(pipeline_fn):
                return pipeline_fn()
            raise TypeError(
                f"MPIExecutor._execute() expects a callable, "
                f"got {type(pipeline_fn).__name__}"
            )

        if self.rank == 0:
            try:
                return pipeline_fn()
            finally:
                self._shutdown_workers()
        else:
            self._run_worker()
            return None

    def _submit(self, tasks, work_dir=None, fail_fast=True):
        """Verify filesystem, dispatch tasks, aggregate results, raise on failure.

        Private — must only be called from within ``_execute()``'s
        pipeline_fn. Calling outside ``_execute()`` means workers
        are not in their recv loop, and ``_dispatch()`` will deadlock


        Parameters
        ----------
        tasks : list of dict
            Task dicts, each with an ``'op'`` key matching a registered
            worker function and an ``'original_index'`` key for result
            reassociation.
        work_dir : path-like or None, default=None
            If provided, run :meth:`verify_shared_filesystem` before
            dispatching to ensure all workers can read/write this path.
        fail_fast : bool, default=True
            Stop dispatching new tasks on first error. In-flight tasks
            (already being computed) are still drained and their results
            collected.

        Returns
        -------
        dict
            Mapping of ``{original_index: result_dict}`` for successes
            only.

        Raises
        ------
        RuntimeError
            If any task failed, with a summary of all failures.
        """
        if work_dir is not None:
            self.verify_shared_filesystem(work_dir)

        _logger.info(
            f"Dispatching {len(tasks)} task(s) to "
            f"{max(self.size - 1, 1)} worker(s)"
        )
        results = self._dispatch(tasks, fail_fast=fail_fast)
        _logger.info("All dispatched tasks returned.")

        successes = {}
        failures = []
        for result in results:
            if not isinstance(result, dict) or result.get('status') == 'error':
                failures.append(result)
                idx = (result.get('original_index', -1)
                       if isinstance(result, dict) else -1)
                err = (result.get('error') if isinstance(result, dict)
                       else repr(result))
                wrk = (result.get('worker_rank', '?')
                       if isinstance(result, dict) else '?')
                _logger.error(
                    f"MPI task failed for index {idx} "
                    f"(worker rank {wrk}): {err}"
                )
            else:
                successes[result['original_index']] = result

        if failures:
            summary = "; ".join(
                f"idx={f.get('original_index', -1)} "
                f"rank={f.get('worker_rank', '?')} "
                f"err={f.get('error', 'unknown')}"
                for f in failures if isinstance(f, dict)
            )
            raise RuntimeError(
                f"{len(failures)} of {len(results)} dispatched MPI "
                f"task(s) failed ({len(tasks)} total); "
                f"aborting. Details: {summary}"
            )

        return successes

    def verify_shared_filesystem(self, work_dir):
        """Rank 0 only: verify all workers can access and write to work_dir.

        On multi-node HPC jobs, default temporary directories (like /tmp)
        live on node-local disks. If workers are on separate physical nodes
        from Rank 0, intermediate file operations will fail. This method
        dispatches lightweight test tasks to confirm cross-node read
        visibility and write permissions before any expensive work begins.

        Parameters
        ----------
        work_dir : path-like
            The working directory path to test across all worker ranks.

        Raises
        ------
        RuntimeError
            If any worker rank fails to read from or write to ``work_dir``.
        """
        if self.comm is None or self.size <= 1:
            return
        if self.rank != 0:
            raise RuntimeError(
                "verify_shared_filesystem() must only be called by Rank 0."
            )

        work_dir_str = str(work_dir)
        pid = os.getpid()
        _logger.info(
            f"Verifying shared filesystem visibility across "
            f"{self.size - 1} worker rank(s)..."
        )
        rank0_test_file = os.path.join(
            work_dir_str, f".mpi_fs_check_{pid}.tmp"
        )
        with open(rank0_test_file, 'w') as f:
            f.write("rank0_ok")

        try:
            check_tasks = [
                {
                    'op': 'check_shared_fs',
                    'original_index': idx,
                    'work_dir': work_dir_str,
                    'test_read_file': rank0_test_file,
                    'worker_rank': idx + 1,
                }
                for idx in range(self.size - 1)
            ]
            check_results = self._dispatch(check_tasks)
        finally:
            if os.path.exists(rank0_test_file):
                try:
                    os.remove(rank0_test_file)
                except OSError:
                    pass

        check_failures = [
            res for res in check_results
            if not isinstance(res, dict) or res.get('status') == 'error'
        ]
        if check_failures:
            first_err = (
                check_failures[0].get('error', repr(check_failures[0]))
                if isinstance(check_failures[0], dict)
                else repr(check_failures[0])
            )
            wrk = (
                check_failures[0].get('worker_rank', '?')
                if isinstance(check_failures[0], dict)
                else '?'
            )
            raise RuntimeError(
                f"\n[MPI Shared Filesystem Error] Worker rank {wrk} failed "
                f"to access or write to working directory "
                f"'{work_dir_str}':\n  {first_err}\n\n"
                f"On multi-node HPC/cluster jobs, intermediate files must "
                f"be created on a shared parallel filesystem (such as "
                f"Lustre, GPFS, or NFS) visible to all nodes, rather than "
                f"node-local /tmp.\n\n"
                f"--> ACTION REQUIRED: Set the TMPDIR environment variable "
                f"to point to your shared network filesystem before "
                f"running OCSMesh:\n"
                f"    export TMPDIR=/scratch/$USER\n"
                f"    # or export TMPDIR=/lustre/..."
            )
        _logger.info("Shared filesystem check passed across all worker ranks.")

    # ── Internal ──────────────────────────────────────────────────

    def _dispatch(self, tasks, fail_fast=False):
        """Rank-0-only: stream tasks dynamically to idle workers.

        Seeds one task per worker, then refills each worker as soon as
        it returns a result — a central work queue with on-demand
        assignment. Returns the list of result/error dicts (order is
        NOT the task order; use ``'original_index'`` to reassociate).

        Parameters
        ----------
        tasks : list of dict
            Self-describing task dicts, each with an ``'op'`` key.
        fail_fast : bool, default=False
            If True, stop dispatching new tasks as soon as the first
            error result is received. In-flight tasks are still drained.

        Returns
        -------
        list of dict
            One result (or structured error) dict per task dispatched.
        """
        # ── Safety checks ──
        if self.comm is None:
            raise RuntimeError("mpi4py is not available")
        if self.rank != 0:
            raise RuntimeError(
                "_dispatch() must only be called by Rank 0."
            )
        if not isinstance(tasks, list):
            raise TypeError(
                f"_dispatch() expects a list of task dicts, "
                f"got {type(tasks).__name__}"
            )

        # Degenerate case: no workers (size == 1)
        # Run tasks locally so single-rank runs stay correct.
        if self.size == 1:
            return self._run_tasks_locally(tasks)

        results = []
        task_iter = iter(tasks)
        inflight = 0  # tasks currently being processed by workers
        saw_error = False  # fail_fast: stop dispatching after first error

        # ── Seed: give each worker one task to start ──
        for worker in range(1, self.size):
            task = next(task_iter, None)
            if task is None:
                break  # fewer tasks than workers
            self.comm.send(task, dest=worker, tag=self._TAGS['TASK'])
            inflight += 1

        # ── Refill: as each worker reports back, hand it the next ──
        MPI = _get_mpi()
        while inflight > 0:
            status = MPI.Status()
            message = self.comm.recv(
                source=MPI.ANY_SOURCE,
                tag=MPI.ANY_TAG,
                status=status,
            )
            worker = status.Get_source()
            inflight -= 1
            results.append(message)

            # Check if this result is an error
            if (isinstance(message, dict)
                    and message.get('status') == 'error'):
                saw_error = True

            # Only dispatch new tasks if we haven't seen an error
            # (or fail_fast is disabled). In-flight tasks continue
            # to drain normally — we just stop sending NEW work.
            if fail_fast and saw_error:
                _logger.warning(
                    f"fail_fast: error received from worker rank "
                    f"{worker}; draining {inflight} in-flight task(s) "
                    f"and skipping remaining queue."
                )
                continue

            next_task = next(task_iter, None)
            if next_task is not None:
                self.comm.send(
                    next_task, dest=worker, tag=self._TAGS['TASK'])
                inflight += 1

        return results

    def _run_worker(self):
        """Worker recv/execute/send loop (point-to-point).

        Each non-zero rank calls this ONCE. The worker blocks in a
        ``recv`` until rank 0 either hands it a task (``TAG_TASK``) or
        tells it to stop (``TAG_STOP``). After running a task it sends
        the result back and loops again.

        A blocking ``recv`` is the correct idle-wait primitive: the
        worker sleeps inside MPI without busy-polling and is woken only
        when a message addressed to it arrives.
        """
        # ── Safety checks ──
        if self.rank == 0:
            raise RuntimeError(
                "_run_worker() must only be called by worker ranks."
            )

        MPI = _get_mpi()
        registry = self._worker_registry()
        _logger.debug(
            f"Rank {self.rank}: entering point-to-point worker loop"
        )

        while True:
            # Block until rank 0 sends us something. ANY_TAG lets a
            # single recv distinguish a task from a stop signal.
            status = MPI.Status()
            message = self.comm.recv(
                source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()

            if tag == self._TAGS['STOP']:
                _logger.debug(
                    f"Rank {self.rank}: received STOP, leaving loop"
                )
                break

            if tag != self._TAGS['TASK']:
                # Defensive: unknown control tag. Report and keep
                # serving so a protocol slip does not silently hang.
                _logger.warning(
                    f"Rank {self.rank}: unexpected tag {tag!r}; "
                    f"ignoring message"
                )
                continue

            task = message
            op = task.get('op') if isinstance(task, dict) else None
            worker_fn = registry.get(op)

            if worker_fn is None:
                # Unknown operation -> structured error, worker stays.
                self.comm.send(
                    {
                        'status': 'error',
                        'original_index': (
                            task.get('original_index', -1)
                            if isinstance(task, dict) else -1),
                        'op': op,
                        'worker_rank': self.rank,
                        'error': f"No worker registered for op "
                                 f"{op!r}",
                        'traceback': '',
                    },
                    dest=0, tag=self._TAGS['ERROR'],
                )
                continue

            try:
                result = worker_fn(task)
                # Some worker functions catch their own exceptions
                # and return {'status': 'error', ...}. Route those
                # through the error channel too.
                if (isinstance(result, dict)
                        and result.get('status') == 'error'):
                    result.setdefault('worker_rank', self.rank)
                    self.comm.send(
                        result, dest=0, tag=self._TAGS['ERROR'])
                else:
                    self.comm.send(
                        result, dest=0, tag=self._TAGS['RESULT'])
            except Exception as exc:  # pylint: disable=broad-exception-caught
                # Task blew up — NOT an MPI rank failure. Catch it,
                # report it, keep the worker available for more tasks.
                self.comm.send(
                    {
                        'status': 'error',
                        'original_index': (
                            task.get('original_index', -1)
                            if isinstance(task, dict) else -1),
                        'op': op,
                        'worker_rank': self.rank,
                        'error': repr(exc),
                        'traceback': traceback.format_exc(),
                    },
                    dest=0, tag=self._TAGS['ERROR'],
                )

        _logger.debug(
            f"Rank {self.rank}: worker loop exited cleanly"
        )

    def _shutdown_workers(self):
        """Send TAG_STOP to each worker individually.

        Individual sends (not a collective) mean each worker exits
        independently — if one already crashed, the others still
        receive their stop signal cleanly.
        """
        _logger.debug(
            f"Rank 0: sending STOP to {self.size - 1} worker(s)"
        )
        for worker in range(1, self.size):
            self.comm.send(None, dest=worker, tag=self._TAGS['STOP'])

    def _run_tasks_locally(self, tasks):
        """Fallback for single-rank: run tasks without MPI."""
        registry = self._worker_registry()
        results = []
        for task in tasks:
            fn = registry.get(task.get('op'))
            if fn is None:
                results.append({
                    'status': 'error',
                    'original_index': task.get('original_index', -1),
                    'error': f"No worker registered for op "
                             f"{task.get('op')!r}",
                })
                continue
            try:
                results.append(fn(task))
            except Exception as exc:  # pylint: disable=broad-exception-caught
                results.append({
                    'status': 'error',
                    'original_index': task.get('original_index', -1),
                    'error': repr(exc),
                    'traceback': traceback.format_exc(),
                })
        return results
