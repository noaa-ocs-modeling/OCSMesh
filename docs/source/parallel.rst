Parallel and MPI execution
==========================

Large size functions are built from many DEM tiles. ``Hfun`` can
compute them serially, with a pool of processes on one node, or with
MPI across nodes. The result is the same in every mode; only the time
changes.

Execution modes
---------------

.. list-table::
   :header-rows: 1
   :widths: 15 40 45

   * - mode
     - how the tiles are processed
     - use it for
   * - ``'serial'`` (default)
     - one after the other, in this process
     - small domains, debugging
   * - ``'parallel'``
     - a ``multiprocessing`` pool of ``nprocs`` workers on this node
     - a workstation or a single HPC node
   * - ``'mpi'``
     - MPI ranks under ``mpiexec``/``srun``: rank 0 coordinates and the
       other ranks process one tile at a time
     - many tiles, more than one node

The mode applies to the ``exact`` method, where every tile gets its own
size function. The ``fast`` method merges the tiles into one raster and
always runs on one process.

.. code-block:: python

    import ocsmesh

    hfun = ocsmesh.Hfun(
        tile_paths, hmin=50, hmax=3000, method='exact',
        base_shape=domain, base_shape_crs='EPSG:4326',
        nprocs=40, execution_mode='parallel')
    hfun.add_contour(level=0, expansion_rate=0.01, target_size=100)
    hfun.add_channel(level=0, width=1000, target_size=200)
    sizes = hfun.meshdata()

``execution_mode`` can also be set after construction
(``hfun.execution_mode = 'mpi'``).

What is distributed
-------------------

Each stage of the ``exact`` method runs one task per tile: constant
values and depth bands, flow limiters, contours and channels (detection
and refinement), patches, line features, constraints and the final
per-tile size function. Stages with nothing to apply are skipped. The
tiles are then combined in priority order on the coordinator: a later
input wins where tiles overlap, but only where it has valid data.

A stage is as slow as its slowest tile. For instance, channel detection
on one large or complex tile can dominate a run; splitting such a tile
helps more than adding ranks.

Running with MPI
----------------

Install the extra (``pip install ocsmesh[mpi]``) or ``mpi4py`` from
conda-forge, built against the MPI of the cluster. Every rank runs the
same script, so the script must build the ``Hfun`` (and call
``meshdata()``) on all ranks; the result is returned on rank 0 and
``None`` on the other ranks:

.. code-block:: python

    hfun = ocsmesh.Hfun(tile_paths, hmin=50, hmax=3000,
                        execution_mode='mpi')
    hfun.add_contour(level=0, expansion_rate=0.01, target_size=100)
    sizes = hfun.meshdata()           # collective: every rank calls it
    if sizes is not None:             # rank 0
        mesh = ocsmesh.MeshDriver(geom, hfun).run()

.. code-block:: bash

    # one rank per tile plus the coordinator
    srun --ntasks=19 --cpus-per-task=1 python build_mesh.py
    mpiexec -n 19 python build_mesh.py

Ranks beyond the number of tiles plus one stay idle. Without a launcher,
or with a single rank, ``'mpi'`` falls back to ``'parallel'`` with a
warning.

To run an independent serial or parallel ``Hfun`` on every rank (for
example one domain per rank), pass ``execution_mode='serial'`` or
``'parallel'`` to the constructor; such a collector makes no collective
calls.

Temporary files
---------------

Tasks exchange size functions through files under ``$TMPDIR`` (default
``/tmp``). For an MPI job on more than one node, ``TMPDIR`` must be on
a filesystem that every node can see (Lustre, GPFS, NFS), e.g.:

.. code-block:: bash

    export TMPDIR=/path/on/shared/scratch/$SLURM_JOB_ID
    mkdir -p "$TMPDIR"

The directory also needs room for a few copies of the clipped tiles.

Environment set at import
-------------------------

When a launcher started more than one task, ``import ocsmesh``:

* sets ``OMP_NUM_THREADS``, ``MKL_NUM_THREADS`` and
  ``OPENBLAS_NUM_THREADS`` to 1, unless they are already set, so that
  ranks sharing a node do not oversubscribe its cores;
* sets the ``multiprocessing`` start method to ``spawn``, because
  forking a process that holds an MPI communicator can deadlock.

Nothing is changed in serial jobs, including single-task SLURM jobs.
Import ``ocsmesh`` before creating any ``multiprocessing`` pool.

Errors
------

A task that fails stops the run with a ``RuntimeError`` listing every
failed tile and its traceback; no partial size function is returned.
