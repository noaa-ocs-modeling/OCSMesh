# Contributing to OCSMesh

Bug reports, questions and pull requests are welcome. Please open an
[issue](https://github.com/noaa-ocs-modeling/OCSMesh/issues) first for
larger changes, so the approach can be discussed.

## Development setup

```bash
git clone https://github.com/noaa-ocs-modeling/OCSMesh
cd OCSMesh
conda env create -f environment.yml -n ocsmesh-dev
conda activate ocsmesh-dev
pip install -e .[testing]
```

`environment.yml` includes `mpi4py` and an MPI library from conda-forge,
so the MPI tests can run locally.

## Branches and pull requests

- Branch from `dev` and open the pull request against `dev`; `main`
  only receives releases.
- Keep one topic per commit, with a message that says what changed and
  why (e.g. `fix(hfun): ...`, `perf(mesh): ...`, `test: ...`).
- Add an entry to the `Unreleased` section of `CHANGELOG.md`.

## Tests

A bug fix comes with a test that fails without the fix; a feature comes
with tests of its behaviour. Run the suite before opening the pull
request:

```bash
python -m unittest discover tests            # everything, serial (~6 min)
python -m pytest tests/api/test_hfun.py      # one file
```

Code that runs differently in serial, parallel and MPI modes must give
the same result in all of them. The tests built on
`tests/api/multitile.py` check this; they also run under MPI, where
they compare against MPI mode as well:

```bash
FI_PROVIDER=tcp mpiexec -n 3 python -m pytest tests/api/test_hfun_clipped_equivalence.py
FI_PROVIDER=tcp mpiexec -n 2 python -m pytest tests/api/mpi_failures_handling_test.py
```

Under MPI, every rank must reach the same collective calls (creating an
`HfunCollector`, `meshdata()`, the refinement stages): build objects on
all ranks and check results on rank 0 only (`multitile.is_manager()`).

## Code style

CI runs `pylint --disable=too-many-positional-arguments ocsmesh` with the
repository's `.pylintrc` and fails below a score of 9.90 or on any
error. Use numpy-style docstrings for public functions.

## Documentation

The documentation is built with Sphinx from `docs/source` and published
on releases:

```bash
pip install .[documentation]
cd docs && make html     # output in docs/build/html
```
