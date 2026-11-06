## What this changes and why

## How it was tested
- [ ] A test that fails without the change (bug fix) or covers the new behaviour
- [ ] `python -m unittest discover tests`
- [ ] For changes to the parallel/MPI paths: the multi-tile tests under `mpiexec -n 3` (see CONTRIBUTING.md)
- [ ] `pylint --disable=too-many-positional-arguments ocsmesh` (score >= 9.90)

## Checklist
- [ ] Entry in the `Unreleased` section of `CHANGELOG.md`
- [ ] Documentation updated (docstrings, `docs/source`, README) if user-facing
