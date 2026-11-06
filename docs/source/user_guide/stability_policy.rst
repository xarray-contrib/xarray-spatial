..  _user_guide.stability_policy:

Stability policy and LTS commitment
===================================

This page states which parts of xarray-spatial are covered by long-term
support (LTS) in the 1.x series, how deprecations work, and what "stable"
means here.

Versioning
----------

xarray-spatial follows semantic versioning from 1.0.0 on.

- Patch releases (1.0.1, 1.0.2, ...) fix bugs and do not change behaviour
  for documented inputs.
- Minor releases (1.1.0, 1.2.0, ...) may add functions, add optional keyword
  arguments, promote a function to a higher tier, raise a dependency floor,
  and deprecate names. They do not remove or rename anything on the LTS
  surface.
- Major releases (2.0.0) are the only place where deprecated names are
  removed or where a documented contract changes incompatibly.

What LTS covers
---------------

The high-level public spellings are the LTS surface:

- every name listed in ``xrspatial.__all__`` (the names you get from
  ``import xrspatial``),
- the public names of the ``xrspatial.mcda`` subpackage,
- the ``.xrs`` accessor on ``xarray.DataArray`` and ``xarray.Dataset`` (see
  :ref:`reference.accessor`).

For the whole 1.x series these names, their call signatures, and their
documented return contracts do not change in backward-incompatible ways.
New optional keyword arguments may be added; existing behaviour for
documented inputs is preserved.

Functions that live only in a submodule (for example
``xrspatial.experimental.min_observable_height``) and anything with a leading
underscore are not part of the LTS surface and may change in a minor release.

Deprecation window
------------------

When a name or keyword argument on the LTS surface is replaced:

1. The old spelling keeps working and emits a ``DeprecationWarning`` that
   names the replacement and the release that will remove it.
2. The warning ships in at least one minor release before removal.
3. Removal happens in the next major release, never in a minor or patch.

The 0.x aliases that 1.0.0 removed are listed on the
:ref:`migration_1_0` page.

Dependencies
------------

The minimum supported Python and dependency versions are the ones pinned in
the ``min-deps`` job of the test workflow. Raising a floor is a minor-release
change and is called out in the changelog. The package declares an upper
bound on NumPy only; new NumPy minors are tested and the bound lifted in a
patch or minor release.

What "stable" requires
----------------------

Following the feature-tier work in #2415, a function reaching the ``stable``
tier needs a documented contract and a release gate, not green cross-backend
tests alone. Concretely:

- a documented input/output contract (dtypes, dims, attrs, nodata handling),
- no open CRITICAL or HIGH ``/deep-sweep`` findings in accuracy or security,
- a notebook example.

Tiers below stable
------------------

``advanced`` and ``experimental`` functions work and are tested but carry
caveats or make no cross-backend parity claim. GPU and some Dask paths
usually stay at these tiers. The README feature matrix is the per-backend
source of truth for each function's tier.

Changing a tier
---------------

A tier moves in a pull request that edits the README matrix cell and states
the reason. Promotion can happen in any release. Demoting a cell from
``stable`` is an incompatible contract change and waits for a major release;
until then the behaviour stays and the caveat goes in the docs.
