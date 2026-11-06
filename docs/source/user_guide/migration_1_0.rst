.. _migration_1_0:

**********************************
Migrating from 0.x to 1.0
**********************************

xarray-spatial 1.0.0 removes the keyword and function aliases that 0.10.x had
deprecated. Each one warned with a ``DeprecationWarning`` that named its
replacement; from 1.0.0 the old spelling raises ``TypeError`` (for keyword
arguments) or ``AttributeError`` / ``ImportError`` (for functions).

Everything else on the public surface is unchanged. If your code ran on
0.10.20 without warnings, it runs on 1.0.0.

Removed aliases
===============

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - Removed in 1.0.0
     - Use instead
     - Deprecated since
   * - ``xrspatial.basins`` / ``xrspatial.hydro.basins`` / ``basins_d8``
     - ``basin`` / ``basin_d8``
     - 0.10.x
   * - ``crop(zones_ids=...)``
     - ``crop(zone_ids=...)``
     - 0.10.x
   * - ``focal.apply(raster=...)`` and ``hotspots(raster=...)``
     - ``agg=``
     - 0.10.x
   * - ``natural_breaks(agg, k, name, num_sample)`` positional ``num_sample``
     - ``natural_breaks(agg, k, num_sample=...)``
     - 0.10.x
   * - ``rasterize(use_cuda=...)``
     - ``rasterize(gpu=...)``
     - 0.10.x
   * - ``reproject(src_vertical_crs=..., tgt_vertical_crs=...)``
     - ``source_vertical_crs=``, ``target_vertical_crs=``
     - 0.10.x
   * - ``open_geotiff(mask_nodata=...)``
     - ``open_geotiff(masked=...)``. Note the default is ``False``.
     - 0.10.x
   * - ``open_geotiff(name=...)``
     - ``open_geotiff(default_name=...)``
     - 0.10.x
   * - ``open_geotiff(mask_and_scale=...)``
     - ``open_geotiff(unpack=...)``
     - 0.10.x
   * - ``plot_geotiff(...)``
     - ``da.xrs.plot(...)``
     - 0.10.0
   * - ``owa(criterion_weights=...)``
     - ``owa(weights=...)``
     - 0.10.x

Two private helpers also lost old spellings: the VRT writer's ``vrt_path=``
and ``crs_wkt=`` (now ``path=`` and ``crs=``) and the GPU reader's
``gpu='auto'`` / ``gpu='strict'`` (now ``on_gpu_failure=``). Neither was
part of the public surface; ``open_geotiff`` takes ``gpu=True`` and
``on_gpu_failure='auto'|'strict'`` as before.

Finding the old spellings
=========================

Run your test suite on 0.10.20 with warnings turned into errors:

.. code-block:: bash

    python -W error::DeprecationWarning -m pytest

Every failure points at a line in the table above.

See the :ref:`stability policy <user_guide.stability_policy>` for how
deprecations work from 1.0 on.
