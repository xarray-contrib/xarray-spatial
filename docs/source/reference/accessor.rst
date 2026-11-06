.. _reference.accessor:

*********************
The ``.xrs`` accessor
*********************

``import xrspatial`` registers an ``xrs`` accessor on ``xarray.DataArray``
and ``xarray.Dataset``. Every method forwards to the top-level function of
the same name with the array as the first argument, so
``dem.xrs.slope()`` is ``xrspatial.slope(dem)``.

The accessor is part of the 1.x LTS surface (see
:ref:`user_guide.stability_policy`). Methods that forward to
``xrspatial.experimental`` say so in their docstring and are not covered.

.. code-block:: python

    import xrspatial

    dem = xrspatial.open_geotiff("dem.tif")
    dem.xrs.slope()
    dem.xrs.hillshade().xrs.to_geotiff("hillshade.tif")

DataArray accessor
==================

.. autoclass:: xrspatial.accessor.XrsSpatialDataArrayAccessor
   :members:
   :undoc-members:

Dataset accessor
================

.. autoclass:: xrspatial.accessor.XrsSpatialDatasetAccessor
   :members:
   :undoc-members:
