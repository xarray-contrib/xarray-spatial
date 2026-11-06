"""Internal VRT writer.

Wraps ``_vrt.write_vrt`` with the ``_build_vrt`` helper: normalisation of
the ``crs`` kwarg to WKT via ``_resolve_crs_to_wkt``, and the parity
surface vs ``to_geotiff`` / ``_write_geotiff_gpu``. ``to_geotiff``'s
``.vrt`` write path is the only caller; it is not part of the public API.
"""
from __future__ import annotations

from .._crs import _resolve_crs_to_wkt
from .._validation import _validate_nodata_arg


def _build_vrt(path: str,
               source_files: list[str], *,
               relative: bool = True,
               crs: int | str | None = None,
               nodata: float | int | None = None,
               dst_offsets: list[tuple[int, int]] | None = None) -> str:
    """Generate a VRT file that mosaics multiple GeoTIFF tiles.

    [internal-only] This helper backs ``to_geotiff``'s ``.vrt`` write
    path and is not part of the public API. VRT mosaic output targets a
    narrow subset of GDAL's VRT spec; a consumer reading the resulting
    ``.vrt`` may hit cross-source nodata mismatch, missing backing files,
    or per-band metadata disagreement. Full GDAL VRT parity, warped /
    reprojection VRTs, and nested VRTs are out of scope for this release.
    See :data:`xrspatial.geotiff.SUPPORTED_FEATURES` for the full tier map.

    Output targets the same narrow subset of GDAL's VRT spec that the
    reader supports (see the "VRT support matrix" section in
    ``docs/source/reference/geotiff.rst`` and the audited matrix in
    ``docs/source/reference/release_gate_geotiff.rst`` for the
    canonical contract):

    * Supported: simple GDAL VRT mosaics over GeoTIFF sources;
      compatible CRS, transform orientation, pixel size, dtype, and
      band count across sources; clean windowed reads on the
      consumer side; lazy / dask reads over the same subset on the
      consumer side; explicit nodata; ``missing_sources='raise'`` as
      the read-side default.
    * Non-goals (the writer does not emit these and the reader is
      allowed to raise on them): warped / reprojection VRTs,
      arbitrary resampling beyond the tested subset, mixed CRS /
      resolution / dtype / band metadata without an opt-in, nested
      VRTs, complex source / mask band / alpha band structures, full
      GDAL VRT parity.

    Parameters
    ----------
    path : str
        [internal-only] Output .vrt file path. Mirrors the ``path`` kwarg
        on ``to_geotiff`` and ``_write_geotiff_gpu`` so the writer trio
        shares a single destination-arg name.
    source_files : list of str
        [internal-only] Paths to the source GeoTIFF files.
    relative : bool, optional
        [internal-only] Store source paths relative to the VRT file
        (default True).
    crs : int, str, or None, optional
        [internal-only] EPSG code (int), WKT string, or PROJ string. If
        None, the CRS is taken from the first source GeoTIFF. Mirrors
        the ``crs`` kwarg on ``to_geotiff`` and ``_write_geotiff_gpu``
        so the same value can be forwarded to whichever writer the
        caller picked without per-writer special-casing.
    nodata : float, int, or None, optional
        [internal-only] NoData value. If None, taken from the first source
        GeoTIFF.
        Integer sentinels (e.g. ``65535`` for uint16, ``-9999`` for
        int32) are accepted so the surface lines up with the
        ``nodata`` kwarg on ``to_geotiff`` and ``_write_geotiff_gpu``.
    dst_offsets : list of (int, int) or None, optional
        [internal-only] Explicit pixel-space placement for
        non-georeferenced sources, one ``(x_off, y_off)`` pair per
        source file. Forwarded to ``_vrt.write_vrt``; see its docstring
        for the contract. ``to_geotiff``'s ``.vrt`` path passes this
        for non-georeferenced input so the index places each tile by
        its pixel offset instead of the identity transform every such
        tile carries (issue #3116).

    Returns
    -------
    str
        Path to the written VRT file.

    Examples
    --------
    Safe usage. Mosaic two compatible tiles; the consumer can then
    read the resulting VRT with the fail-closed defaults. Paths
    below are illustrative; replace with paths to real GeoTIFF
    files on disk:

    >>> from xrspatial.geotiff import _build_vrt, open_geotiff
    >>> vrt_path = _build_vrt(  # doctest: +SKIP
    ...     'mosaic.vrt',
    ...     source_files=['tile_west.tif', 'tile_east.tif'],
    ... )
    >>> da = open_geotiff(vrt_path)  # doctest: +SKIP

    Intentionally raises (on the read side). If the source tiles
    disagree on their per-band nodata sentinels, the default
    ``band_nodata=None`` on ``open_geotiff`` / ``_read_vrt`` rejects
    the mosaic with ``MixedBandMetadataError``. The writer does not
    pre-validate cross-tile metadata; the failure mode lives on the
    read side:

    >>> from xrspatial.geotiff import MixedBandMetadataError
    >>> # tile_a.tif declares nodata=-9999; tile_b.tif declares nodata=0
    >>> bad_path = _build_vrt(  # doctest: +SKIP
    ...     'mixed_nodata.vrt',
    ...     source_files=['tile_a.tif', 'tile_b.tif'],
    ... )
    >>> try:  # doctest: +SKIP
    ...     open_geotiff(bad_path)
    ... except MixedBandMetadataError:
    ...     pass  # fix the source tiles or pass band_nodata='first'.
    """
    # Explicit signature (previously ``**kwargs``) so ``inspect.signature``,
    # IDE autocomplete, and ``mypy --strict`` can see the accepted kwargs
    # without parsing the docstring. ``crs`` normalises through
    # ``_resolve_crs_to_wkt`` before forwarding because the internal
    # writer still only speaks WKT.
    if path is None:
        # Explicit ``path=None`` (including positional ``_build_vrt(None,
        # sources)``) is rejected up front so the error message names the
        # offending kwarg instead of crashing deep in
        # ``os.path.dirname(os.path.abspath(None))``.
        raise TypeError(
            "_build_vrt: 'path' must be a str, got None")
    if source_files is None:
        raise TypeError(
            "_build_vrt: missing required argument 'source_files'")

    # Reject bool / non-numeric nodata at the entry point so _build_vrt
    # matches the to_geotiff / _write_geotiff_gpu surface. ``bool`` is a
    # subclass of ``int`` in Python, so a typo like ``nodata=True`` would
    # slip past every downstream ``isinstance(nodata, (int, float))``
    # guard and the VRT XML emitter would write ``<NoDataValue>True
    # </NoDataValue>``. No reader parses ``"True"`` as numeric, so the
    # round-trip would silently drop the sentinel.
    _validate_nodata_arg(nodata)

    resolved_wkt = _resolve_crs_to_wkt(crs)

    from .._vrt import write_vrt as _write_vrt_internal
    return _write_vrt_internal(
        path, source_files,
        relative=relative,
        crs_wkt=resolved_wkt,
        nodata=nodata,
        dst_offsets=dst_offsets,
    )
